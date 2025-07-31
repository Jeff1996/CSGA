'''
架构说明: windows transformer + Transforemr-VQ, 用于验证Transformer-VQ的: 码本利用率低(单样本推理过程激活的向量很少), 训练不稳定(码本模式崩溃), 使用欧氏距离作为相似性度量不合理, 性能不如CSGA
1. 向量量化采用欧氏距离
2. 采用EMA更新码本
3. 层级堆叠方式参考swin_debug.py
4. 引入量化损失

'''
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.autograd import Function

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmpretrain.registry import MODELS
from ..utils.embed_seg import PatchEmbed, PatchMerging
from ..utils.vqt_tools import reduce_sum

import os

# # 使用Flash-Attention 2.x的API
# from flash_attn import flash_attn_func

# 自定义梯度传播过程
class vecQuantization(Function):  
    @staticmethod  
    def forward(ctx, x, c):
        # # 前向传播（基于余弦相似度/单位化点积）
        # # x = x - x.mean(dim=(0, 2), keepdim=True)                            # 这里可以改为EMA更新的均值
        # # x = x / torch.norm(x, dim=-1, keepdim=True)
        # # codebook = c - c.mean(dim=-2, keepdim=True)
        # # codebook = codebook / torch.norm(codebook, dim=-1, keepdim=True)
        # cosSim = torch.einsum('bhld,hsd->bhls', x, c)                       # 相似度矩阵, [B, heads, L, S]
        # delta_index = cosSim.argmax(dim=-1)                                 # 索引矩阵, [B, heads, L]
        # delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()          # one-hot索引矩阵, [B, heads, L, S]

        # 前向传播（基于欧式距离）
        x2 = torch.sum(torch.square(x), dim=-1).unsqueeze(-1)               # [B, heads, L, d] -> [B, heads, L] -> [B, heads, L, 1]
        xc = torch.einsum('bhld,hsd->bhls', x, c)                           # [B, heads, L, S]
        c2 = torch.sum(torch.square(c), dim=-1).unsqueeze(1).unsqueeze(0)   # [heads, S, d] -> [heads, S] -> [heads, 1, S] -> [1, heads, 1, S]
        distance2 = x2 - 2*xc + c2                                          # 待量化序列中每一个向量与码表中每一个向量的欧氏距离的平方, [B, L, S]
        delta_index = distance2.argmin(dim=-1)                              # 索引矩阵, [B, heads, L]
        delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()           # one-hot索引矩阵, [B, heads, L, S]

        # print('------------ 调试点 ------------')
        # B, heads, L = delta_index.shape
        # num_tokens = B * heads * L
        # for index in torch.unique(delta_index):
        #     print('{}: {:.3f}%'.format(index, (delta_index==index).float().sum() / num_tokens * 100))

        # 保存必要的中间变量用于梯度反向传播
        ctx.save_for_backward(delta_onehot)

        return delta_onehot, c                                              # forward的输出个数与backward的输入个数相同
  
    @staticmethod  
    def backward(ctx, grad_output_delta_onehot, grad_output_c):  
        # print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('c', grad_output_c.min(), grad_output_c.mean(), grad_output_c.max()))
        # 获取中间变量
        (delta_onehot, ) = ctx.saved_tensors

        # # --- 梯度反传方案1, 码本通过EMA更新, 不需要梯度 ---
        # # 来自码本的梯度
        # grad_x = torch.einsum('bhls,hsd->bhld', delta_onehot, grad_output_c)# 来自码表c的梯度
        # # 来自分配矩阵Δ的梯度
        # # 被废弃了，因为会导致梯度爆炸

        # --- 梯度反传方案2，需要归一化码本梯度：具体而言，码表中的某个向量被多少个token调用，就要复制多少次梯度，如果不进行归一化，将会发生梯度爆炸 ---
        delta_onehot_sum = torch.einsum('bhls->hs', delta_onehot).unsqueeze(dim=-1).clip(min=1.0)   # bhls -> hs -> hs1
        grad_output_c_norm = grad_output_c / delta_onehot_sum
        grad_x = torch.einsum('bhls,hsd->bhld', delta_onehot, grad_output_c_norm)   # 来自码表c的梯度

        return grad_x, None                                                 # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None

# 向量量化器
class Quantizer(nn.Module):
    '''
    序列量化器
    1. 保存一个可更新的量化码表C
    2. 构造量化的K序列K^
    3. 获取输入矩阵K量化后的索引矩阵Δ
    '''
    def __init__(self, heads: int, codes: int, dim: int, ema: bool = True):
        '''
        codes   : 码表中行向量的个数
        dim     : 码表每个行向量的维数(与K矩阵的维数一致)
        '''
        super(Quantizer, self).__init__()
        self.ema = ema                                                          # 是否采用EMA更新码表
        c_sum_init = self.initCodebook(heads, codes, dim)                       # 码本初始化（完全参考Transformer-VQ）
        c_count_init = torch.ones(heads, codes)                                 # 初始状态下，规定码本中的每个向量由一个元素组成

        if ema:
            # 使用EMA更新的码表
            self.c_sum = nn.Parameter(c_sum_init, requires_grad=False)          # 汇总的元素集合
            self.c_count = nn.Parameter(c_count_init, requires_grad=False)      # 汇总的元素集合

            # # 也可以考虑使用register_buffer方法定义上述两个不需要梯度更新的参数
            # self.register_buffer('c_sum', c_sum_init)
            # self.register_buffer('c_count', c_count_init)

            c_sum_add = torch.zeros(heads, codes, dim)                          # 用于累计更新量
            self.c_sum_add = nn.Parameter(c_sum_add, requires_grad=False)

            c_count_add = torch.zeros(heads, codes)                             # 用于归一化更新量
            self.c_count_add = nn.Parameter(c_count_add, requires_grad=False)

            self.c_count = nn.Parameter(c_count_add, requires_grad=False)       # 用于统计整个更新过程，码表中的每个向量被更新的次数

            self.update_count = 1                                               # 更新量累计次数统计
            self.update_interval = 1                                            # 码表更新间隔(码表两次更新间的更新量累计次数)
        else:
            # 使用梯度更新的码表
            self.c = nn.Parameter(c_sum_init)

        self.gamma = 0.99                                                       # EMA超参数（历史成分占比）
        self.vecQuantization = vecQuantization()                                # 自定义梯度反传过程的向量量化函数

    def stopGradient(self, x: torch.tensor):
        '''
        梯度停止函数(stop gradient)
        '''
        return x.detach()
    
    def STE(self, value_forward: torch.tensor, grad_backward: torch.tensor):
        '''
        梯度直传函数(Straight-Through Estimator)
        解决由于argmin操作导致的梯度中断，
        前向传播时使用value_forward变量值，
        反向传播时grad_backward变量将继承value_forward变量的梯度

        输入：
        value_forward: 前向传播提供值，反向传播被继承梯度
        grad_backward: 前向传播无贡献，反向传播继承value_forward的梯度
        '''
        assert value_forward.shape == grad_backward.shape, "value_forward and grad_backward have different shapes!"
        return grad_backward + self.stopGradient(value_forward - grad_backward)

    def initCodebook(self, heads: int, s: int, dim: int, start: int=0, lam: int=10000):
        '''
        参考Transformer-VQ, 初始化码本
        '''
        pos_seq = start + torch.arange(s)
        inv_lams = 1 / (lam ** (torch.arange(0, dim, 2) / dim))       # Transformer原生的正余弦位置编码
        pre = pos_seq[..., None] * inv_lams[None, ...]
        sin = torch.sin(pre)
        cos = torch.cos(pre)
        codebook = torch.concatenate([sin, cos], axis=-1)
        codebook = (dim**-0.25) * codebook[None, ...]
        codebook = codebook.repeat(heads, 1, 1)
        return codebook

    def getCodebook(self):
        '''
        返回归一化的码表
        c: [heads, codes, dim]
        '''
        if self.ema:
            c = self.c_sum / torch.clip(self.c_count[..., None], 1e-6)          # EMA更新的码本(不在是nn.Parameter类, 变成了普通tensor)
        else:
            c = self.c.data                                                     # 梯度更新的码本
        return c

    def emaAccumulate(self, delta_onehot, x):
        '''
        累计码表更新量
        更新量累计在量化之后，码表更新在量化之前，由此才不会影响梯度反向传播
        '''
        c_sum_add = torch.einsum('bhls,bhld->hsd', delta_onehot, x)             # [heads, S, dim]
        c_count_add = torch.einsum('bhls->hs', delta_onehot)                    # [heads, S]
        try:
            reduce_sum(c_sum_add)                                               # 多卡同步
            reduce_sum(c_count_add)                                             # 多卡同步
        except:
            pass
        self.c_sum_add.data = self.c_sum_add + c_sum_add                        # 累计
        self.c_count_add.data = self.c_count_add + c_count_add                  # 累计
        self.c_count.data = self.c_count + c_count_add                          # 测试用参数，不清零

    def updateCodebook(self):
        '''
        EMA更新码表
        delta_onehot: tensor, [batch size, heads, L, S], 量化K的one-hot索引矩阵Δ
        x           : tensor, [batch size, heads, L, dim], K矩阵
        '''
        # 计算更新量（单位化）
        c_sum_add = self.c_sum_add.data                                         # [heads, S, dim]
        c_count_add = torch.clip(self.c_count_add.data, min=1e-6)                # [heads, S], clip操作防止除以0, 但是又不能设得太大, 否则长期没有成员的码本向量会越来越趋近于0向量

        # 更新
        self.c_sum.data = self.c_sum * self.gamma + c_sum_add * (1 - self.gamma)
        self.c_count.data = self.c_count * self.gamma + c_count_add * (1 - self.gamma)

        # 清空累计
        self.c_sum_add.fill_(0.)
        self.c_count_add.fill_(0.)

    # 测试用的聚类分配程序，没有设计梯度反向传播规则
    def clustering(self, x, c):
        # 前向传播（基于欧式距离）
        x2 = torch.sum(torch.square(x), dim=-1).unsqueeze(-1)               # [B, heads, L, d] -> [B, heads, L] -> [B, heads, L, 1]
        xc = torch.einsum('bhld,hsd->bhls', x, c)                           # [B, heads, L, S]
        c2 = torch.sum(torch.square(c), dim=-1).unsqueeze(1).unsqueeze(0)   # [heads, S, d] -> [heads, S] -> [heads, 1, S] -> [1, heads, 1, S]
        distance2 = x2 - 2*xc + c2                                          # 待量化序列中每一个向量与码表中每一个向量的欧氏距离的平方, [B, L, S]
        delta_index = distance2.argmin(dim=-1)                              # 索引矩阵, [B, heads, L]
        delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()           # one-hot索引矩阵, [B, heads, L, S]
        delta_onehot.requires_grad_()
        c.requires_grad_()
        return delta_onehot, c

    def forward(self, x):
        '''
        输入
        x               : tensor, [batch size, heads, L, dim], K矩阵

        输出
        delta_onehot    : tensor, [batch size, heads, L , S], 量化K的索引矩阵Δ
        c               : tensor, [heads, S, dim], 量化码表
        '''
        # 更新码表(避免影响本次梯度反向传播, 在量化操作前进行更新)
        if self.ema and self.training and (self.update_count % self.update_interval == 0):
            self.updateCodebook()
            # self.saveLog('./log.txt')

        # 量化(需要将c矩阵获取到的梯度中继给x, delta_onehot的梯度则停掉)
        delta_onehot, c = self.vecQuantization.apply(x, self.getCodebook())

        # # 配合STE试试: 和猜想的一样，没有用，因为k_hat除了参与量化损失计算，并没有参与主要的推理
        # delta_onehot, c = self.clustering(x, self.getCodebook())

        # 累计码表更新量(量化操作之后进行)
        if self.ema and self.training:
            self.update_count += 1
            self.emaAccumulate(delta_onehot, x)

        return delta_onehot, c

# 基于聚类的量化注意力
class QuantizationAttn(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        codes=256,
        attn_drop_rate=0.,
        proj_drop_rate=0.,
        dropout_layer=dict(type='DropPath', drop_prob=0.),
        init_cfg=None,
        id_stage=0,
        id_block=0
    ):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.id_stage = id_stage
        self.id_block = id_block
        self.num_heads_local = num_heads
        self.num_heads_global = num_heads

        head_embed_dims = embed_dims // num_heads
        # 固定缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5

        # # 构造常规qkv
        # self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        # 分开构造qkv
        self.q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)

        # 实例化聚类器
        self.codes = codes              # 码本中的向量数
        self.cluster = Quantizer(num_heads, codes, head_embed_dims)
        
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        # 慢速注意力需要这两项
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)


    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.gradients = {
            'q': [],
            'k': [],
            'k_hat': [],
            'self.k.weight': [],
            'self.k.bias': [],
            'delta_onehot': [],
            'c': [],
            'qcT': [],
            'qct_exp': [],
            'deltaTv': [],
            'deltaT1': [],
            'numerator': [],
            'denominator': [],
        }

    def hook_getGradient_q(self, grad: torch.Tensor):
        self.gradients['q'].append(grad)

    def hook_getGradient_k(self, grad: torch.Tensor):
        # self.gradients['k'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('k', grad.min(), grad.mean(), grad.max()))
        # exit(0)

    def hook_getGradient_k_hat(self, grad: torch.Tensor):
        # self.gradients['k_hat'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('k_hat', grad.min(), grad.mean(), grad.max()))
        # exit(0)

    def hook_getGradient_k_weight(self, grad: torch.Tensor):
        # self.gradients['self.k.weight'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('self.k.weight', grad.min(), grad.mean(), grad.max()))
        # exit(0)

    def hook_getGradient_k_bias(self, grad: torch.Tensor):
        # self.gradients['self.k.bias'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('self.k.bias', grad.min(), grad.mean(), grad.max()))
        # exit(0)

    def hook_getGradient_delta_onehot(self, grad: torch.Tensor):
        # self.gradients['delta_onehot'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('delta_onehot', grad.min(), grad.mean(), grad.max()))

    def hook_getGradient_c(self, grad: torch.Tensor):
        # self.gradients['c'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('c', grad.min(), grad.mean(), grad.max()))

    def hook_getGradient_qcT(self, grad: torch.Tensor):
        # self.gradients['qcT'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('qcT', grad.min(), grad.mean(), grad.max()))


    def hook_getGradient_qcT_exp(self, grad: torch.Tensor):
        self.gradients['qct_exp'].append(grad)

    def hook_getGradient_deltaTv(self, grad: torch.Tensor):
        self.gradients['deltaTv'].append(grad)

    def hook_getGradient_deltaT1(self, grad: torch.Tensor):
        self.gradients['deltaT1'].append(grad)

    def hook_getGradient_numerator(self, grad: torch.Tensor):
        # self.gradients['numerator'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('numerator', grad.min(), grad.mean(), grad.max()))


    def hook_getGradient_denominator(self, grad: torch.Tensor):
        # self.gradients['denominator'].append(grad)
        print('grad of {:15s}, min: {:10.10f}, mean: {:10.10f}, max: {:10.10f}'.format('denominator', grad.min(), grad.mean(), grad.max()))


    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): input features with shape of (B, H, W, C)
        输出：
            x (tensor): [B, H, W, C]
        """
        batch_size, H, W, C = x.shape
        # # [3, batch_size, num_heads, L, head_embed_dims]
        # qkv = self.qkv(x).reshape(batch_size, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # # [batch_size, num_heads, L, head_embed_dims]
        # q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q(x).reshape(batch_size, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(batch_size, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(batch_size, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 使用固定缩放系数
        q = q * self.scale_base

        # [batch_size, num_heads, L, S], [num_heads, S, head_embed_dims]
        delta_onehot, c = self.cluster(k)
        # print(delta_onehot.requires_grad, c.requires_grad)
        # exit(0)

        # 量化损失计算
        k_hat = torch.einsum('bhls,hsd->bhld', delta_onehot, c)
        # print(k_hat.requires_grad)
        # exit(0)

        # # STE
        # # k = self.cluster.STE(k, k_hat)
        # k_hat = self.cluster.STE(k_hat, k)

        # # L2范数量化损失度量(参考)
        # error_quantization = torch.norm(k - k_hat, dim=-1).square().sum(dim=1).mean()
        # L2范数量化损失度量(修改:设置距离损失上限，防止过大导致训练不稳定)
        error_quantization = torch.norm(k - k_hat, dim=-1).square().sum(dim=1).mean().clip(max=10. * self.num_heads)
        # 相对量化损失
        # error_quantization = (torch.norm(k - k_hat, dim=-1) / torch.norm(k).clip(min=1e-6)).mean()
        # # 取消量化损失
        # error_quantization = torch.tensor(0., device=k.device)

        # [batch_size, num_heads, L, S]
        qcT = torch.einsum('bhld,hsd->bhls', q, c)
        qcT = qcT - qcT.max(dim=-1, keepdim=True)[0]
        qcT_exp = torch.exp(qcT)

        # 计算softmax分子
        # [batch_size, num_heads, S, head_embed_dims]
        deltaTv = torch.einsum('bhls,bhld->bhsd', delta_onehot, v)
        # [batch_size, num_heads, L, head_embed_dims]
        numerator = torch.einsum('bhls,bhsd->bhld', qcT_exp, deltaTv)
        # 计算softmax分母
        # [batch_size, num_heads, S]
        deltaT1 = torch.einsum('bhls->bhs', delta_onehot)
        # [batch_size, num_heads, L, 1]
        denominator = torch.einsum('bhls,bhs->bhl', qcT_exp, deltaT1).unsqueeze(-1).clip(min=1e-6)

        # print(k.min(), k.mean(), k.max())
        # print(qcT.min(), qcT.mean(), qcT.max())
        # print(deltaTv.min(), deltaTv.mean(), deltaTv.max())
        # print(deltaT1.min(), deltaT1.mean(), deltaT1.max())
        # print(numerator.min(), numerator.mean(), numerator.max())
        # print('cluster results: id_stage {}, id_block {}, max {}, total {}'.format(self.id_stage, self.id_block, deltaT1.max().data, (2**(7-self.id_stage))**2))
        # exit(0)

        # # 统计梯度信息
        # # if int(os.environ['RANK']) == 0:
        # q.register_hook(self.hook_getGradient_q)
        # k.register_hook(self.hook_getGradient_k)
        # k_hat.register_hook(self.hook_getGradient_k_hat)
        # self.k.weight.register_hook(self.hook_getGradient_k_weight)
        # self.k.bias.register_hook(self.hook_getGradient_k_bias)
        # delta_onehot.register_hook(self.hook_getGradient_delta_onehot)
        # c.register_hook(self.hook_getGradient_c)
        # qcT.register_hook(self.hook_getGradient_qcT)
        # qcT_exp.register_hook(self.hook_getGradient_qcT_exp)
        # deltaTv.register_hook(self.hook_getGradient_deltaTv)
        # deltaT1.register_hook(self.hook_getGradient_deltaT1)
        # numerator.register_hook(self.hook_getGradient_numerator)
        # denominator.register_hook(self.hook_getGradient_denominator)

        # 计算注意力加权的v
        # [batch_size, num_heads, L, head_embed_dims]
        x = numerator / denominator
        x = x.transpose(1, 2).reshape(batch_size, H, W, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        # # ------------------------------------------------------------ debug ------------------------------------------------------------
        # print('------------------------- id_stage: {:02d}, id_block: {:02d} -------------------------'.format(self.id_stage, self.id_block))
        # if self.window_size[0] > 1 or self.window_size[1] > 1:
        #     print('cluster_scale_learned: {:.6f}, cluster_scale_total: {:.6f}'.format(self.cluster.scale.data, self.cluster.scale * self.cluster.scale_base))
        # print('block_scale_learned: {:.6f}, block_scale_total: {:.6f}'.format(self.scale.data, self.scale * self.scale_base))
        # print('\n')
        # # ------------------------------------------------------------------------------------------------------------------------

        return x, error_quantization

# 分块注意力
class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.,
        proj_drop_rate=0.,
        dropout_layer=dict(type='DropPath', drop_prob=0.),
        init_cfg=None,
        id_stage=0,
        id_block=0
    ):

        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.id_stage = id_stage
        self.id_block = id_block
        self.num_heads_local = num_heads
        self.num_heads_global = num_heads

        head_embed_dims = embed_dims // num_heads
        # 固定缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        # # 可学习的缩放系数
        # self.scale_base = qk_scale or head_embed_dims**-0.5
        # self.scale = nn.Parameter(torch.tensor(1.0))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        self.num_heads_local))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]    # [B*num_windows, heads, N, dim]

        # 使用固定缩放系数
        q = q * self.scale_base

        # # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)
        # q = q * self.scale * self.scale_base                            # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

        attn = (q @ k.transpose(-2, -1))                                # [B*num_windows, num_heads, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )                           # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()   # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)                 # [B*num_windows, num_heads, N, dim]

        error_quantization = torch.tensor(0.0, device=x.device)

        x = self.proj(x)
        x = self.proj_drop(x)

        # # ------------------------------------------------------------ debug ------------------------------------------------------------
        # print('------------------------- id_stage: {:02d}, id_block: {:02d} -------------------------'.format(self.id_stage, self.id_block))
        # print('block_scale_learned: {:.6f}, block_scale_total: {:.6f}'.format(self.scale.data, self.scale * self.scale_base))
        # print('\n')
        # # ------------------------------------------------------------------------------------------------------------------------

        return x, error_quantization

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

# 在这里切换分块注意力和量化全局注意力
class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        shift_size=0,
        gla=False,          # 全局注意力控制变量
        qkv_bias=True,
        qk_scale=None,
        codes=256,
        attn_drop_rate=0,
        proj_drop_rate=0,
        dropout_layer=dict(type='DropPath', drop_prob=0.),
        init_cfg=None,
        id_stage=0,
        id_block=0
    ):
        super().__init__(init_cfg=init_cfg)

        if not gla:
            self.window_size = window_size
        else:
            self.window_size = 2**(3 - id_stage)   # 全局注意力使用较小一点的窗口，避免深层网络聚类中心太少
        self.shift_size = shift_size
        self.gla = gla      # 全局注意力控制变量
        assert 0 <= self.shift_size < self.window_size

        if not gla:
            self.w_msa = WindowMSA(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=to_2tuple(self.window_size),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=proj_drop_rate,
                dropout_layer=dropout_layer,
                init_cfg=None,
                id_stage=id_stage,
                id_block=id_block
            )
        else:
            self.w_msa = QuantizationAttn(
                embed_dims=embed_dims,
                num_heads=num_heads,
                window_size=to_2tuple(self.window_size),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                codes=codes,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=proj_drop_rate,
                dropout_layer=dropout_layer,
                init_cfg=None,
                id_stage=id_stage,
                id_block=id_block
            )

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        '''
        query: tensors, [batch_size, L, C]
        hw_shape: tuple, (H, W)
        '''
        # if not self.training:
        #     print('id_stage: ', self.w_msa.id_stage)
        #     print('id_block: ', self.w_msa.id_block)
        #     print('hw_shape', hw_shape)
        #     print('query.shape: ', query.shape)

        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b), mode='reflect')                # F.pad的pad顺序为原始tensor从右往左的通道顺序，每个通道有2个参数

        H_pad, W_pad = query.shape[1], query.shape[2]

        if not self.gla:    # 使用分块注意力
            # cyclic shift
            if self.shift_size > 0:
                shifted_query = torch.roll(
                    query,
                    shifts=(-self.shift_size, -self.shift_size),
                    dims=(1, 2))

                # calculate attention mask for SW-MSA
                img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size,
                                -self.shift_size), slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size,
                                -self.shift_size), slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                # nW, window_size, window_size, 1
                mask_windows = self.window_partition(img_mask)
                mask_windows = mask_windows.view(
                    -1, self.window_size * self.window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                                float(-100.0)).masked_fill(
                                                    attn_mask == 0, float(0.0))
            else:
                shifted_query = query
                attn_mask = None

            # nW*B, window_size, window_size, C
            query_windows = self.window_partition(shifted_query)
            # nW*B, window_size*window_size, C
            query_windows = query_windows.view(-1, self.window_size**2, C)

            # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
            attn_windows, error_quantization = self.w_msa(query_windows, mask=attn_mask)

            # merge windows
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

            # B H' W' C
            shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(
                    shifted_x,
                    shifts=(self.shift_size, self.shift_size),
                    dims=(1, 2))
            else:
                x = shifted_x
        else:   # 使用聚类全局注意力
            x, error_quantization = self.w_msa(query)

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x, error_quantization

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        window_size=7,
        shift=False,
        gla = False,    # 全局注意力控制变量
        qkv_bias=True,
        qk_scale=None,
        codes=256,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        with_cp=False,
        init_cfg=None,
        id_stage=0,
        id_block=0
    ):

        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.gla = gla

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            gla = gla,  # 全局注意力控制变量
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            codes=codes,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None,
            id_stage=id_stage,
            id_block=id_block
        )

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):
        # # 完全checkpoints
        # def _inner_forward(x, hw_shape):
        #     identity = x
        #     x = self.norm1(x)
        #     x, error_quantization = self.attn(x, hw_shape)

        #     x = x + identity

        #     identity = x
        #     x = self.norm2(x)
        #     x = self.ffn(x, identity=identity)

        #     return x, error_quantization
        
        # if self.with_cp and x.requires_grad and self.gla:
        #     x, error_quantization = cp.checkpoint(_inner_forward, x, hw_shape)
        # else:
        #     x, error_quantization = _inner_forward(x, hw_shape)

        # return x, error_quantization

        # # 仅attn checkpoints
        # # attn
        # def _inner_forward(x, hw_shape):
        #     identity = x
        #     x = self.norm1(x)
        #     x, error_quantization = self.attn(x, hw_shape)
        #     x = x + identity
        #     return x, error_quantization
        
        # if self.with_cp and x.requires_grad and self.gla:
        #     x, error_quantization = cp.checkpoint(_inner_forward, x, hw_shape)
        # else:
        #     x, error_quantization = _inner_forward(x, hw_shape)
        # # ffn    
        # identity = x
        # x = self.norm2(x)
        # x = self.ffn(x, identity=identity)
        # return x, error_quantization

        # # 仅ffn checkpoints
        # def _inner_forward(x):
        #     identity = x
        #     x = self.norm2(x)
        #     x = self.ffn(x, identity=identity)
        #     return x
        # # attn
        # identity = x
        # x = self.norm1(x)
        # x, error_quantization = self.attn(x, hw_shape)
        # x = x + identity
        # # ffn
        # if self.with_cp and x.requires_grad:
        #     x = cp.checkpoint(_inner_forward, x)
        # else:
        #     x = _inner_forward(x)
        # return x, error_quantization

        # 自由组合
        def _inner_forward_attn(x, hw_shape):
            identity = x
            x = self.norm1(x)
            x, error_quantization = self.attn(x, hw_shape)
            x = x + identity
            return x, error_quantization

        def _inner_forward_ffn(x):
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x
        # attn
        if self.with_cp and x.requires_grad and not self.gla:
            x, error_quantization = cp.checkpoint(_inner_forward_attn, x, hw_shape)
        else:
            x, error_quantization = _inner_forward_attn(x, hw_shape)
        # ffn
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward_ffn, x)
        else:
            x = _inner_forward_ffn(x)
        return x, error_quantization


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        depth,
        window_size=7,
        qkv_bias=True,
        qk_scale=None,
        codes=256,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        downsample=None,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        with_cp=False,
        init_cfg=None,
        id_stage=0
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                # shift=False if i % 2 == 0 else True,
                shift=False,        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意，这里关闭了循环移位 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                gla=False if i % 2 == 0 else True,
                # gla=False,
                qkv_bias=qkv_bias,
                qk_scale=None if i % 2 == 0 else qk_scale,
                codes=codes,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                id_stage=id_stage,
                id_block=i
            )
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        '''
        x: []
        hw_shape: (H, W)
        '''
        error_quantization_blocks = []
        for block in self.blocks:
            x, error_quantization = block(x, hw_shape)
            error_quantization_blocks.append(error_quantization)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape, error_quantization_blocks
        else:
            return x, hw_shape, x, hw_shape, error_quantization_blocks

@MODELS.register_module()
class SwinTransformerVQ(BaseModule):
    """Swin Transformer backbone.

    This backbone is the implementation of `Swin Transformer:
    Hierarchical Vision Transformer using Shifted
    Windows <https://arxiv.org/abs/2103.14030>`_.
    Inspiration from https://github.com/microsoft/Swin-Transformer.

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int | float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LN').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        codes=256,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        with_cp=False,
        pretrained=None,
        frozen_stages=-1,
        init_cfg=None
    ):
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super().__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                codes=codes,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
                id_stage = i,
            )
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v
                else:
                    state_dict[k] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    print_log('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                if table_key in self.state_dict():
                    table_current = self.state_dict()[table_key]
                    L1, nH1 = table_pretrained.size()
                    L2, nH2 = table_current.size()
                    if nH1 != nH2:
                        print_log(f'Error in loading {table_key}, pass')
                    elif L1 != L2:
                        S1 = int(L1**0.5)
                        S2 = int(L2**0.5)
                        table_pretrained_resized = F.interpolate(
                            table_pretrained.permute(1, 0).reshape(
                                1, nH1, S1, S1),
                            size=(S2, S2),
                            mode='bicubic')
                        state_dict[table_key] = table_pretrained_resized.view(
                            nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x, data_samples = None):

        if not data_samples is None:
            labels = []
            for data_sample in data_samples:
                label = data_sample.gt_sem_seg.data.float()     # [1, H_original, W_original]
                labels.append(label)
            labels = torch.stack(labels, dim=0)                 # [B, 1, H_original, W_original]
            # print('----------------------------------------- ', labels.shape)
            # exit(0)
        else:
            labels = None

        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        error_quantization_stages = []

        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape, error_quantization_blocks = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(
                    -1, *out_hw_shape,
                    self.num_features[i]
                ).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
            error_quantization_stages.append(torch.stack(error_quantization_blocks))
        error_quantization = torch.concat(error_quantization_stages, dim=0)

        return tuple(outs), error_quantization
