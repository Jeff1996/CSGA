'''
架构说明: swin_debug_old_1111.py
1. 使用动态生成的码表(k-means聚类)，迭代过程中使用EMA更新码表
2. 对单位化的K矩阵进行量化处理
3. 局部分支与量化全局分支采用并行结构，特征融合方式为特征拼接/相加
4. qk_scale = scale_learned * scale_base(30)
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

from mmseg.registry import MODELS
from ..utils.embed import PatchEmbed, PatchMerging
from ..utils.vqt_tools import reduce_sum

# stage_list = []
# stage_list = [0, 1]
# stage_list = [0]
# stage_list = [1]
# stage_list = [2]
# stage_list = [3]
stage_list = [0, 1, 2, 3]

# 自定义梯度传播过程
class vecQuantization(Function):  
    @staticmethod  
    def forward(ctx, x, c):
        '''
        输入: 
        x           : tensor, [batch_size, heads, num_windows, N, dim], 分块的K矩阵
        c           : tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        输出:
        delta_onehot: tensor, [batch_size, heads, num_windows, N, num_windows], 量化的K矩阵
        c           : tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        # 前向传播（基于余弦相似度/单位化点积）
        cosSim = torch.einsum('bhrcd,bhsd->bhrcs', x, c)                    # 相似度矩阵, [batch_size, heads, num_windows, N, num_windows]
        delta_index = cosSim.argmax(dim=-1)                                 # 索引矩阵, [batch_size, heads, num_windows, N]
        delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()          # one-hot索引矩阵, [batch_size, heads, num_windows, N, num_windows]

        # # 前向传播（基于欧式距离）
        # x2 = torch.sum(torch.square(x), dim=-1).unsqueeze(-1)               # [B, heads, L, d] -> [B, heads, L] -> [B, heads, L, 1]
        # xc = torch.einsum('bhld,hsd->bhls', x, c)                           # [B, heads, L, S]
        # c2 = torch.sum(torch.square(c), dim=-1).unsqueeze(1).unsqueeze(0)   # [heads, S, d] -> [heads, S] -> [heads, 1, S] -> [1, heads, 1, S]
        # distance2 = x2 - 2*xc + c2                                          # 待量化序列中每一个向量与码表中每一个向量的欧氏距离的平方, [B, L, S]
        # delta_index = distance2.argmin(dim=-1)                              # 索引矩阵, [B, heads, L]
        # delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()           # one-hot索引矩阵, [B, heads, L, S]

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
        # 获取中间变量
        (delta_onehot, ) = ctx.saved_tensors
        # 梯度反传
        grad_x = torch.einsum('bhrcs,bhsd->bhrcd', delta_onehot, grad_output_c) # 来自码表c的梯度
        # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None
        return grad_x, None

# 向量量化器
class Quantizer(nn.Module):
    '''
    序列量化器
    1. 保存一个可训练的量化码表C
    2. 构造量化的K序列K^
    3. 获取输入矩阵K量化后的索引矩阵Δ
    '''
    def __init__(self, gamma: float=0.5, iterations: int=1):
        '''
        codes   : 码表中行向量的个数
        dim     : 码表每个行向量的维数(与K矩阵的维数一致)
        '''
        super(Quantizer, self).__init__()
        self.gamma = gamma                                      # EMA超参数（历史成分占比）
        self.iterations = iterations                            # 码表更新迭代次数
        self.vecQuantization = vecQuantization()                # 自定义梯度反传过程的向量量化函数

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

    # 使用类平均池化初始化聚类中心(码表)
    def initCodebook(self, x: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, heads, num_windows, N, dim], 分块的K矩阵
        输出:
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        c_init = torch.sum(x, dim=-2)                                           # [batch_size, heads, num_windows, dim], 将一个块内的元素求和
        c_init = c_init / torch.norm(c_init, dim=-1, keepdim=True)              # [batch_size, heads, num_windows, dim], 单位化码表
        return c_init

    # 迭代更新码表
    def updateCodebook(self, x: torch.Tensor, c: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, heads, num_windows*N, dim], 分块的K矩阵
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        输出：
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        affinity = torch.einsum('bhnd,bhld->bhnl', c, x)        # [batch_size, heads, num_windows, L], KC^T

        argmax_hw = affinity.argmax(dim=-1)                     # [batch_size, heads, num_windows]
        argmax_n = affinity.argmax(dim=-2)                      # [batch_size, heads, L]

        # [batch_size, heads, num_windows, L]
        argmax_hw_onehot = F.one_hot(argmax_hw, x.shape[-2]).float()
        # [batch_size, heads, num_windows, L]
        argmax_n_onehot = F.one_hot(argmax_n, c.shape[-2]).float().transpose(-2, -1)

        # attn = torch.softmax(attn, dim=-1) + torch.softmax(attn, dim=-2)
        # attn = torch.softmax(attn, dim=-1)
        attn = argmax_hw_onehot + argmax_n_onehot

        delta_c = torch.einsum('bhnl,bhld->bhnd', attn, x)   # [batch_size, heads, num_windows, dim]
        delta_c = delta_c / torch.norm(delta_c, dim=-1, keepdim=True)
        c_new = self.gamma * c + (1 - self.gamma) * delta_c # 更新码表
        # c_new = delta_c
        c_new = c_new / torch.norm(c_new, dim=-1, keepdim=True)
        return c_new

    # 结合码表初始化与码表更新
    def getCodebook(self, x: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, heads, num_windows, N, dim], 分块的K矩阵
        输出:
        c: tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        # 初始化聚类中心
        c = self.initCodebook(x)                            # [batch_size, heads, num_windows, dim]
        # 通过k-means聚类更新码表c
        x = x.flatten(start_dim=2, end_dim=3)               # [batch_size, heads, L, dim]
        for _ in range(self.iterations):
            c = self.updateCodebook(x, c)
        return c

    def forward(self, x: torch.Tensor, batch_size: int=1):
        '''
        输入
        x           : tensor, [batch_size, heads, num_windows, N, dim], K矩阵
        输出
        delta_onehot: tensor, [batch_size, heads, num_windows, N, num_windows], 量化的K矩阵
        c           : tensor, [batch_size, heads, num_windows, dim], 每个样本、每个头一个码表
        '''
        # 量化x
        delta_onehot, c = self.vecQuantization.apply(x, self.getCodebook(x))

        return delta_onehot, c

# 局部和全局分支采用相同head
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
        # # 固定缩放系数
        # self.scale_base = qk_scale or head_embed_dims**-0.5
        # 可学习的缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        self.scale = nn.Parameter(torch.tensor(1.0))

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
        if self.id_stage in stage_list:
            # self.proj = nn.Linear(embed_dims*2, embed_dims)
            self.proj = nn.Linear(embed_dims, embed_dims)
        else:
            self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

        if self.id_stage in stage_list:
            # 在这里添加量化器
            self.quantizer = Quantizer(
                gamma=0.5,      # 码表迭代更新过程的EMA参数
                iterations=2    # 码表更新迭代次数, 0此表示不迭代，直接使用类池化结果作为码表
            )                                                           # 输入序列量化器
            self.drop = build_dropout(dropout_layer)                    # 随机丢弃局部或全局注意力(一次只能丢一个)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def localAttention(self, q, k, v, mask=None, drop_path=False):
        '''
        局部注意力
        q/k/v: [B*num_windows, heads, N, dim]
        
        '''
        B, heads_sub, N, dim = q.shape

        attn = (q @ k.transpose(-2, -1))                                # [B*num_windows, heads_sub, N, N]

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, heads_sub, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, heads_sub, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v                                                    # [B*num_windows, heads_sub, N, dim]

        if drop_path:
            x = self.drop(x)

        return x

    def globalAttention(self, q, k, v, batch_size, drop_path=False):
        '''
        稀疏全局注意力
        q/k/v: [B*num_windows, heads, N, dim]

        '''
        B, heads_sub, N, dim = q.shape
        num_windows = B // batch_size

        # [batch_size, heads_sub, num_windows, N, head_embed_dims]
        q = q.view(batch_size, num_windows, heads_sub, N, dim).permute(0, 2, 1, 3, 4).contiguous()
        k = k.view(batch_size, num_windows, heads_sub, N, dim).permute(0, 2, 1, 3, 4).contiguous()
        v = v.view(batch_size, num_windows, heads_sub, N, dim).permute(0, 2, 1, 3, 4).contiguous()

        # 构造量化码表
        # [batch_size, heads, num_windows, N, num_windows], [batch_size, heads, num_windows, dim]
        delta_onehot, c = self.quantizer(k)

        # [batch_size, heads_sub, num_windows, N, head_embed_dims]
        k_hat = torch.einsum('bhrcs,bhsd->bhrcd', delta_onehot, c)

        # 量化后的窗口注意力
        # [batch_size, heads_sub, num_windows, N, S]
        qcT = torch.einsum('bhrcd,bhsd->bhrcs', q, c)

        # 防溢出处理
        # qcT = qcT - qcT.max(dim=-1, keepdim=True)[0]
        
        # exp
        # [batch_size, heads_sub, num_windows, N, S]
        qcT_exp = torch.exp(qcT)

        # 计算softmax分子
        deltaTv = torch.einsum('bhrcs,bhrcv->bhsv', delta_onehot, v)    # [B, heads_sub, S, dim]
        numerator = torch.einsum('bhrcs,bhsv->bhrcv', qcT_exp, deltaTv) # [B, heads_sub, num_windows, N, dim]
        # 计算softmax分母
        deltaT1 = torch.einsum('bhrcs->bhs', delta_onehot)              # [B, heads_sub, S]
        denominator = torch.einsum('bhrcs,bhs->bhrc', qcT_exp, deltaT1).unsqueeze(-1)   # [B, heads_sub, num_windows, N, 1]
        denominator[denominator==0] = 1e-6                              # 防止除以0

        # 计算注意力加权的v
        x = numerator / denominator                                     # [B, heads_sub, num_windows, N, dim]
        x = x.transpose(1, 2).contiguous().view(B, heads_sub, N, dim)   # [B*num_windows, heads_sub, N, dim]

        # 计算量化损失
        # # L2量化损失
        # error_quantization = torch.norm(k - k_hat.detach(), dim=-1).square().mean()
        # 余弦相似度损失
        error_quantization = (1 - torch.einsum('bhrcmk,bhrckn->bhrcmn', k.unsqueeze(-2), k_hat.detach().unsqueeze(-1))).mean()

        if drop_path:
            x = self.drop(x)

        return x, error_quantization

    def forward(self, x, batch_size, mask=None):
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

        # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        q = q / torch.norm(q, dim=-1, keepdim=True)
        k = k / torch.norm(k, dim=-1, keepdim=True)
        q = q * self.scale * self.scale_base                # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

        if self.id_stage in stage_list:
            # if torch.rand(1).item() > 0.5:
            #     drop_path_local = True
            #     drop_path_global = False
            # else:
            #     drop_path_local = False
            #     drop_path_global = True
            # if not self.training:
            #     drop_path_local = False
            #     drop_path_global = False

            drop_path_local = False
            drop_path_global = False

            x1 = self.localAttention(q, k, v, mask, drop_path_local)                            # 局部注意力, [B*num_windows, heads, N, dim]
            x2, error_quantization = self.globalAttention(q, k, v, batch_size, drop_path_global) # 全局注意力, [B*num_windows, heads, N, dim]
            # x2, error_quantization = torch.zeros_like(x1, device=x1.device), torch.tensor(0.0, device=x.device)
            # x1 = torch.zeros_like(x2, device=x2.device)
            # x = torch.concat((x1, x2), dim=1).transpose(1, 2).reshape(B, N, C*2) # [B*num_windows, N, C*2]
            x = (x1 + x2).transpose(1, 2).reshape(B, N, C)                                      # [B*num_windows, N, C]
        else:
            x = self.localAttention(q, k, v, mask, False).transpose(1, 2).reshape(B, N, C)
            error_quantization = torch.tensor(0.0, device=x.device)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, error_quantization

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


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
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0,
        proj_drop_rate=0,
        dropout_layer=dict(type='DropPath', drop_prob=0.),
        init_cfg=None,
        id_stage=0,
        id_block=0
    ):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            dropout_layer=dropout_layer,
            init_cfg=None,
            id_stage=id_stage,
            id_block=id_block
        )

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

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
        attn_windows, error_quantization = self.w_msa(query_windows, batch_size=B, mask=attn_mask)

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
        qkv_bias=True,
        qk_scale=None,
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

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
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

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x, error_quantization = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x, error_quantization

        if self.with_cp and x.requires_grad:
            x, error_quantization = cp.checkpoint(_inner_forward, x)
        else:
            x, error_quantization = _inner_forward(x)

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
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
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
class SwinTransformer(BaseModule):
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

    def __init__(self,
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
                 init_cfg=None):
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

    def forward(self, x):
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
        return outs, error_quantization
