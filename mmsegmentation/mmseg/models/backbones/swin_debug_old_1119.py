'''
架构说明: swin_debug_old_1119.py
1. 使用下采样的k、v矩阵降低计算复杂度
2. 使用平均池化进行初步下采样，使用掩膜进行非极大值抑制，实现标准的k-means聚类(仅迭代1次)
3. 局部分支(WindowMSA)与下采样全局分支(ClusterAttn)采用交替级联结构
4. qk_scale = 30，主分支和聚类过程各引入了一个初始化为1的可学习缩放系数，qk使用余弦相似度（单位化点积），k矩阵下采样的得到的聚类中心需要重新单位化
5. 全局分支引入相对位置编码

训练结果: 
    /home/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/20241120_161212/20241120_161212.log
    /home/hjf/workspace/mmsegmentation/work_dirs/swin-tiny-patch4-window7-LN_upernet_2xb1-80k_ade20k-512x512/20241122_112633/20241122_112633.log
训练配置: batchsize 8 -> 16, 80k, ade20k, 1118架构80k迭代权重
mIoU: 43.09 -> 43.32
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
# stage_list = [0, 1, 2, 3]
iterations = 1

# 自定义梯度传播过程
class vecQuantization(Function):  
    @staticmethod  
    def forward(ctx, x, c):
        '''
        输入: 
        x           : tensor, [batch_size, num_windows, N, embed_dims], 分块的X矩阵
        c           : tensor, [batch_size, S, embed_dims], 每个样本一个码表
        输出:
        delta_onehot: tensor, [batch_size, num_windows, N, S], 量化的X矩阵的索引矩阵
        c           : tensor, [batch_size, S, embed_dims], 每个样本一个码表
        '''
        batch_size, num_windows, N, _ = x.shape
        _, S, _ = c.shape

        delta_index = torch.arange(num_windows, device=x.device).view(1, num_windows, 1).repeat(batch_size, 1, N)
        delta_onehot = F.one_hot(delta_index, S).float()

        # # 相似性度量与量化（基于点积）
        # sim = torch.einsum('brcd,bsd->brcs', x, c)                      # 相似度矩阵, [batch_size, num_windows, N, S]
        # delta_index = sim.argmax(dim=-1)                                # 索引矩阵, [batch_size, num_windows, N]
        # delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()      # one-hot索引矩阵, [batch_size, num_windows, N, S]

        # # 相似性度量与量化（基于欧式距离）
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

        return delta_onehot, c                                                  # forward的输出个数与backward的输入个数相同
  
    @staticmethod  
    def backward(ctx, grad_output_delta_onehot, grad_output_c):  
        # 获取中间变量
        (delta_onehot, ) = ctx.saved_tensors
        # 梯度反传
        grad_x = torch.einsum('brcs,bsd->brcd', delta_onehot, grad_output_c)   # 来自码表c的梯度
        # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None
        return grad_x, None

# 自定义k-means聚类梯度反向传播规则
class gradKMeans(Function):  
    @staticmethod
    def forward(ctx, affinity_onehot, x):
        '''
        输入: 
        affinity_onehot : tensor, [batch_size, num_heads, L, L'], 
        x               : tensor, [batch_size, num_heads, L, head_embed_dims], 
        输出:
        c               : tensor, [batch_size, num_heads, L', head_embed_dims], 聚类中心
        '''
        c_sum = torch.einsum(
                'bhlm,bhld->bhmd', 
                affinity_onehot, 
                x
            )
        affinity_onehot_sum = torch.torch.sum(affinity_onehot, dim=-2).unsqueeze(-1)
        affinity_onehot_sum[affinity_onehot_sum == 0] = 1
        c = c_sum / affinity_onehot_sum

        # 保存必要的中间变量用于梯度反向传播
        ctx.save_for_backward(affinity_onehot)

        # forward的输出个数与backward的输入个数相同
        return c
  
    @staticmethod  
    def backward(ctx, grad_output_c):  
        '''
        grad_output_c: tensor, [batch_size, num_heads, L', head_embed_dims]
        '''
        # 获取中间变量
        # [batch_size, num_heads, L, L']
        (affinity_onehot, ) = ctx.saved_tensors
        grad_output_c = grad_output_c / torch.sum(affinity_onehot, dim=-2).unsqueeze(-1)    # 梯度缩小
        grad_x = torch.einsum('bhlm,bhmd->bhld', affinity_onehot, grad_output_c)            # 梯度分配

        # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None
        return None, grad_x

# 向量量化器
class Cluster(nn.Module):
    '''
    序列量化器
    1. 保存一个可训练的量化码表C
    2. 构造量化的K序列K^
    3. 获取输入矩阵K量化后的索引矩阵Δ
    '''
    def __init__(self, window_size: tuple, iterations: int, qk_scale: float):
        '''
        codes   : 码表中行向量的个数
        dim     : 码表每个行向量的维数(与K矩阵的维数一致)
        '''
        super(Cluster, self).__init__()
        self.window_size = window_size              # (window_h, window_w)
        self.iterations = iterations                # 码表更新迭代次数
        self.scale_base = qk_scale                  # 固定的基础缩放系数
        self.scale = nn.Parameter(torch.tensor(1.0))# 可学习的缩放系数的缩放系数
        # self.scale = 1

        self.subsample = nn.AvgPool2d(kernel_size=self.window_size, stride=self.window_size)

        # # [batch_size, num_heads, L, L']
        # self.relative_position_bias_init = nn.Parameter(
        #     torch.zeros((1, 1, 1, 1), dtype=torch.float32), 
        #     requires_grad=False
        # )                                           # 初始相对位置编码

        # self.gradKMeans_c_q = gradKMeans()          # 自定以k-means梯度反向传播规则
        # self.gradKMeans_c_v = gradKMeans()          # 自定以k-means梯度反向传播规则

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

    # 相对位置掩膜构造函数
    @staticmethod
    def getMask(size: tuple, index_onehot: torch.Tensor, gain: float=1.0):
        '''
        size: 特征图尺寸, [h, w]
        index_onehot: 聚类结果(每个像素对应的聚类中心的one-hot索引), [B, num_heads, L, S]
        gain: 增益系数
        '''
        assert type(size) == tuple, 'Data type of size in function <getMask> should be <tuple>!'
        assert size.__len__() == 2, 'Length of size should be 2!'
        coords_h = torch.arange(size[0])
        coords_w = torch.arange(size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))                          # 构造坐标窗口元素坐标索引，[2, h, w]
        # 一维化特征图像素坐标，[2, L]
        coords_featuremap = torch.flatten(coords, start_dim=1).float().to(index_onehot.device)
        # [B, num_heads, 2, L]
        coords_featuremap = coords_featuremap.reshape(1, 1, 2, -1).repeat(index_onehot.shape[0], index_onehot.shape[1], 1, 1)
        # [B, num_heads, 1, S]
        index_onehot_sum = torch.sum(index_onehot, dim=-2, keepdim=True)
        index_onehot_sum[index_onehot_sum==0] = 1
        
        # 聚类中心坐标，[B, num_heads, 2, S]
        coords_clustercenter = torch.einsum('bhcl,bhls->bhcs', coords_featuremap, index_onehot) / index_onehot_sum
            
        # 构造相对位置矩阵, 第一个矩阵是h方向的相对位置差, 第二个矩阵是w方向的相对位置差
        relative_coords = coords_featuremap[:, :, :, :, None] - coords_clustercenter[:, :, :, None, :]
        distance = torch.sqrt(                                                              # [B, num_heads, L, S]
            torch.square(relative_coords[:,:,0,:,:]) + torch.square(relative_coords[:,:,1,:,:])
        )
        # exp操作用于处理distance中的0, [B, num_heads, L, S]
        distance_exp = torch.exp(distance)
        # 距离越远的token注意力增强越少(加性增强), 最大值为1*gain, 最小值可以接近0, [B, num_heads, L, S]
        mask = (1 / distance_exp) * gain
        return mask

    # 获取初始相对位置编码(与初始化聚类中心构造方法有关), 如果图片尺寸不变化，则只需要构造一次
    def initRPE(self, shape_x, shape_c, device):
        batch_size, num_heads, L_, _ = shape_c
        _, _, H, W, _ = shape_x
        H_, W_ = H // self.window_size[0], W // self.window_size[1]
        N = self.window_size[0] * self.window_size[1]

        delta_index = torch.arange(L_, device=device).view(1, 1, L_, 1).repeat(batch_size, num_heads, 1, N)
        delta_index = delta_index.reshape(
            batch_size, num_heads, H_, W_, self.window_size[0], self.window_size[1]
        ).permute(
            0, 1, 2, 4, 3, 5
        ).reshape(
            batch_size, num_heads, H_ * self.window_size[0], W_ * self.window_size[1]
        ).reshape(
            batch_size, num_heads, H*W
        )                                                           # [B, num_heads, L]
        delta_onehot = F.one_hot(delta_index, L_).float()           # [B, num_heads, L, L']
        relative_position_bias_init = self.getMask((H, W), delta_onehot) # [B, num_heads, L, L']
        return relative_position_bias_init

    # 使用类平均池化初始化聚类中心(码表)
    def initCenter(self, x_k: torch.Tensor, x_v: torch.Tensor):
        '''
        输入: 
        x: tensor, [batch_size, num_heads, H, W, head_embed_dims], 待聚类数据
        输出:
        c_init: tensor, [batch_size, num_heads, L', head_embed_dims], 每个样本一个码表
        '''
        batch_size, num_heads, H, W, head_embed_dims = x_k.shape

        # [batch_size*num_heads, head_embed_dims, H, W]
        x_k_temp = x_k.permute(0, 1, 4, 2, 3).reshape(batch_size*num_heads, head_embed_dims, H, W)
        # [batch_size, num_heads, L', head_embed_dims]
        c_q_init = self.subsample(x_k_temp).reshape(
            batch_size, num_heads, head_embed_dims, -1
        ).transpose(-2, -1)
        # 单位化c_q_init, 便于后续进行余弦相似度计算
        c_q_init = F.normalize(c_q_init, dim=-1)

        # [batch_size*num_heads, head_embed_dims, H, W]
        x_v_temp = x_v.permute(0, 1, 4, 2, 3).reshape(batch_size*num_heads, head_embed_dims, H, W)
        # [batch_size, num_heads, L', head_embed_dims]
        c_v_init = self.subsample(x_v_temp).reshape(
            batch_size, num_heads, head_embed_dims, -1
        ).transpose(-2, -1)
        return c_q_init, c_v_init

    # 迭代更新码表
    def updateCenter(self, x: torch.Tensor, c: torch.Tensor, relative_position_bias: None):
        '''
        输入: 
        x: tensor, [batch_size, num_heads, H, W, head_embed_dims], 待聚类数据
        c: tensor, [batch_size, num_heads, L', head_embed_dims], 初始化聚类中心
        relative_position_bias: tensor, [batch_size, num_heads, L, L'], 相对位置编码
        输出：
        c_new: tensor, [B, num_heads, L', head_embed_dims], 更新后的聚类中心
        '''
        batch_size, num_heads, H, W, head_embed_dims = x.shape
        _, _, L_, _ = c.shape

        x = x.reshape(batch_size, num_heads, -1, head_embed_dims)

        # # 不可导方案
        # # [B, num_heads, L, L']
        # affinity = torch.einsum('bhld,bhmd->bhlm', x * self.scale_base, c) + relative_position_bias
        # # [B, num_heads, L]
        # affinity_argmax = torch.argmax(affinity, dim=-1)
        # # [B, num_heads, L, L']
        # affinity_onehot = F.one_hot(affinity_argmax, L_).float()
        # affinity_onehot_sum = torch.torch.sum(affinity_onehot, dim=-2).unsqueeze(-1)
        # affinity_onehot_sum[affinity_onehot_sum == 0] = 1
        # # [B, num_heads, L', head_embed_dims]
        # c_sum = torch.einsum('bhlm,bhld->bhmd', affinity_onehot, x)
        # # [B, num_heads, L', head_embed_dims] / [B, num_heads, L', 1]
        # c_new = c_sum / affinity_onehot_sum

        # # 可导方案1
        # # [B, num_heads, L, L']
        # affinity = torch.einsum('bhld,bhmd->bhlm', x * self.scale_base * self.scale, c)
        # if not relative_position_bias is None:
        #     affinity = affinity + relative_position_bias
        # # [B, num_heads, L, L']
        # affinity = torch.softmax(affinity, dim=-1)

        # 可导方案2: 
        # [B, num_heads, L, L']
        affinity = torch.einsum('bhld,bhmd->bhlm', x * self.scale_base * self.scale, c)
        if not relative_position_bias is None:
            affinity = affinity + relative_position_bias
        # 使用掩膜进行非极大值抑制
        affinity_mask = torch.zeros_like(affinity)
        affinity_mask[affinity < affinity.max(dim=-1, keepdim=True)[0]] = -torch.inf
        affinity = affinity + affinity_mask
        # [B, num_heads, L, L']
        affinity = torch.softmax(affinity, dim=-1)

        # 更新聚类中心
        # [B, num_heads, L', head_embed_dims]
        c_sum = torch.einsum('bhlm,bhld->bhmd', affinity, x)
        # 直接单位化，就不用affinity归一化了
        c_new = F.normalize(c_sum, dim=-1)
        
        # # [B, num_heads, L, L']
        # affinity_sum = torch.torch.sum(affinity, dim=-2).unsqueeze(-1)
        # affinity_sum[affinity_sum == 0] = 1
        # # [B, num_heads, L', head_embed_dims] / [B, num_heads, L', 1]
        # c_new = c_sum / affinity_sum

        return c_new, affinity

    # 聚类中心初始化与聚类中心更新
    def getCenter(self, x_k: torch.Tensor, x_v: torch.Tensor):
        '''
        输入: 
        x_k/x_v: [batch_size, num_heads, H, W, head_embed_dims]
        输出:
        '''
        batch_size, num_heads, H, W, head_embed_dims = x_k.shape

        if self.iterations > 0:
            # # 不可导方案
            # with torch.no_grad():
            #     # [batch_size, num_heads, L', head_embed_dims]
            #     c_q, c_v = self.initCenter(x_k, x_v)
            #     relative_position_bias = self.initRPE(x_k.shape, c_q.shape, x_k.device)
            #     # 迭代更新聚类中心
            #     for _ in range(self.iterations):
            #         # [batch_size, num_heads, L', head_embed_dims], [batch_size, num_heads, L, L']
            #         c_q, affinity_onehot = self.updateCenter(x_k, c_q, relative_position_bias)
            #         # 更新相对位置编码
            #         relative_position_bias = self.getMask((H, W), affinity_onehot)
            # c_q_out = self.gradKMeans_c_q.apply(affinity_onehot, x_k.reshape(batch_size, num_heads, -1, head_embed_dims))
            # c_v_out = self.gradKMeans_c_v.apply(affinity_onehot, x_v.reshape(batch_size, num_heads, -1, head_embed_dims))

            # 可导方案
            # [batch_size, num_heads, L', head_embed_dims]
            c_q, _ = self.initCenter(x_k, x_v)
            with torch.no_grad():
                relative_position_bias = self.initRPE(x_k.shape, c_q.shape, x_k.device)
            # relative_position_bias = None
            for _ in range(self.iterations):
                # [batch_size, num_heads, L', head_embed_dims], [batch_size, num_heads, L, L']
                c_q, affinity = self.updateCenter(x_k, c_q, relative_position_bias)
                # 更新相对位置编码(注意需要断开affinity的梯度)
                with torch.no_grad():
                    relative_position_bias = self.getMask((H, W), affinity)
            c_q_out = c_q

            c_v_out_sum = torch.einsum(
                'bhlm,bhld->bhmd', 
                affinity, 
                x_v.reshape(batch_size, num_heads, -1, head_embed_dims)
            )
            affinity_sum = torch.torch.sum(affinity, dim=-2).unsqueeze(-1)
            affinity_sum[affinity_sum == 0] = 1
            c_v_out = c_v_out_sum / affinity_sum

        else:
            # [batch_size, num_heads, L', head_embed_dims]
            c_q, c_v = self.initCenter(x_k, x_v)
            with torch.no_grad():
                relative_position_bias = self.initRPE(x_k.shape, c_q.shape, x_k.device)
            c_q_out = c_q
            c_v_out = c_v

        return c_q_out, c_v_out, relative_position_bias


    def forward(self, x_k: torch.Tensor, x_v: torch.Tensor):
        '''
        输入
        x_k : tensor, [batch_size, num_heads, H, W, head_embed_dims], 待聚类的K矩阵
        x_v : tensor, [batch_size, num_heads, H, W, head_embed_dims], 待聚类的V矩阵
        输出
        c_q : tensor, [batch_size, num_heads, L', head_embed_dims], 聚类的K矩阵
        c_v : tensor, [batch_size, num_heads, L', head_embed_dims], 聚类的V矩阵
        '''
        c_q, c_v, relative_position_bias = self.getCenter(x_k, x_v)

        # 量化x
        # [batch_size, num_windows, N, S], [batch_size, S, embed_dims]
        # delta_onehot, c = self.vecQuantization.apply(x, self.getCodebook(x))

        return c_q, c_v, relative_position_bias

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
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

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

        # # 使用固定缩放系数
        # q = q * self.scale_base

        # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q * self.scale * self.scale_base                            # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

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
        return x, error_quantization

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

# 基于聚类的全局注意力
class ClusterAttn(BaseModule):
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

        # 构造常规qkv
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        # 实例化聚类器
        if self.window_size[0] > 1 or self.window_size[1] > 1:
            self.cluster = Cluster(self.window_size, iterations=iterations, qk_scale=self.scale_base)
        
        # 聚类中心线性映射
        # self.c_qv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)
        # self.c_q = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)


    def forward(self, x: torch.Tensor):
        """
        Args:
            x (tensor): input features with shape of (B, H, W, C)
        输出：
            x (tensor): [B, H, W, C]
        """
        batch_size, H, W, C = x.shape
        # [3, batch_size, num_heads, L, head_embed_dims]
        qkv = self.qkv(x).reshape(batch_size, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, L, head_embed_dims]
        x_q, x_k, x_v = qkv[0], qkv[1], qkv[2]

        # # 使用固定缩放系数
        # x_q = x_q * self.scale_base

        # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        x_q = F.normalize(x_q, dim=-1)
        x_k = F.normalize(x_k, dim=-1)
        x_q = x_q * self.scale * self.scale_base            # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

        if self.window_size[0] > 1 or self.window_size[1] > 1:
            # [batch_size, num_heads, L', head_embed_dims] * 2, [batch_size, num_heads, L, L']
            c_q, c_v, relative_position_bias = self.cluster(
                x_k.reshape(batch_size, self.num_heads, H, W, C // self.num_heads), 
                x_v.reshape(batch_size, self.num_heads, H, W, C // self.num_heads)
            )
        else:
            # [batch_size, num_heads, L', head_embed_dims]
            c_q, c_v = x_k, x_v
            # [batch_size, num_heads, L]
            index = torch.arange(H*W, device=x_k.device).reshape(1, 1, H*W)
            # [batch_size, num_heads, L, L']
            index_onehot = F.one_hot(index, H*W).float()
            # [batch_size, num_heads, L, L']
            relative_position_bias = Cluster.getMask((H, W), index_onehot)

        error_quantization = torch.tensor(0.0, device=x.device)

        # 计算注意力
        # [batch_size, num_heads, L, L']
        attn = torch.einsum('bhld,bhmd->bhlm', x_q, c_q)
        if not relative_position_bias is None:
            attn = attn + relative_position_bias
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # [batch_size, L, head_embed_dims]
        x = torch.einsum('bhlm,bhmd->bhld', attn, c_v).transpose(1, 2).reshape(batch_size, H, W, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, error_quantization


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
            self.w_msa = ClusterAttn(
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
            gla = gla,  # 全局注意力控制变量
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
                # shift=False if i % 2 == 0 else True,
                shift=False,        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 注意，这里关闭了循环移位 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
                gla=False if i % 2 == 0 else True,
                # gla=False,
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
