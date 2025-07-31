'''
架构说明: swin_debug_patent.py
1. 此架构为专利专用架构
2. 结合局部注意力机制与聚类稀疏全局注意力
3. 局部注意力添加固定的相对位置编码
4. q\k采用余弦相似度进行交互
5. 聚类过程采用自定义的梯度反向传播规则(还未实现)
6. 聚类过程和注意力计算过程都引入可学习的缩放系数

训练配置: 
    50 epochs, 
    batchsize 4 * 32 = 128, 
    stride(4*4), image_size = 224,
    swin-t预训练权重
top1: 
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
# from ..utils.vqt_tools import reduce_sum

# 聚类中心构造与向量量化
class Cluster(nn.Module):
    def __init__(self, iterations: int=1, head_embed_dims: int=32, qk_scale: float=15.0):
        '''
        window_size : 下采样倍率
        iterations  : k-means聚类次数
        qk_scale    : 基础缩放系数
        '''
        super(Cluster, self).__init__()
        self.iterations = iterations            # k-means聚类次数
        # # k-means聚类
        # self.proj = nn.Sequential(
        #     nn.Linear(head_embed_dims, 4*head_embed_dims, bias=False),
        #     nn.GELU(),
        #     nn.Linear(4*head_embed_dims,head_embed_dims, bias=False),
        #     nn.GELU()
        # )
        self.scale_base = qk_scale              # 固定的基础缩放系数
        self.scale = nn.Parameter(              # 可学习的缩放系数的缩放系数
            torch.tensor(1.0)
        )

    # 相对位置掩膜构造函数
    @staticmethod
    @torch.no_grad
    def getMask(size: tuple, index_onehot: torch.Tensor, gain: float=1.0):
        '''
        size: 特征图尺寸, (h, w)
        index_onehot: 聚类结果(每个像素对应的聚类中心的one-hot索引), [B, num_heads, L, S]
        gain: 增益系数
        '''
        assert type(size) == tuple, 'Data type of size in function <getMask> should be <tuple>!'
        assert size.__len__() == 2, 'Length of size should be 2!'
        coords_h = torch.arange(size[0])
        coords_w = torch.arange(size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 构造坐标窗口元素坐标索引，[2, h, w]
        # 一维化特征图像素坐标，[2, L]
        coords_featuremap = torch.flatten(coords, start_dim=1).float().to(index_onehot.device)
        # [B, num_heads, 2, L]
        coords_featuremap = coords_featuremap.reshape(
            1, 1, 2, -1
        ).repeat(index_onehot.shape[0], index_onehot.shape[1], 1, 1)
        # [B, num_heads, 1, S]
        index_onehot_sum = torch.sum(index_onehot, dim=-2, keepdim=True)
        index_onehot_sum[index_onehot_sum==0] = 1
        
        # 聚类中心坐标，[B, num_heads, 2, S]
        coords_clustercenter = torch.einsum(
            'bhcl,bhls->bhcs', coords_featuremap, index_onehot
        ) / index_onehot_sum
        # 对于没有对应元素的类，其距离设置为无穷远
        coords_clustercenter[coords_clustercenter < 1e-12] = torch.inf

        # 构造相对位置矩阵, 第一个矩阵是h方向的相对位置差, 第二个矩阵是w方向的相对位置差
        relative_coords = coords_featuremap[:, :, :, :, None] - coords_clustercenter[:, :, :, None, :]
        distance = torch.sqrt(                                      # [B, num_heads, L, S]
            torch.square(relative_coords[:,:,0,:,:]) + torch.square(relative_coords[:,:,1,:,:])
        )
        # exp操作用于处理distance中的0, [B, num_heads, L, S]
        distance_exp = torch.exp(distance)
        # 距离越远的token注意力增强越少(加性增强), 最大值为1*gain, 最小值可以接近0, [B, num_heads, L, S]
        mask = (1 / distance_exp) * gain
        return mask

    # 迭代更新码表
    def updateCenter(self, x: torch.Tensor, c: torch.Tensor, relative_position_bias: None):
        '''
        输入: 
        x: tensor, [batch_size, num_heads, L, head_embed_dims], 待聚类数据
        c: tensor, [batch_size, num_heads, L', head_embed_dims], 初始化聚类中心
        relative_position_bias: tensor, [batch_size, num_heads, L, L'], 相对位置编码
        输出：
        c_new: tensor, [B, num_heads, L', head_embed_dims], 更新后的聚类中心
        affinity: 
        '''
        # [B, num_heads, L, L']
        scale = self.scale_base * self.scale
        affinity_raw = torch.einsum('bhld,bhmd->bhlm', x, c)
        affinity = affinity_raw  * scale
        if not relative_position_bias is None:
            affinity = affinity + relative_position_bias
        
        # 增加的辅助损失
        if self.training:
            # affinity_aux = affinity_raw * scale.detach()
            # # 考虑到相对位置编码不会发生变化，因此可以不引入相对位置编码
            # if not relative_position_bias is None:
            #     affinity_aux = affinity_aux + relative_position_bias
            affinity_aux = None           # 不启用affinity loss
        else:
            affinity_aux = None

        # 使用掩膜进行非极大值抑制
        affinity_mask = torch.zeros_like(affinity)
        affinity_mask[affinity < affinity.max(dim=-1, keepdim=True)[0]] = -torch.inf
        # [B, num_heads, L, L']
        affinity_onehot = torch.softmax(affinity + affinity_mask, dim=-1)

        # 更新聚类中心
        # [B, num_heads, L', head_embed_dims]
        c_sum = torch.einsum('bhlm,bhld->bhmd', affinity_onehot, x)
        # 直接单位化，就不用affinity归一化了
        c_new = F.normalize(c_sum, dim=-1)

        return c_new, affinity_onehot, affinity_aux, scale.detach()

    # 聚类中心初始化与聚类中心更新
    def getCenter(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        '''
        输入
        x                       : tensor, [batch_size, num_heads, H, W, head_embed_dims], 待聚类的K矩阵
        delta_onehot_x          : [batch_size*num_heads, S, H_x, W_x], 来自上一个block的聚类结果, 将在这里进行更新
        输出
        delta_onehot            : tensor, [batch_size, num_heads, L, L'], one-hot矩阵
        c                       : tensor, [batch_size, num_heads, L', head_embed_dims], 聚类的K矩阵
        relative_position_bias  : tensor, [batch_size, num_heads, L, L']
        affinity                : tensor, [batch_size, num_heads, L, L'], 乘上缩放系数, 加上相对位置编码的余弦相似度
        scale                   : tensor, [1, ], 缩放系数, 用于辅助损失标签的构造
        '''
        batch_size, num_heads, H, W, head_embed_dims = x.shape
        L_ = delta_onehot_x.shape[1]
        x = x.reshape(batch_size, num_heads, -1, head_embed_dims)

        # 如果头数有增加（不能减少）（一般是翻一倍），则对现有聚类索引进行复制，[batch_size, num_heads, L, L']
        delta_onehot = delta_onehot_x.reshape(batch_size, -1, L_, H*W).transpose(-2, -1)
        delta_onehot = delta_onehot.repeat(1, num_heads//delta_onehot.shape[1], 1, 1)

        # 初始化聚类中心，[batch_size, num_heads, L', head_embed_dims], 如果某个聚类中心没有元素，则为零向量
        c = torch.einsum('bhlm,bhld->bhmd', delta_onehot, x)
        c = F.normalize(c, dim=-1)      # 单位化c_init, 便于后续进行余弦相似度计算
        relative_position_bias = self.getMask((H, W), delta_onehot)
        affinity_aux = None
        scale = None

        # 对x做一个特征映射

        # relative_position_bias = None
        for _ in range(self.iterations):
            # [batch_size, num_heads, L', head_embed_dims], [batch_size, num_heads, L, L']
            c, delta_onehot, affinity_aux, scale = self.updateCenter(x, c, relative_position_bias)
            # 更新相对位置编码(注意需要断开delta_onehot的梯度)
            relative_position_bias = self.getMask((H, W), delta_onehot)

        return delta_onehot, c, relative_position_bias, affinity_aux, scale

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        '''
        输入
        x                       : tensor, [batch_size, num_heads, H, W, head_embed_dims], 待聚类的K矩阵
        delta_onehot_x          : [batch_size*num_heads, S, H_x, W_x], 来自上一个block的聚类结果, 将在这里进行更新
        输出
        delta_onehot            : tensor, [batch_size, num_heads, L, L'], one-hot矩阵, 更新后的聚类结果
        c                       : tensor, [batch_size, num_heads, L', head_embed_dims], 聚类的K矩阵
        relative_position_bias  : tensor, [batch_size, num_heads, L, L']
        affinity_aux            : tensor, [batch_size, num_heads, L, L'], 乘上缩放系数, 加上相对位置编码的余弦相似度
        scale                   : tensor, [1, ], 缩放系数, 用于辅助损失标签的构造
        '''
        delta_onehot, c, relative_position_bias, affinity_aux, scale = self.getCenter(x, delta_onehot_x)
        return delta_onehot, c, relative_position_bias, affinity_aux, scale

# 混合分块注意力（局部）与聚类稀疏全局注意力的注意力机制
class MixAttention(BaseModule):
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
        self.window_size = window_size
        self.num_heads = num_heads
        self.id_stage = id_stage
        self.id_block = id_block
        self.num_heads = num_heads

        head_embed_dims = embed_dims // num_heads
        # # 固定缩放系数
        # self.scale_base = qk_scale or head_embed_dims**-0.5
        # 可学习的缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        self.scale = nn.Parameter(torch.tensor(1.0))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size - 1) * (2 * window_size - 1), self.num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size, self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        # 构造常规qkv
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        # 实例化聚类器
        self.cluster = Cluster(iterations=1, head_embed_dims=head_embed_dims, qk_scale=qk_scale)
        
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        # 慢速注意力需要这两项
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    # 矩阵分块
    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (B, num_windows, window_size*window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(
            B, H // window_size, window_size, W // window_size, window_size, C
        )
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.reshape(B, -1, window_size*window_size, C)
        return windows

    # 分块矩阵还原
    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (B, num_windows, num_heads, N, dim)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        B = windows.shape[0]
        window_size = self.window_size
        nW_h, nW_w = H // window_size, W // window_size

        x = windows.permute(
            0, 1, 3, 2, 4
        ).reshape(
            B, nW_h, nW_w, window_size, window_size, -1
        )
        x = x.permute(
            0, 1, 3, 2, 4, 5
        ).reshape(
            B, H, W, -1
        )
        return x

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        """
        Args:
            x (tensor): input features with shape of (B, H, W, C)
            delta_onehot_x: [batch_size*num_heads, S, H_x, W_x]
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        输出：
            x (tensor): [B, H, W, C]
        """
        B, H, W, C = x.shape
        L = H * W
        S = delta_onehot_x.shape[1]

        # x分块，不然后面要分别对q\k\v三个矩阵分块
        x_windows = self.window_partition(x)    # [B, nW, N, C]
        nW = x_windows.shape[1]                 # 总分块数量
        N = x_windows.shape[2]                  # 每个分块内的元素数

        qkv = self.qkv(x_windows).reshape(      # [B, nW, N, 3*C]
            B, nW, N, 3, self.num_heads, C // self.num_heads
        ).permute(                              # [B, nW, N, 3, num_heads, head_embed_dims]
            3, 0, 1, 4, 2, 5
        )                                       # [3, B, nW, num_heads, N, head_embed_dims]
        q_windows, k_windows, v_windows = qkv[0], qkv[1], qkv[2]
                                                # [B, nW, num_heads, N, head_embed_dims]
        # # 使用固定缩放系数
        # q = q * self.scale_base

        # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        q_windows = F.normalize(q_windows, dim=-1)
        k_windows = F.normalize(k_windows, dim=-1)
        q_windows = q_windows * self.scale * self.scale_base
                                                # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)
        # 计算局部注意力
        attn_local = (q_windows @ k_windows.transpose(-2, -1))
                                                # [B, nW, num_heads, N, N]
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            N, N, -1
        )                                       # [N, N, num_heads]
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()               # [num_heads, N, N]
        attn_local = attn_local + relative_position_bias.unsqueeze(0).unsqueeze(0)
                                                # [B, nW, num_heads, N, N]

        # 计算全局注意力
        k = self.window_reverse(
            k_windows, H, W
        ).reshape(                              # [B, H, W, C]
            B, H, W, self.num_heads, C // self.num_heads
        ).permute(                              # [B, H, W, num_heads, head_embed_dims]
            0, 3, 1, 2, 4
        ).contiguous()                          # [B, head_embed_dims, H, W, num_heads]

        v = self.window_reverse(
            v_windows, H, W
        ).reshape(                              # [B, H, W, C]
            B, L, self.num_heads, C // self.num_heads
        ).permute(                              # [B, L, num_heads, head_embed_dims]
            0, 2, 1, 3
        ).contiguous()                          # [B, num_heads, L, head_embed_dims]

        # [B, num_heads, L, S], [B, num_heads, S, head_embed_dims], [B, num_heads, L, S], [B, num_heads, L, S], [1,]
        delta_onehot, c_k, relative_position_bias_, affinity, scale = self.cluster(
            k, delta_onehot_x
        )
        delta_onehot_x = delta_onehot.transpose(-2, -1).reshape(-1, S, H, W)
        c_v = torch.einsum('bhls,bhld->bhsd', delta_onehot, v)
        delta_onehot_sum = torch.sum(delta_onehot, dim=-2).unsqueeze(-1)
        delta_onehot_sum[delta_onehot_sum == 0] = 1
        c_v = c_v / delta_onehot_sum            # [B, num_heads, S, head_embed_dims]
        attn_global = torch.einsum('bwhnd,bhsd->bwhns', q_windows, c_k)
                                                # [B, nW, num_heads, N, S]
        # 注意力拼接
        attn_mix = torch.cat([attn_local, attn_global], dim=-1)
                                                # [B, nW, num_heads, N, N+S]
        c_v = c_v.unsqueeze(
            1
        ).repeat(                               # [B, 1, num_heads, S, head_embed_dims]
            1, nW, 1, 1, 1
        )                                       # [B, nW, num_heads, S, head_embed_dims]
        v_mix = torch.cat([v_windows, c_v], dim=-2)
                                                # [B, nW, num_heads, N+S, head_embed_dims]

        attn_mix = self.softmax(attn_mix)
        attn_mix = self.attn_drop(attn_mix)
        x = torch.einsum('bwhns,bwhsd->bwhnd', attn_mix, v_mix)
                                                # [B, nW, num_heads, N, head_embed_dims]
        x = self.window_reverse(x, H, W)        # [B, H, W, C]

        x = self.proj(x)                        # [B, H, W, C]
        x = self.proj_drop(x)

        # print('id_stage: {}, id_block: {}, shape of x: {}'.format(self.id_stage, self.id_block, x.shape))


        return x, delta_onehot_x

# 整合各类注意力的模块
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

        self.window_size = window_size          # 仅分块局部注意力需要此参数
        self.w_msa = MixAttention(              # 混合注意力
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=self.window_size,
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

    def forward(self, query, hw_shape, delta_onehot_x: torch.Tensor, labels = None):
        '''
        delta_onehot_x: [batch_size*num_heads, S, H_x, W_x]
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
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b), mode='reflect')
                                                # F.pad的pad顺序为原始tensor从右往左的通道顺序，每个通道有2个参数
        delta_onehot_x = F.pad(delta_onehot_x, (0, pad_r, 0, pad_b), mode='reflect')
                                                # [batch_size*num_heads, S, H_pad, W_pad]
        H_pad, W_pad = query.shape[1], query.shape[2]

        x, delta_onehot_x = self.w_msa(query, delta_onehot_x)

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
            delta_onehot_x = delta_onehot_x[:, :, :H, :W]
                                                # [batch_size*num_heads, S, H_pad, W_pad]
        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x, delta_onehot_x

# attn + ffn
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

    def forward(self, x, hw_shape, delta_onehot_x: torch.Tensor, labels = None):
        '''
        delta_onehot_x: [batch_size*num_heads, S, H_x, W_x]
        '''

        # # 完全checkpoint
        # def _inner_forward(x, hw_shape, delta_onehot_x: torch.Tensor, labels = None):
        #     identity = x
        #     x = self.norm1(x)
        #     x, delta_onehot_x_dst = self.attn(x, hw_shape, delta_onehot_x, labels)

        #     x = x + identity

        #     identity = x
        #     x = self.norm2(x)
        #     x = self.ffn(x, identity=identity)

        #     return x, delta_onehot_x_dst

        # if self.with_cp and x.requires_grad:
        #     x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, hw_shape, delta_onehot_x, labels)
        # else:
        #     x, delta_onehot_x_dst = _inner_forward(x, hw_shape, delta_onehot_x, labels)

        # # 仅attn checkpoint
        # def _inner_forward(x, hw_shape, delta_onehot_x: torch.Tensor, labels = None):
        #     identity = x
        #     x = self.norm1(x)
        #     x, delta_onehot_x_dst = self.attn(x, hw_shape, delta_onehot_x, labels)
        #     x = x + identity
        #     return x, delta_onehot_x_dst

        # if self.with_cp and x.requires_grad:
        #     x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, hw_shape, delta_onehot_x, labels)
        # else:
        #     x, delta_onehot_x_dst = _inner_forward(x, hw_shape, delta_onehot_x, labels)
        # identity = x
        # x = self.norm2(x)
        # x = self.ffn(x, identity=identity)

        # 仅ffn checkpoint
        def _inner_forward(x):
            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)
            return x
        identity = x
        x = self.norm1(x)

        x, delta_onehot_x_dst = self.attn(x, hw_shape, delta_onehot_x, labels)
        x = x + identity

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x, delta_onehot_x_dst

# 多个完整transformer层构成的block
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

    def forward(self, x, hw_shape, delta_onehot_x: torch.Tensor, labels = None):
        '''
        x: []
        hw_shape: (H, W)
        delta_onehot_x: [batch_size*num_heads, S, H_x, W_x]
        '''
        for block in self.blocks:
            x, delta_onehot_x = block(x, hw_shape, delta_onehot_x, labels)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            # 对聚类索引进行下采样：
            # stage0->1, [128, 128, 16**2] -> [64, 64, 16**2]
            # stage1->2, [64, 64, 16**2]   -> [32, 32, 16**2]
            # stage2->3, [32, 32, 16**2]   -> [16, 16, 16**2]
            # print('------------------ shape of delta_onehot_x: {}\n ------------------'.format(delta_onehot_x.shape))
            # [batch_size*num_heads, S, H_x, W_x]
            delta_onehot_x = F.interpolate(delta_onehot_x, down_hw_shape, mode='nearest')
            return x_down, down_hw_shape, x, hw_shape, delta_onehot_x
        else:
            return x, hw_shape, x, hw_shape, delta_onehot_x

@MODELS.register_module()
class SwinTransformerPatent(BaseModule):
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

    # 构造初始聚类列表
    @torch.no_grad
    def initIndex(self, shape_x: tuple, shape_c=None, stride=None, device: torch.device='cpu') -> torch.Tensor:
        assert shape_c is None or stride is None, '不能同时指定聚类中心尺寸和stride'
        H_x, W_x = shape_x
        # 获取聚类中心尺寸
        if not shape_c is None:
            H_c, W_c = shape_c
        else:
            pad_h = (stride[0] - H_x % stride[0]) % stride[0]
            pad_w = (stride[1] - W_x % stride[1]) % stride[1]
            H_c = (H_x + pad_h) // stride[0]
            W_c = (W_x + pad_w) // stride[1]
        
        # 构造索引值
        delta_index_c = torch.arange(
            0, H_c*W_c, device=device
        ).reshape(
            1, 1, H_c, W_c
        ).float()                                           # [1, 1, H_c, W_c]
        delta_index_x = F.interpolate(
            delta_index_c, shape_x, mode='nearest'
        ).long()                                            # [1, 1, H_x, W_x]
        delta_onehot_x = F.one_hot(
            delta_index_x, H_c*W_c
        ).permute(
            0, 1, 4, 2, 3
        ).reshape(
            1, H_c*W_c, H_x, W_x                            # [batch_size*num_heads, S, H, W]
        ).float()                                           # [1, H_c*W_c, H_x, W_x]

        return delta_onehot_x

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

        if labels is None:
            # 获取初始聚类索引值, [B, S, H_x, W_x]
            # delta_onehot_x = self.initIndex(hw_shape, shape_c=(16, 16), device=x.device).repeat(x.shape[0], 1, 1, 1)
            delta_onehot_x = self.initIndex(hw_shape, stride=(4, 4), device=x.device).repeat(x.shape[0], 1, 1, 1)
            # print('------------------ shape of delta_onehot_x: {}\n ------------------'.format(delta_onehot_x.shape))
        else:
            H_x, W_x = hw_shape
            stride=(8, 8)
            pad_h = (stride[0] - H_x % stride[0]) % stride[0]
            pad_w = (stride[1] - W_x % stride[1]) % stride[1]
            H_c = (H_x + pad_h) // stride[0]
            W_c = (W_x + pad_w) // stride[1]
            labels_subsample = F.interpolate(labels, hw_shape, mode='nearest').long()   # [batch_size, 1, H_x, W_x]
            delta_onehot_x = F.one_hot(
                labels_subsample, H_c*W_c
            ).permute(
                0, 1, 4, 2, 3
            ).reshape(
                -1, H_c*W_c, H_x, W_x                                                   # [batch_size, S, H, W]
            ).float()

        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape, delta_onehot_x = stage(x, hw_shape, delta_onehot_x, labels)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(
                    -1, *out_hw_shape,
                    self.num_features[i]
                ).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)

