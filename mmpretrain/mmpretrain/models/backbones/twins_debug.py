# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, normal_init, trunc_normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpretrain.registry import MODELS
from ..utils import ConditionalPositionEncoding, MultiheadAttention

iterations = 1      # k-means聚类次数

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

# 基于聚类的量化注意力
class ClusterAttn(nn.Module):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
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
        attn_drop_rate=0.,
        proj_drop_rate=0.,
        dropout_layer=dict(type='DropPath', drop_prob=0.),
        qkv_bias=True,
        qk_scale=None,
    ):
        super(ClusterAttn, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        # 可学习的缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 构造常规qkv
        self.q = nn.Linear(embed_dims, embed_dims * 1, bias=qkv_bias)
        self.kv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)

        # 实例化聚类器
        self.cluster = Cluster(iterations, head_embed_dims, qk_scale)
        
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        # 慢速注意力需要这两项
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
        
        self.out_drop = build_dropout(dropout_layer)
    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        """
        Args(实际给的):
            x               : [B, L, dims], L是包含cls_token的序列长度
            delta_onehot_x  : [B, S, H_x, W_x], S是聚类中心的数量
        Args(需要的):
            x (tensor)      : input features with shape of (B, H, W, C)
            delta_onehot_x  : [batch_size*num_heads, S, H_x, W_x]

        输出(实际的):
            x (tensor)      : input features with shape of (B, H, W, C)
            delta_onehot_x  : [batch_size*num_heads, S, H_x, W_x]
        输出(需要的):
            x               : [B, L, dims], L是包含cls_token的序列长度
            delta_onehot_x  : [B, S, H_x, W_x], S是聚类中心的数量
        """
        H, W = delta_onehot_x.shape[-2:]

        batch_size, L, C = x.shape      # 注意这里L = num_cls_tokens + H * W
        L_ = delta_onehot_x.shape[1]
        # [3, batch_size, num_heads, L, head_embed_dims]
        q = self.q(x).reshape(batch_size, L, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(batch_size, L, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, L, head_embed_dims]
        q, k, v = q[0], kv[0], kv[1]
        # [batch_size, num_heads, H*W, head_embed_dims]
        k = k[:, :, -H*W:, :]           # 排除辅助tokens
        v = v[:, :, -H*W:, :]

        # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q * self.scale * self.scale_base            # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

        # [batch_size, num_heads, L, L'], [batch_size, num_heads, L', head_embed_dims], [batch_size, num_heads, L, L'], [batch_size, num_heads, L, L'], [1,]
        delta_onehot, c_k, relative_position_bias, affinity, scale = self.cluster(
            k.reshape(batch_size, self.num_heads, H, W, C // self.num_heads), delta_onehot_x
        )

        delta_onehot_x = delta_onehot.transpose(-2, -1).reshape(-1, L_, H, W)

        # # 量化损失计算(会带来负面影响，导致scale系数很大)
        # k_hat = torch.einsum('bhlm,bhmd->bhld', delta_onehot, c)
        # # L2范数量化损失度量
        # error_quantization = torch.norm(k - k_hat, dim=-1).square().mean()
        # # 余弦相似度量化损失度量
        # error_quantization = (1 - torch.einsum('bhld,bhmd->bhlm', k, k_hat)).mean()
        error_quantization = torch.tensor(0.0, device=x.device)

        # # [batch_size, num_heads, L, L']
        # qcT = torch.einsum('bhld,bhmd->bhlm', q, c)
        # if not relative_position_bias is None:
        #     qcT = qcT + relative_position_bias
        # qcT = qcT - qcT.max(dim=-1, keepdim=True)[0]
        # qcT_exp = torch.exp(qcT)

        # # 计算softmax分子
        # # [batch_size, num_heads, L', head_embed_dims]
        # deltaTv = torch.einsum('bhlm,bhld->bhmd', delta_onehot, v)
        # # [batch_size, num_heads, L, head_embed_dims]
        # numerator = torch.einsum('bhlm,bhmd->bhld', qcT_exp, deltaTv)
        # # 计算softmax分母
        # # [batch_size, num_heads, L']
        # deltaT1 = torch.einsum('bhlm->bhm', delta_onehot)
        # # [batch_size, num_heads, L, 1]
        # denominator = torch.einsum('bhlm,bhm->bhl', qcT_exp, deltaT1).unsqueeze(-1)
        # denominator[denominator==0] = 1e-6                              # 防止除以0

        # # 计算注意力加权的v
        # # [batch_size, num_heads, L, head_embed_dims]
        # x = numerator / denominator
        # x = x.transpose(1, 2).reshape(batch_size, H, W, C)

        c_v = torch.einsum('bhlm,bhld->bhmd', delta_onehot, v)
        delta_onehot_sum = torch.sum(delta_onehot, dim=-2).unsqueeze(-1)
        delta_onehot_sum[delta_onehot_sum == 0] = 1
        c_v = c_v / delta_onehot_sum

        # 慢速
        attn = (q @ c_k.transpose(-2, -1))                                  # [batch_size, num_heads, L, L']
        # if not relative_position_bias is None:
        #     attn = attn + relative_position_bias
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ c_v).transpose(1, 2).reshape(batch_size, L, C)

        # # 使用Flash-Attention 1.x的API
        # q = q.transpose(1, 2).reshape(batch_size*L, self.num_heads, C // self.num_heads).half()
        # c_k = c_k.transpose(1, 2).reshape(batch_size*L_, self.num_heads, C // self.num_heads).half()
        # c_v = c_v.transpose(1, 2).reshape(batch_size*L_, self.num_heads, C // self.num_heads).half()
        # cu_seqlens_q = torch.arange(0, (batch_size + 1) * L, step=L, dtype=torch.int32, device=q.device)
        # cu_seqlens_kv = torch.arange(0, (batch_size + 1) * L_, step=L_, dtype=torch.int32, device=q.device)
        # x = flash_attn_unpadded_func(
        #     q, c_k, c_v, 
        #     cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv, 
        #     max_seqlen_q=L, max_seqlen_k=L_, 
        #     dropout_p=self.attn_drop_rate if self.training else 0.0, 
        #     softmax_scale=1.0
        # ).reshape(batch_size, L, C).float()

        # # 使用Flash-Attention 2.x的API
        # q = q.transpose(1, 2).half()       # [batch_size, L, num_heads, head_embed_dims]
        # c_k = c_k.transpose(1, 2).half()   # [batch_size, L_, num_heads, head_embed_dims]
        # c_v = c_v.transpose(1, 2).half()   # [batch_size, L_, num_heads, head_embed_dims]
        # x = flash_attn_func(q, c_k, c_v, dropout_p=self.attn_drop_rate if self.training else 0.0, softmax_scale=1.0)  # [batch_size, L, num_heads, head_embed_dims]
        # x = x.reshape(batch_size, L, C).float()

        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.out_drop(x)

        # return x, delta_onehot_x, error_quantization, affinity, scale
        return x, delta_onehot_x

class GlobalSubsampledAttention(MultiheadAttention):
    """Global Sub-sampled Attention (GSA) module.

    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        input_dims (int, optional): The input dimension, and if None,
            use ``embed_dims``. Defaults to None.
        attn_drop (float): Dropout rate of the dropout layer after the
            attention calculation of query and key. Defaults to 0.
        proj_drop (float): Dropout rate of the dropout layer after the
            output projection. Defaults to 0.
        dropout_layer (dict): The dropout config before adding the shortcut.
            Defaults to ``dict(type='Dropout', drop_prob=0.)``.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        proj_bias (bool) If True, add a learnable bias to output projection.
            Defaults to True.
        v_shortcut (bool): Add a shortcut from value to output. It's usually
            used if ``input_dims`` is different from ``embed_dims``.
            Defaults to False.
        sr_ratio (float): The ratio of spatial reduction in attention modules.
            Defaults to 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 norm_cfg=dict(type='LN'),
                 qkv_bias=True,
                 sr_ratio=1,
                 **kwargs):
        super(GlobalSubsampledAttention,
              self).__init__(embed_dims, num_heads, **kwargs)

        self.qkv_bias = qkv_bias
        self.q = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.kv = nn.Linear(self.input_dims, embed_dims * 2, bias=qkv_bias)

        # remove self.qkv, here split into self.q, self.kv
        delattr(self, 'qkv')

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # use a conv as the spatial-reduction operation, the kernel_size
            # and stride in conv are equal to the sr_ratio.
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio)
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        assert H * W == N, 'The product of h and w of hw_shape must be N, ' \
                           'which is the 2nd dim number of the input Tensor x.'

        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, *hw_shape)  # BNC_2_BCHW
            x = self.sr(x)
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # BCHW_2_BNC
            x = self.norm(x)

        kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                self.head_dims).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_drop = self.attn_drop if self.training else 0.
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

# 在这里修改全局注意力类型
class GSAEncoderLayer(BaseModule):
    """Implements one encoder layer with GlobalSubsampledAttention(GSA).

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (float): The ratio of spatial reduction in attention modules.
            Defaults to 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        num_fcs=2,
        qkv_bias=True,
        qk_scale=15.0,                  # cluster attention使用的缩放系数
        attn_type='origin',             # Global attention类型['origin', 'clusterattn']
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        sr_ratio=1.,
        with_cp=False,
        init_cfg=None
    ):
        super(GSAEncoderLayer, self).__init__(init_cfg=init_cfg)
        self.attn_type = attn_type
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]

        if attn_type == 'origin':
            self.attn = GlobalSubsampledAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                qkv_bias=qkv_bias,
                norm_cfg=norm_cfg,
                sr_ratio=sr_ratio
            )
        elif attn_type == 'clusterattn':
            self.attn = ClusterAttn(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop_rate=attn_drop_rate,
                proj_drop_rate=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=0.0), # 因为外层已经启用了drop path，所以内部就不需要了
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )
        else:
            raise NotImplementedError
        
        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        # # 完全checkpoints
        # def _inner_forward(x: torch.Tensor, delta_onehot_x: torch.Tensor):
        #     # attn
        #     identity = x
        #     x = self.norm1(x)
        #     if self.attn_type == 'clusterattn':
        #         x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)        # [B, L, dims], [B, S, H_x, W_x]
        #     else:
        #         H, W = delta_onehot_x.shape[-2:]
        #         x = self.attn(x, (H, W))                                  # [B, L, dims]
        #         delta_onehot_x_dst = delta_onehot_x
        #     x = identity + self.drop_path(x)
        #     # ffn
        #     x = x + self.drop_path(self.ffn(self.norm2(x)))
        #     return x, delta_onehot_x_dst
        # if self.with_cp and x.requires_grad:
        #     x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        # else:
        #     x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)
        # return x, delta_onehot_x_dst

        # 仅ffn部分checkpoints
        def _inner_forward(x: torch.Tensor):
            x = x + self.drop_path(self.ffn(self.norm2(x)))
            return x
        # attn
        identity = x
        x = self.norm1(x)
        if self.attn_type == 'clusterattn':
            x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)    # [B, L, dims], [B, S, H_x, W_x]
        else:
            H, W = delta_onehot_x.shape[-2:]
            x = self.attn(x, (H, W))                                  # [B, L, dims]
            delta_onehot_x_dst = delta_onehot_x
        x = identity + self.drop_path(x)
        # ffn
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x, delta_onehot_x_dst

# swin式局部注意力块（仅attn）
class LocallyGroupedSelfAttention(BaseModule):
    """Locally-grouped Self Attention (LSA) module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: False.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        window_size(int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 window_size=1,
                 init_cfg=None):
        super(LocallyGroupedSelfAttention, self).__init__(init_cfg=init_cfg)

        assert embed_dims % num_heads == 0, \
            f'dim {embed_dims} should be divided by num_heads {num_heads}'

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.window_size = window_size

    def forward(self, x, hw_shape):
        B, N, C = x.shape
        H, W = hw_shape
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of Local-groups
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        # calculate attention mask for LSA
        Hp, Wp = x.shape[1:-1]
        _h, _w = Hp // self.window_size, Wp // self.window_size
        mask = torch.zeros((1, Hp, Wp), device=x.device)
        mask[:, -pad_b:, :].fill_(1)
        mask[:, :, -pad_r:].fill_(1)

        # [B, _h, _w, window_size, window_size, C]
        x = x.reshape(B, _h, self.window_size, _w, self.window_size,
                      C).transpose(2, 3)
        mask = mask.reshape(1, _h, self.window_size, _w,
                            self.window_size).transpose(2, 3).reshape(
                                1, _h * _w,
                                self.window_size * self.window_size)
        # [1, _h*_w, window_size*window_size, window_size*window_size]
        attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                          float(-1000.0)).masked_fill(
                                              attn_mask == 0, float(0.0))

        # [3, B, _w*_h, nhead, window_size*window_size, dim]
        qkv = self.qkv(x).reshape(B, _h * _w,
                                  self.window_size * self.window_size, 3,
                                  self.num_heads, C // self.num_heads).permute(
                                      3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, _h*_w, n_head, window_size*window_size, window_size*window_size]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + attn_mask.unsqueeze(2)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.window_size,
                                                  self.window_size, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.window_size,
                                         _w * self.window_size, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# swin层（attn+ffn）
class LSAEncoderLayer(BaseModule):
    """Implements one encoder layer with LocallyGroupedSelfAttention(LSA).

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
           Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): Enable bias for qkv if True. Default: True
        qk_scale (float | None, optional): Override default qk scale of
           head_dim ** -0.5 if set. Default: None.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        window_size (int): Window size of LSA. Default: 1.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 qk_scale=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=1,
                 init_cfg=None):

        super(LSAEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.norm1 = build_norm_layer(norm_cfg, embed_dims, postfix=1)[1]
        self.attn = LocallyGroupedSelfAttention(
            embed_dims, 
            num_heads,
            qkv_bias, 
            qk_scale,
            attn_drop_rate, 
            drop_rate,
            window_size
        )

        self.norm2 = build_norm_layer(norm_cfg, embed_dims, postfix=2)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=False)

        self.drop_path = build_dropout(
            dict(type='DropPath', drop_prob=drop_path_rate)
        ) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        H, W = delta_onehot_x.shape[-2:]
        x = x + self.drop_path(self.attn(self.norm1(x), (H, W)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x, delta_onehot_x


@MODELS.register_module()
class PCPVTMod(BaseModule):
    """The backbone of Twins-PCPVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        arch (dict, str): PCPVT architecture, a str value in arch zoo or a
            detailed configuration dict with 7 keys, and the length of all the
            values in dict should be the same:

            - depths (List[int]): The number of encoder layers in each stage.
            - embed_dims (List[int]): Embedding dimension in each stage.
            - patch_sizes (List[int]): The patch sizes in each stage.
            - num_heads (List[int]): Numbers of attention head in each stage.
            - strides (List[int]): The strides in each stage.
            - mlp_ratios (List[int]): The ratios of mlp in each stage.
            - sr_ratios (List[int]): The ratios of GSA-encoder layers in each
              stage.

        in_channels (int): Number of input channels. Defaults to 3.
        out_indices (tuple[int]): Output from which stages.
            Defaults to ``(3, )``.
        qkv_bias (bool): Enable bias for qkv if True. Defaults to False.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Defaults to 0.0
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        norm_after_stage(bool, List[bool]): Add extra norm after each stage.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import PCPVT
        >>> import torch
        >>> pcpvt_cfg = {'arch': "small",
        >>>              'norm_after_stage': [False, False, False, True]}
        >>> model = PCPVT(**pcpvt_cfg)
        >>> x = torch.rand(1, 3, 224, 224)
        >>> outputs = model(x)
        >>> print(outputs[-1].shape)
        torch.Size([1, 512, 7, 7])
        >>> pcpvt_cfg['norm_after_stage'] = [True, True, True, True]
        >>> pcpvt_cfg['out_indices'] = (0, 1, 2, 3)
        >>> model = PCPVT(**pcpvt_cfg)
        >>> outputs = model(x)
        >>> for feat in outputs:
        >>>     print(feat.shape)
        torch.Size([1, 64, 56, 56])
        torch.Size([1, 128, 28, 28])
        torch.Size([1, 320, 14, 14])
        torch.Size([1, 512, 7, 7])
    """
    arch_zoo = {
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims':    [64, 128, 320, 512],
                         'depths':        [3, 4, 6, 3],
                         'num_heads':     [1, 2, 5, 8],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [8, 8, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims':    [64, 128, 320, 512],
                         'depths':        [3, 4, 18, 3],
                         'num_heads':     [1, 2, 5, 8],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [8, 8, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims':    [64, 128, 320, 512],
                         'depths':        [3, 8, 27, 3],
                         'num_heads':     [1, 2, 5, 8],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [8, 8, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1]}),
    }   # yapf: disable

    essential_keys = {
        'embed_dims', 'depths', 'num_heads', 'patch_sizes', 'strides',
        'mlp_ratios', 'sr_ratios'
    }

    def __init__(
        self,
        arch,
        in_channels=3,
        out_indices=(3, ),
        qkv_bias=False,
        qk_scale=None,
        attn_type='origin',
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_cfg=dict(type='LN'),
        norm_after_stage=False,
        with_cp=False,
        init_cfg=None
    ):
        super(PCPVTMod, self).__init__(init_cfg=init_cfg)
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            assert isinstance(arch, dict) and (
                set(arch) == self.essential_keys
            ), f'Custom arch needs a dict with keys {self.essential_keys}.'
            self.arch_settings = arch

        self.depths = self.arch_settings['depths']
        self.embed_dims = self.arch_settings['embed_dims']
        self.patch_sizes = self.arch_settings['patch_sizes']
        self.strides = self.arch_settings['strides']
        self.mlp_ratios = self.arch_settings['mlp_ratios']
        self.num_heads = self.arch_settings['num_heads']
        self.sr_ratios = self.arch_settings['sr_ratios']

        self.num_extra_tokens = 0  # there is no cls-token in Twins
        self.num_stage = len(self.depths)
        for key, value in self.arch_settings.items():
            assert isinstance(value, list) and len(value) == self.num_stage, (
                'Length of setting item in arch dict must be type of list and'
                ' have the same length.')

        # patch_embeds
        self.patch_embeds = ModuleList()
        self.position_encoding_drops = ModuleList()
        self.stages = ModuleList()

        for i in range(self.num_stage):
            # use in_channels of the model in the first stage
            if i == 0:
                stage_in_channels = in_channels
            else:
                stage_in_channels = self.embed_dims[i - 1]

            self.patch_embeds.append(
                PatchEmbed(
                    in_channels=stage_in_channels,
                    embed_dims=self.embed_dims[i],
                    conv_type='Conv2d',
                    kernel_size=self.patch_sizes[i],
                    stride=self.strides[i],
                    padding='corner',
                    norm_cfg=dict(type='LN')))

            self.position_encoding_drops.append(nn.Dropout(p=drop_rate))

        # PEGs 位置编码，每个stage单独一个位置编码
        self.position_encodings = ModuleList([
            ConditionalPositionEncoding(embed_dim, embed_dim)
            for embed_dim in self.embed_dims
        ])

        # stochastic depth
        total_depth = sum(self.depths)
        self.dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule
        cur = 0

        for k in range(len(self.depths)):
            _block = ModuleList([
                GSAEncoderLayer(
                    embed_dims=self.embed_dims[k],
                    num_heads=self.num_heads[k],
                    feedforward_channels=self.mlp_ratios[k] * self.embed_dims[k],
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=self.dpr[cur + i],
                    num_fcs=2,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_type=attn_type,
                    act_cfg=dict(type='GELU'),
                    norm_cfg=norm_cfg,
                    sr_ratio=self.sr_ratios[k],
                    with_cp=with_cp,
                ) for i in range(self.depths[k])
            ])
            self.stages.append(_block)
            cur += self.depths[k]

        self.out_indices = out_indices

        assert isinstance(norm_after_stage, (bool, list))
        if isinstance(norm_after_stage, bool):
            self.norm_after_stage = [norm_after_stage] * self.num_stage
        else:
            self.norm_after_stage = norm_after_stage
        assert len(self.norm_after_stage) == self.num_stage, \
            (f'Number of norm_after_stage({len(self.norm_after_stage)}) should'
             f' be equal to the number of stages({self.num_stage}).')

        for i, has_norm in enumerate(self.norm_after_stage):
            assert isinstance(has_norm, bool), 'norm_after_stage should be ' \
                                               'bool or List[bool].'
            if has_norm and norm_cfg is not None:
                norm_layer = build_norm_layer(norm_cfg, self.embed_dims[i])[1]
            else:
                norm_layer = nn.Identity()

            self.add_module(f'norm_after_stage{i}', norm_layer)

    def init_weights(self):
        if self.init_cfg is not None:
            super(PCPVTMod, self).init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

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

    def forward(self, x):
        # print('shape of input:', x.shape)
        outputs = list()

        b = x.shape[0]

        for i in range(self.num_stage):
            x, hw_shape = self.patch_embeds[i](x)

            if i == 0:  # 仅在第一个stage构造初始聚类分布
                # 获取初始聚类索引值, [B, S, H_x, W_x], pvt网络中, 初始特征图尺寸为输入图片的1/4, 所以对于224*224的输入图片尺寸, stride设置为(4, 4)
                delta_onehot_x = self.initIndex(hw_shape, stride=(4, 4), device=x.device).repeat(x.shape[0], 1, 1, 1)
            else:       # 其余stage沿用之前的聚类分布（下采样）
                delta_onehot_x = F.interpolate(delta_onehot_x, hw_shape, mode='nearest')

            # print('shape of x in stage {}: {} == {}'.format(i, x.shape, hw_shape))
            h, w = hw_shape
            x = self.position_encoding_drops[i](x)
            for j, blk in enumerate(self.stages[i]):            # stage
                x, delta_onehot_x = blk(x, delta_onehot_x)
                
                if j == 0:                                      # 每个stage的第一个layer后进行一次位置编码
                    x = self.position_encodings[i](x, hw_shape)

            norm_layer = getattr(self, f'norm_after_stage{i}')
            x = norm_layer(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

            if i in self.out_indices:
                outputs.append(x)

        return tuple(outputs)


@MODELS.register_module()
class SVTMod(PCPVTMod):
    """The backbone of Twins-SVT.

    This backbone is the implementation of `Twins: Revisiting the Design
    of Spatial Attention in Vision Transformers
    <https://arxiv.org/abs/1512.03385>`_.

    Args:
        arch (dict, str): SVT architecture, a str value in arch zoo or a
            detailed configuration dict with 8 keys, and the length of all the
            values in dict should be the same:

            - depths (List[int]): The number of encoder layers in each stage.
            - embed_dims (List[int]): Embedding dimension in each stage.
            - patch_sizes (List[int]): The patch sizes in each stage.
            - num_heads (List[int]): Numbers of attention head in each stage.
            - strides (List[int]): The strides in each stage.
            - mlp_ratios (List[int]): The ratios of mlp in each stage.
            - sr_ratios (List[int]): The ratios of GSA-encoder layers in each
              stage.
            - windiow_sizes (List[int]): The window sizes in LSA-encoder layers
              in each stage.

        in_channels (int): Number of input channels. Defaults to 3.
        out_indices (tuple[int]): Output from which stages.
            Defaults to (3, ).
        qkv_bias (bool): Enable bias for qkv if True. Defaults to False.
        drop_rate (float): Dropout rate. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.0
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.2.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        norm_after_stage(bool, List[bool]): Add extra norm after each stage.
            Defaults to False.
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.

    Examples:
        >>> from mmpretrain.models import SVT
        >>> import torch
        >>> svt_cfg = {'arch': "small",
        >>>            'norm_after_stage': [False, False, False, True]}
        >>> model = SVT(**svt_cfg)
        >>> x = torch.rand(1, 3, 224, 224)
        >>> outputs = model(x)
        >>> print(outputs[-1].shape)
        torch.Size([1, 512, 7, 7])
        >>> svt_cfg["out_indices"] = (0, 1, 2, 3)
        >>> svt_cfg["norm_after_stage"] = [True, True, True, True]
        >>> model = SVT(**svt_cfg)
        >>> output = model(x)
        >>> for feat in output:
        >>>     print(feat.shape)
        torch.Size([1, 64, 56, 56])
        torch.Size([1, 128, 28, 28])
        torch.Size([1, 320, 14, 14])
        torch.Size([1, 512, 7, 7])
    """
    arch_zoo = {
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims':    [64, 128, 256, 512],
                         'depths':        [2, 2, 10, 4],
                         'num_heads':     [2, 4, 8, 16],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [4, 4, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1],
                         'window_sizes':  [7, 7, 7, 7]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims':    [96, 192, 384, 768],
                         'depths':        [2, 2, 18, 2],
                         'num_heads':     [3, 6, 12, 24],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [4, 4, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1],
                         'window_sizes':  [7, 7, 7, 7]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims':    [128, 256, 512, 1024],
                         'depths':        [2, 2, 18, 2],
                         'num_heads':     [4, 8, 16, 32],
                         'patch_sizes':   [4, 2, 2, 2],
                         'strides':       [4, 2, 2, 2],
                         'mlp_ratios':    [4, 4, 4, 4],
                         'sr_ratios':     [8, 4, 2, 1],
                         'window_sizes':  [7, 7, 7, 7]}),
    }  # yapf: disable

    essential_keys = {
        'embed_dims', 'depths', 'num_heads', 'patch_sizes', 'strides',
        'mlp_ratios', 'sr_ratios', 'window_sizes'
    }

    def __init__(
        self,
        arch,
        in_channels=3,
        out_indices=(3, ),
        qkv_bias=False,
        qk_scale=15.0,
        attn_type='origin',
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        norm_cfg=dict(type='LN'),
        norm_after_stage=False,
        with_cp=False,
        init_cfg=None
    ):
        super(SVTMod, self).__init__(
            arch, 
            in_channels, 
            out_indices, 
            qkv_bias,
            qk_scale,
            attn_type,
            drop_rate, 
            attn_drop_rate, 
            drop_path_rate,
            norm_cfg, 
            norm_after_stage, 
            with_cp,
            init_cfg
        )

        self.window_sizes = self.arch_settings['window_sizes']

        for k in range(self.num_stage):
            for i in range(self.depths[k]):
                # in even-numbered layers of each stage, replace GSA with LSA
                if i % 2 == 0:
                    ffn_channels = self.mlp_ratios[k] * self.embed_dims[k]
                    self.stages[k][i] = LSAEncoderLayer(
                        embed_dims=self.embed_dims[k],
                        num_heads=self.num_heads[k],
                        feedforward_channels=ffn_channels,
                        drop_rate=drop_rate,
                        norm_cfg=norm_cfg,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=self.dpr[sum(self.depths[:k])+i],
                        qkv_bias=qkv_bias,
                        window_size=self.window_sizes[k]
                    )
