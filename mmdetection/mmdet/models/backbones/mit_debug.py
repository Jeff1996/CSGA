# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import MultiheadAttention, PatchEmbed
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.model.weight_init import (constant_init, normal_init, trunc_normal_init)

from mmdet.registry import MODELS

# # Flash-Attneion 1.x
# from flash_attn.flash_attn_interface import flash_attn_unpadded_func

# # Flash-Attneion 2.x
# from flash_attn import flash_attn_func

iterations = 1      # k-means聚类次数

def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)

def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()

class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)

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
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        # 实例化聚类器
        self.cluster = Cluster(iterations, head_embed_dims, qk_scale)
        
        # self.attn_drop = nn.Dropout(attn_drop_rate)
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

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

        batch_size, L, C = x.shape      # 
        L_ = delta_onehot_x.shape[1]
        # [3, batch_size, num_heads, L, head_embed_dims]
        qkv = self.qkv(x).reshape(batch_size, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, L, head_embed_dims]
        q, k, v = qkv[0], qkv[1], qkv[2]
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
        if not relative_position_bias is None:
            attn = attn + relative_position_bias
        attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
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


class EfficientMultiheadAttention(MultiheadAttention):
    """An implementation of Efficient Multi-head Attention of Segformer.

    This module is modified from MultiheadAttention which is a module from
    mmcv.cnn.bricks.transformer.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut. Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        qkv_bias (bool): enable bias for qkv if True. Default True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.,
        proj_drop=0.,
        dropout_layer=None,
        init_cfg=None,
        batch_first=True,
        qkv_bias=False,
        norm_cfg=dict(type='LN'),
        sr_ratio=1
    ):
        super().__init__(
            embed_dims,
            num_heads,
            attn_drop,
            proj_drop,
            dropout_layer=dropout_layer,
            init_cfg=init_cfg,
            batch_first=batch_first,
            bias=qkv_bias
        )

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = Conv2d(
                in_channels=embed_dims,
                out_channels=embed_dims,
                kernel_size=sr_ratio,
                stride=sr_ratio
            )
            # The ret[0] of build_norm_layer is norm name.
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        # handle the BC-breaking from https://github.com/open-mmlab/mmcv/pull/1418 # noqa
        from mmseg import digit_version, mmcv_version
        if mmcv_version < digit_version('1.3.17'):
            warnings.warn('The legacy version of forward function in'
                          'EfficientMultiheadAttention is deprecated in'
                          'mmcv>=1.3.17 and will no longer support in the'
                          'future. Please upgrade your mmcv.')
            self.forward = self.legacy_forward

    def forward(self, x, hw_shape, identity=None):

        x_q = x
        if self.sr_ratio > 1:               # 下采样
            x_kv = nlc_to_nchw(x, hw_shape) # 序列二维化
            x_kv = self.sr(x_kv)            # 下采样
            x_kv = nchw_to_nlc(x_kv)        # 二位特征图一维序列化
            x_kv = self.norm(x_kv)          # LayerNorm
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            x_q = x_q.transpose(0, 1)
            x_kv = x_kv.transpose(0, 1)

        out = self.attn(query=x_q, key=x_kv, value=x_kv)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

    def legacy_forward(self, x, hw_shape, identity=None):
        """multi head attention forward in mmcv version < 1.3.17."""

        x_q = x
        if self.sr_ratio > 1:
            x_kv = nlc_to_nchw(x, hw_shape)
            x_kv = self.sr(x_kv)
            x_kv = nchw_to_nlc(x_kv)
            x_kv = self.norm(x_kv)
        else:
            x_kv = x

        if identity is None:
            identity = x_q

        # `need_weights=True` will let nn.MultiHeadAttention
        # `return attn_output, attn_output_weights.sum(dim=1) / num_heads`
        # The `attn_output_weights.sum(dim=1)` may cause cuda error. So, we set
        # `need_weights=False` to ignore `attn_output_weights.sum(dim=1)`.
        # This issue - `https://github.com/pytorch/pytorch/issues/37583` report
        # the error that large scale tensor sum operation may cause cuda error.
        out = self.attn(query=x_q, key=x_kv, value=x_kv, need_weights=False)[0]

        return identity + self.dropout_layer(self.proj_drop(out))


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Segformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        qkv_bias (bool): enable bias for qkv if True.
            Default: True.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        qkv_bias=True,
        qk_scale=None,
        attn_type = 'origin',
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        batch_first=True,
        sr_ratio=1,
        with_cp=False
    ):
        super().__init__()

        # The ret[0] of build_norm_layer is norm name.
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.attn_type = attn_type
        if attn_type == 'origin':
            self.attn = EfficientMultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                batch_first=batch_first,
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
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )
        else:
            raise NotImplementedError

        # The ret[0] of build_norm_layer is norm name.
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg)

        self.with_cp = with_cp

    # def forward(self, x, hw_shape):

    #     def _inner_forward(x):
    #         x = self.attn(self.norm1(x), hw_shape, identity=x)
    #         x = self.ffn(self.norm2(x), hw_shape, identity=x)
    #         return x

    #     if self.with_cp and x.requires_grad:
    #         x = cp.checkpoint(_inner_forward, x)
    #     else:
    #         x = _inner_forward(x)
    #     return x

    def forward(self, x, delta_onehot_x=None):
        '''
        x               : [B, L, dims], L是包含cls_token的序列长度
        delta_onehot_x  : [B, S, H_x, W_x], S是聚类中心的数量
        '''
        def _inner_forward(x, delta_onehot_x: torch.Tensor):
            hw_shape = delta_onehot_x.shape[-2:]
            if self.attn_type == 'clusterattn':
                identity = x
                x, delta_onehot_x_dst = self.attn(self.norm1(x), delta_onehot_x)   # [B, L, dims], [B, S, H_x, W_x]
                x = identity + x
            else:
                x = self.attn(self.norm1(x), hw_shape, identity=x)
                delta_onehot_x_dst = delta_onehot_x
            x = self.ffn(self.norm2(x), hw_shape, identity=x)
            return x, delta_onehot_x_dst

        if self.with_cp and x.requires_grad:
            x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        else:
            x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)

        return x, delta_onehot_x_dst


@MODELS.register_module()
class MixVisionTransformerMod(BaseModule):
    """The backbone of Segformer.

    This backbone is the implementation of `SegFormer: Simple and
    Efficient Design for Semantic Segmentation with
    Transformers <https://arxiv.org/abs/2105.15203>`_.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', eps=1e-6),
        pretrained=None,
        init_cfg=None,
        with_cp=False
    ):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios
        self.with_cp = with_cp
        assert num_stages == len(num_layers) == len(num_heads) == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(                       # 512 -> 128 -> 64 -> 32 -> 16
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2,
                norm_cfg=norm_cfg
            )
            layer = ModuleList([
                TransformerEncoderLayer(                    # 128^2 @ (128/8)^2 -> 64^2 @ (64/4)^2 -> 32^2 @ (32/2)^2 -> 16^2 @ (16/1)^2 -> 
                    embed_dims=embed_dims_i,
                    num_heads=num_heads[i],
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[cur + idx],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    # attn_type='origin' if i % 2 == 0 else 'clusterattn',           # 原生efficient_vit+稀疏全局注意力vit交替的结构
                    # attn_type='origin',           # 原生efficient_vit
                    attn_type='clusterattn',           # 稀疏全局注意力vit
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    sr_ratio=sr_ratios[i]) for idx in range(num_layer)
            ])
            in_channels = embed_dims_i
            # The ret[0] of build_norm_layer is norm name.
            norm = build_norm_layer(norm_cfg, embed_dims_i)[1]
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

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
        outs = []

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)               # PatchEmbed, x: (B, out_h * out_w, embed_dims), hw_shape: (out_h, out_w)
            if i == 0:  # 获取初始聚类索引值, [B, S, H_x, W_x], 由于分类任务的初始特征图尺寸较小, 224*224的图片第一次emded之后的尺寸为56*56，为了保证聚类中心接近16*16，stride取(4, 4)
                delta_onehot_x = self.initIndex(hw_shape, stride=(16, 16), device=x.device).repeat(x.shape[0], 1, 1, 1)
            else:       # 聚类集合随着特征图的下采样而下采样
                if self.strides[i] > 1:
                    delta_onehot_x = F.interpolate(delta_onehot_x, hw_shape, mode='nearest')

            for block in layer[1]:
                x, delta_onehot_x = block(x, delta_onehot_x)            # Transformer
            x = layer[2](x)                         # Norm
            x = nlc_to_nchw(x, hw_shape)            # (B, L, C) -> (B, C, H, W)
            if i in self.out_indices:
                outs.append(x)                      # 这里模仿swin的输出形状
        return outs
