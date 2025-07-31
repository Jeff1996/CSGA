# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.registry import MODELS
from ..utils import PatchEmbed, resize

# # Flash-Attneion 1.x
# from flash_attn.flash_attn_interface import flash_attn_unpadded_func

# # Flash-Attneion 2.x
# from flash_attn import flash_attn_func

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
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

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
        # print('------ shape of q: {}, shape of c_k: {}, shape of c_v: {} ------'.format(q.shape, c_k.shape, c_v.shape))

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

# 使用flash-attention替换原始的self-attention
class FlashAttn(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        input_dims=None,
        attn_drop_rate=0.,
        proj_drop=0.,
        dropout_layer=dict(type='Dropout', drop_prob=0.),
        qkv_bias=True,
        qk_scale=None,
        proj_bias=True,
        v_shortcut=False,
    ) -> None:
        super(FlashAttn, self).__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads
        self.v_shortcut = v_shortcut

        # 固定缩放系数
        self.scale_base = qk_scale or self.head_dims**-0.5
        # # 可学习的缩放系数
        # self.scale_base = qk_scale or self.head_dims**-0.5
        # self.scale = nn.Parameter(torch.tensor(1.0))

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop_rate = attn_drop_rate
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # 慢速注意力需要这两项
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)

        self.out_drop = build_dropout(dropout_layer)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]        # [B, num_heads, N, dims]

        # 使用固定缩放系数
        q = q * self.scale_base

        # # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        # q = F.normalize(q, dim=-1)
        # k = F.normalize(k, dim=-1)
        # q = q * self.scale * self.scale_base                            # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)
        # print('------ shape of q: {}, shape of k: {}, shape of v: {} ------'.format(q.shape, k.shape, v.shape))

        # 慢速
        attn = (q @ k.transpose(-2, -1))                                  # [batch_size, num_heads, L, L']
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.embed_dims)

        # # 使用Flash-Attention 1.x的API
        # q = q.transpose(1, 2).reshape(B*N, self.num_heads, self.head_dims).half()   # [B*N, num_heads, head_dims]
        # k = k.transpose(1, 2).reshape(B*N, self.num_heads, self.head_dims).half()
        # v = v.transpose(1, 2).reshape(B*N, self.num_heads, self.head_dims).half()
        # cu_seqlens_q = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=q.device)
        # cu_seqlens_kv = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=q.device)
        # x = flash_attn_unpadded_func(
        #     q, k, v, 
        #     cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_kv, 
        #     max_seqlen_q=N, max_seqlen_k=N, 
        #     dropout_p=self.attn_drop if self.training else 0.0, 
        #     softmax_scale=1.0, 
        # ).reshape(B, N, self.embed_dims).float()

        # # 使用Flash-Attention 2.x的API
        # q = q.transpose(1, 2).half()      # [batch_size, L, num_heads, head_embed_dims]
        # k = k.transpose(1, 2).half()      # [batch_size, L, num_heads, head_embed_dims]
        # v = v.transpose(1, 2).half()      # [batch_size, L, num_heads, head_embed_dims]
        # x = flash_attn_func(q, k, v, dropout_p=self.attn_drop_rate if self.training else 0.0, softmax_scale=1.0)    # [batch_size, L, num_heads, head_embed_dims]
        # x = x.reshape(B, N, self.embed_dims).float()

        x = self.proj(x)
        x = self.out_drop(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x

class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
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
        num_fcs=2,
        qkv_bias=True,
        qk_scale=None,
        attn_type='origin',
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        batch_first=True,
        attn_cfg=dict(),
        ffn_cfg=dict(),
        with_cp=False
    ):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        # attn_cfg.update(
        #     dict(
        #         embed_dims=embed_dims,
        #         num_heads=num_heads,
        #         attn_drop=attn_drop_rate,
        #         proj_drop=drop_rate,
        #         batch_first=batch_first,
        #         bias=qkv_bias))

        # self.build_attn(attn_cfg)

        self.attn_type = attn_type
        if attn_type == 'origin':
            self.attn = MultiheadAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                batch_first=batch_first,
                bias=qkv_bias
            )
        elif attn_type == 'flashattn':
            self.attn = FlashAttn(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop_rate=attn_drop_rate,
                proj_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                qkv_bias=qkv_bias,
                # qk_scale=qk_scale,
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

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    # def build_attn(self, attn_cfg):
    #     self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x, delta_onehot_x=None):
        '''
        x               : [B, L, dims], L是包含cls_token的序列长度
        delta_onehot_x  : [B, S, H_x, W_x], S是聚类中心的数量
        '''
        def _inner_forward(x, delta_onehot_x: torch.Tensor):
            identity = x
            x = self.norm1(x)
            if self.attn_type == 'origin':
                x = self.attn(x, identity=identity)
                delta_onehot_x_dst = delta_onehot_x
            elif self.attn_type == 'flashattn':
                x = identity + self.attn(x)
                delta_onehot_x_dst = delta_onehot_x
            else:                                                       # attn_type == 'clusterattn'
                x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)    # [B, L, dims], [B, S, H_x, W_x]
                x = identity + x

            x = self.ffn(self.norm2(x), identity=x)
            return x, delta_onehot_x_dst

        if self.with_cp and x.requires_grad:
            x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        else:
            x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)

        return x, delta_onehot_x_dst


@MODELS.register_module()
class VisionTransformerMod(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        patch_pad  (str | int | None): The padding method in patch embedding.
            Default: 'corner'.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_origin (bool): Whether to output the original input embedding.
            Default: False
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_bias (dict): Whether use bias in convolution of PatchEmbed Block.
            Default: True.
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        pre_norm (bool): Whether to add a norm before Transformer Layers.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        frozen_exclude (List): List of parameters that are not to be frozen.
            Default: ["all"], "all" means there are no frozen parameters.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        patch_pad='corner',
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_origin=False,
        out_indices=-1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN'),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        patch_bias=True,
        pre_norm=False,
        final_norm=False,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False,
        with_cp=False,
        frozen_exclude=['all'],
        pretrained=None,
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained
        self.out_origin = out_origin
        self.frozen_exclude = frozen_exclude

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_pad,
            bias=patch_bias,
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        self.pre_norm = pre_norm

        if self.pre_norm:
            self.pre_ln_name, pre_ln = build_norm_layer(
                norm_cfg, embed_dims, postfix='_pre')
            self.add_module(self.pre_ln_name, pre_ln)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_type='flashattn' if i % 2 == 0 else 'clusterattn',           # 原生vit+稀疏全局注意力vit交替的结构
                    # attn_type='flashattn' if i % 2 == 0 else 'flashattn',
                    # attn_type='clusterattn',
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True
                )
            )

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1
            )
            self.add_module(self.norm1_name, norm1)

        self._freeze()

    @property
    def pre_ln(self):
        return getattr(self, self.pre_ln_name)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if isinstance(self.init_cfg, dict) and \
                self.init_cfg.get('type') in ['Pretrained', 'Pretrained_Part']:
            checkpoint = CheckpointLoader.load_checkpoint(
                self.init_cfg['checkpoint'], logger=None, map_location='cpu')

            # --- 这一段是新增+修改的，用于解决权重字典嵌套和前缀的问题 ---
            if 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            elif 'model' in checkpoint:
                checkpoint = checkpoint['model']
            else:
                pass

            if self.init_cfg.get('type') == 'Pretrained':
                state_dict = OrderedDict()
                for k, v in checkpoint.items():
                    if k.startswith('backbone.'):
                        state_dict[k[9:]] = v
                    else:
                        state_dict[k] = v
            # ---------------------------------------------------------

            elif self.init_cfg.get('type') == 'Pretrained_Part':
                state_dict = checkpoint.copy()
                para_prefix = 'image_encoder'
                prefix_len = len(para_prefix) + 1
                for k, v in checkpoint.items():
                    state_dict.pop(k)
                    if para_prefix in k:
                        state_dict[k[prefix_len:]] = v

            if 'pos_embed' in state_dict.keys():
                if self.pos_embed.shape != state_dict['pos_embed'].shape:
                    print_log(msg=f'Resize the pos_embed shape from '
                              f'{state_dict["pos_embed"].shape} to '
                              f'{self.pos_embed.shape}')
                    h, w = self.img_size
                    pos_size = int(
                        math.sqrt(state_dict['pos_embed'].shape[1] - 1))
                    state_dict['pos_embed'] = self.resize_pos_embed(
                        state_dict['pos_embed'],
                        (h // self.patch_size, w // self.patch_size),
                        (pos_size, pos_size), self.interpolate_mode)

            load_state_dict(self, state_dict, strict=False, logger=None)
        elif self.init_cfg is not None:
            super().init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    def _freeze(self):
        if 'all' in self.frozen_exclude:
            return
        for name, param in self.named_parameters():
            if not any([exclude in name for exclude in self.frozen_exclude]):
                param.requires_grad = False

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

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

    def forward(self, inputs):
        B = inputs.shape[0]

        # x: [B, L, dims]; hw_shape: [512/16=32, 512/16=32]
        x, hw_shape = self.patch_embed(inputs)

        # 获取初始聚类索引值, [B, S, H_x, W_x], 特征图尺寸为32*32，为了保证聚类数与分类任务接近，stride设置为2
        delta_onehot_x = self.initIndex(hw_shape, stride=(2, 2), device=x.device).repeat(x.shape[0], 1, 1, 1)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

        if self.pre_norm:
            x = self.pre_ln(x)

        outs = []
        if self.out_origin:             # 默认为False
            if self.with_cls_token:
                # Remove class token and reshape token for decoder head
                out = x[:, 1:]
            else:
                out = x
            B, _, C = out.shape
            out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
            if self.output_cls_token:
                out = [out, x[:, 0]]
            outs.append(out)

        for i, layer in enumerate(self.layers):
            x, delta_onehot_x = layer(x, delta_onehot_x)        # 
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
