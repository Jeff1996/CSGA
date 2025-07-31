'''
架构说明: swin_debug_old_0104.py
0. 使用迭代的聚类中心
1. 使用动态量化码表，使用掩膜进行非极大值抑制，实现标准的k-means聚类(仅迭代1次)，构造聚类中心
2. 
3. 局部分支(WindowMSA)与量化全局分支(ClusterAttn)采用交替级联结构
4. qk_scale = 15，主分支和聚类过程各引入了一个初始化为1的可学习缩放系数，qk使用余弦相似度（单位化点积），k矩阵下采样的得到的聚类中心需要重新单位化
5. 全局分支引入相对位置编码

训练结果: 
    /home/hjf/workspace/mmpretrain/work_dirs/swin-tiny_debug_4xb64_in1k/20250105_174012/20250105_174012.log
训练配置: 
    50 epochs, 
    batchsize 4 * 43 * 6 = 1032, 
    stride(4*4), 
    swin-t预训练权重
top1: 80.626
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

# # Flash-Attneion 1.x
# from flash_attn.flash_attn_interface import flash_attn_unpadded_func

# # Flash-Attneion 2.x
# from flash_attn import flash_attn_func

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

        x = self.proj(x)
        x = self.proj_drop(x)

        # # ------------------------------------------------------------ debug ------------------------------------------------------------
        # print('------------------------- id_stage: {:02d}, id_block: {:02d} -------------------------'.format(self.id_stage, self.id_block))
        # print('block_scale_learned: {:.6f}, block_scale_total: {:.6f}'.format(self.scale.data, self.scale * self.scale_base))
        # print('\n')
        # # ------------------------------------------------------------------------------------------------------------------------

        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

# 余弦相似度
def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

# 聚类稀疏全局注意力
class Clustering(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        center_w=10, 
        center_h=10, 
        qkv_bias=True, 
        return_center=False, 
        num_clustering=3
    ):
        super().__init__()
        assert dim % heads == 0, 'dim != heads*head_dim'
        self.heads = int(heads)
        self.head_dim = dim // self.heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)                 # 输出

        # 余弦相似度仿射变换
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        # 聚类中心构造
        self.centers_proposal = nn.AdaptiveAvgPool2d((center_w, center_h))

        self.return_center = return_center
        self.softmax = nn.Softmax(dim=-2)
        self.num_clustering = num_clustering

    def forward(self, x: torch.Tensor): 
        '''
        x: tensor, [B, H, W, C]
        '''
        b, h, w, c = x.shape
        # [B, 3*C, H, W] -> [b, 3, heads, head_dim, h, w] -> [3, b, heads, head_dim, h, w]
        x, value, feature = self.qkv(x).reshape(b, h, w, 3, self.heads, self.head_dim).permute(3, 0, 4, 5, 1, 2)

        x = x.reshape(-1, self.head_dim, h, w)                          # [b*heads, head_dim, h, w]
        value = value.reshape(-1, self.head_dim, h, w)                  # [b*heads, head_dim, h, w]
        feature = feature.reshape(-1, self.head_dim, h, w)              # [b*heads, head_dim, h, w]

        # 构造聚类中心
        centers = self.centers_proposal(x)                              # 初始化聚类中心，[-1, head_dim, h_center, w_center]
        _, head_dim, h_center, w_center = centers.shape
        centers_feature = self.centers_proposal(feature).permute(0, 2, 3, 1).reshape(-1, h_center*w_center, head_dim) # 聚类中心skip connection
        
        # processing before cluster
        centers = centers.permute(0, 2, 3, 1).reshape(-1, h_center*w_center, head_dim)
        value = value.permute(0, 2, 3, 1).reshape(-1, h*w, head_dim)
        feature = feature.permute(0, 2, 3, 1).reshape(-1, h*w, head_dim)
        
        # 迭代更新聚类中心
        for _ in range(self.num_clustering):
            attn = (centers @ value.transpose(-2, -1))                  # [b*heads, L', L]
            attn = self.softmax(attn)
            centers = (attn @ feature)
        
        # 使用聚类中心重新表示特征序列
        # similarity, [b*heads, L', L]
        similarity = torch.sigmoid(self.sim_beta + self.sim_alpha * pairwise_cos_sim(centers, x.reshape(-1, head_dim, h*w).permute(0, 2, 1)))
        
        # assign each point to one center
        _, max_idx = similarity.max(dim=1, keepdim=True)
        mask = torch.zeros_like(similarity)
        mask.scatter_(1, max_idx, 1.)                                   # 沿码表方向的one-hot矩阵
        similarity= similarity * mask                                   # 屏蔽无关相似度，相较于直接使用one-hot矩阵替换similarity，解决了agmax无法进行梯度反传的问题

        # [b*heads, L', L, 1] * [b*heads, 1, L, head_dim] -> [b*heads, L', L, head_dim] -> [b*heads, L', head_dim]
        out = ((similarity.unsqueeze(dim=-1) * feature.unsqueeze(dim=1)).sum(dim=2) + centers_feature) / (mask.sum(dim=-1,keepdim=True)+ 1.0) 

        if self.return_center:
            out = out.permute(0, 2, 1).reshape(-1, c, h_center, w_center)
            return out
        else:
            # [b*heads, L', 1, head_dim] * [b*heads, L', L, 1] -> [b*heads, L', L, head_dim] -> [b*heads, L, head_dim]
            out = (out.unsqueeze(dim=2) * similarity.unsqueeze(dim=-1)).sum(dim=1)
            # [b*heads, L, head_dim] -> [b, dim, h, w]
            # out = out.permute(0, 2, 1).reshape(-1, c, h, w)
            out = out.reshape(b, self.heads, h, w, self.head_dim).permute(0, 2, 3, 1, 4).reshape(b, h, w, c)
            out = self.proj(out)
            return out

# 在这里切换注意力类型
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
            self.w_msa = Clustering(
                dim=embed_dims,
                heads=num_heads,
                qkv_bias=qkv_bias,
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
            attn_windows = self.w_msa(query_windows, batch_size=B, mask=attn_mask)

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
            x = self.w_msa(query)

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x

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

# layer(在这里修改checkpoints范围)
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

        def _inner_forward(x, hw_shape):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)
            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, hw_shape)
        else:
            x = _inner_forward(x, hw_shape)

        return x

# stage
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
                qk_scale=None if i % 2 == 0 else qk_scale,
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
        x: [B, L, C]
        hw_shape: (H, W)
        '''
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape

@MODELS.register_module()
class SwinTransformerCluster(BaseModule):
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

    def forward(self, x, data_samples = None):

        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []

        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(
                    -1, *out_hw_shape,
                    self.num_features[i]
                ).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return tuple(outs)