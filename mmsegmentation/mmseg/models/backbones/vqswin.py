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
from ..utils.vqt_tools import ClusterTool, getMask

class ScaleOffset(nn.Module):
    def __init__(self, features):
        super(ScaleOffset, self).__init__()
        # self.gamma = nn.Parameter(torch.ones(1, 1, features))
        self.gamma = nn.Parameter(torch.randn(1, 1, features)+1)  # 均值为1，方差为1的正态分布初始化
        self.beta = nn.Parameter(torch.zeros(1, 1, features))
    def forward(self, x):
        return x * self.gamma + self.beta

# 自定义梯度传播过程
class vecQuantization(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, cluster: ClusterTool):
        '''
        x: tensor, [B, L, dims]
        c: tensor, [B, 2**dims, dims]
        '''
        # 前向传播
        device = x.device
        deltas, codebooks = cluster.Apply(x.detach().cpu())
        deltas = deltas.to(device)
        codebooks = codebooks.to(device)
        deltas_onehot = F.one_hot(deltas, codebooks.shape[-2]).float()      # one-hot索引矩阵, [B, L, S]
        # 保存必要的中间变量用于梯度反向传播
        # ctx.save_for_backward(deltas_onehot, codebooks)
        ctx.save_for_backward(deltas_onehot)
        return deltas_onehot, codebooks                                     # forward的输出个数与backward的输入个数相同

    @staticmethod
    def backward(ctx, grad_output_deltas_onehot: torch.Tensor, grad_output_codebooks: torch.Tensor):
        # 获取中间变量
        # (deltas_onehot, codebooks) = ctx.saved_tensors
        (deltas_onehot, ) = ctx.saved_tensors

        # # 梯度反传
        # grad_x_deltas_onehot = torch.einsum('bls,bsd->bld', grad_output_deltas_onehot, codebooks)   # 来自deltas_onehot的梯度
        # grad_x_codebooks = torch.einsum('bls,bsd->bld', deltas_onehot, grad_output_codebooks)       # 来自码表codebooks的梯度
        # grad_x = grad_x_deltas_onehot + grad_x_codebooks
        grad_x = torch.einsum('bls,bsd->bld', deltas_onehot, grad_output_codebooks)       # 来自码表codebooks的梯度

        # # 打印中间值
        # if int(os.environ['RANK']) == 0 and Flag_debug:
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('grad_output_deltas_onehot', grad_output_deltas_onehot.max(), grad_output_deltas_onehot.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}\n'.format('grad_output_codebooks', grad_output_codebooks.max(), grad_output_codebooks.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}\n'.format('grad_x', grad_x.max(), grad_x.min()))

        return grad_x, None                                                 # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None

# 向量量化器
class Quantizer(nn.Module):
    '''
    序列量化器
    1. 保存一个可训练的量化码表C
    2. 构造量化的K序列K^
    3. 获取输入矩阵K量化后的索引矩阵Δ
    '''
    def __init__(self, cluster: ClusterTool):
        '''
        cluster: 聚类方法
        '''
        super(Quantizer, self).__init__()
        self.cluster = cluster
        self.vecQuantization = vecQuantization()

    def forward(self, x):
        '''
        输入
        x               : tensor, [batch size, L, dim], K矩阵

        输出
        delta_onehot    : tensor, [batch size, L , S], 量化K的索引矩阵Δ
        c               : tensor, [C, dim], 量化码表
        '''
        # 量化
        deltas_onehot, codebooks = self.vecQuantization.apply(x, self.cluster)
        return deltas_onehot, codebooks

# 加入全局注意力的attention
class WindowMSA(BaseModule):
    """
    Window based multi-head self-attention (W-MSA) module with relative position bias.

    Args:
        embed_dims (int)                    : Number of input channels.
        num_heads (int)                     : Number of attention heads.
        window_size (tuple[int])            : The height and width of the window.
        qkv_bias (bool, optional)           :  If True, add a learnable bias to q, k, v. Default: True.
        qk_scale (float | None, optional)   : Override default qk scale of head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional)    : Dropout ratio of attention weight. Default: 0.0
        proj_drop_rate (float, optional)    : Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional)    : The Config for initialization. Default: None.
    """

    def __init__(
        self,
        embed_dims,                         # 嵌入特征维度
        num_heads,                          # 头数
        window_size,                        # 分块大小
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.,                  # 
        proj_drop_rate=0.,                  # 
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.window_size = window_size      # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads   # 每个头的特征维度
        self.scale = qk_scale or head_embed_dims**-0.5

        # # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 旧代码 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        # )                                   # 2*Wh-1 * 2*Ww-1, nH
        # # About 2x faster than original impl
        # Wh, Ww = self.window_size
        # rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        # rel_position_index = rel_index_coords + rel_index_coords.T
        # rel_position_index = rel_position_index.flip(1).contiguous()
        # self.register_buffer('relative_position_index', rel_position_index)
        # self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop_rate)
        # # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 新增内容 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        gain = 1
        self.shift_size = shift_size
        if shift_size == 0:
            mask = getMask((window_size, window_size), gain)                    # [1, 1, window_size**2, window_size**2]
            self.register_buffer('mask', mask)                                  # 局部注意力增强掩膜（加性）
        else:
            # 左下角大片区域的mask，与unshifted时的mask一样
            mask_main = getMask((window_size, window_size), gain).squeeze()     # [window_size**2, window_size**2]
            self.register_buffer('mask_main', mask_main)

            # mask0, 左上角的一个mask，该mask包含四个不相邻的区域
            # [1, 1, shift_size**2, shift_size**2]
            mask0_topleft = getMask((shift_size, shift_size), gain)
            # [shift_size**2, shift_size**2], 不能使用squeeze方法，否则当shift_size=1时会出bug
            mask0_topleft = mask0_topleft.view(*mask0_topleft.shape[-2:])

            # [shift_size*(window_size-shift_size), shift_size*(window_size-shift_size)]
            mask0_topright = getMask((shift_size, window_size-shift_size), gain).squeeze()

            # [(window_size-shift_size)*shift_size, (window_size-shift_size)*shift_size]
            mask0_downleft = getMask((window_size-shift_size, shift_size), gain).squeeze()

            # [1, 1, (window_size-shift_size)**2, (window_size-shift_size)**2]
            mask0_downright = getMask((window_size-shift_size, window_size-shift_size), gain)
            # [(window_size-shift_size)**2, (window_size-shift_size)**2], 
            # 不能使用squeeze方法，否则当window_size-shift_size=1时会出bug
            mask0_downright = mask0_downright.view(*mask0_downright.shape[-2:])
            mask0 = torch.zeros_like(mask_main)
            # top left
            for index_h in range(shift_size):
                for index_w in range(shift_size):
                    mask0[
                        index_h:shift_size*window_size:window_size, 
                        index_w:shift_size*window_size:window_size
                    ] = mask0_topleft[
                        index_h::shift_size, 
                        index_w::shift_size
                    ]
            # top right
            for index_h in range(window_size-shift_size):
                for index_w in range(window_size-shift_size):
                    mask0[
                        index_h+shift_size:shift_size*window_size+shift_size:window_size, 
                        index_w+shift_size:shift_size*window_size+shift_size:window_size
                    ] = mask0_topright[
                        index_h::window_size-shift_size, 
                        index_w::window_size-shift_size
                    ]
            # down left
            for index_h in range(shift_size):
                for index_w in range(shift_size):
                    mask0[
                        index_h+shift_size*window_size::window_size, 
                        index_w+shift_size*window_size::window_size
                    ] = mask0_downleft[
                        index_h::shift_size, 
                        index_w::shift_size
                    ]
            # down right
            for index_h in range(window_size-shift_size):
                for index_w in range(window_size-shift_size):
                    mask0[
                        index_h+shift_size*window_size+shift_size::window_size, 
                        index_w+shift_size*window_size+shift_size::window_size
                    ] = mask0_downright[
                        index_h::window_size-shift_size, 
                        index_w::window_size-shift_size
                    ]
            self.register_buffer('mask0', mask0)                                # [stride**2, stride**2]

            # mask1, 右上方的mask，该mask包含上下两个不相邻的区域
            # [shift_size*window_size, shift_size*window_size]
            mask1_up = getMask((shift_size, window_size), gain).squeeze()
            # [(window_size-shift_size)*window_size, (window_size-shift_size)*window_size]
            mask1_down = getMask((window_size-shift_size, window_size), gain).squeeze()
            mask1 = torch.zeros_like(mask_main)
            mask1[:shift_size*window_size, :shift_size*window_size] = mask1_up
            mask1[shift_size*window_size:, shift_size*window_size:] = mask1_down
            self.register_buffer('mask1', mask1)                                # [window_size**2, window_size**2]

            # mask2, 左下方的mask，该mask包含左右两个不相邻的区域
            # [window_size*shift_size, window_size*shift_size]
            mask2_left = getMask((window_size, shift_size), gain).squeeze()
            # [window_size*(window_size-shift_size), window_size*(window_size-shift_size)]
            mask2_right = getMask((window_size, window_size-shift_size), gain).squeeze()
            mask2 = torch.zeros_like(mask_main)
            for index_h in range(shift_size):
                for index_w in range(shift_size):
                    mask2[index_h::window_size, index_w::window_size] = \
                        mask2_left[index_h::shift_size, index_w::shift_size]
            for index_h in range(window_size-shift_size):
                for index_w in range(window_size-shift_size):
                    mask2[index_h+shift_size::window_size, index_w+shift_size::window_size] = \
                        mask2_right[index_h::window_size-shift_size, index_w::window_size-shift_size]
            self.register_buffer('mask2', mask2)                                # [stride**2, stride**2]    

        # 特征压缩: qk特征维度压缩至num_heads * dims, v则保持基础特征维度. [B, num_windows, num_heads, Wh*Ww, C//num_heads]
        self.dim_base = embed_dims
        self.dim_z = embed_dims
        self.dim_quantization = 5                                       # 量化聚类的特征维数
        self.dim_qk = num_heads * self.dim_quantization
        self.dim_v = embed_dims
        self.proj_z = nn.Sequential(
            nn.Linear(self.dim_base, self.dim_z, bias = False),         # 通过一个线性映射构造Z矩阵
            nn.SiLU(),
            nn.Linear(self.dim_z, self.dim_qk, bias = False),           # 通过一个线性映射构造Z矩阵
        )
        self.proj_q = nn.Sequential(
            ScaleOffset(self.dim_qk), 
        )
        self.proj_k = nn.Sequential(
            ScaleOffset(self.dim_qk), 
        )
        self.proj_v = nn.Sequential(
            nn.Linear(self.dim_base, self.dim_v, bias = False),         # 通过一个线性映射构造V矩阵
            nn.SiLU(),
        )

        # # k矩阵量化: 分头量化, [B*num_heads, num_windows, Wh*Ww, C//num_heads]
        # # 获取量化矩阵k_hat, 码表C
        # self.levels = 4
        # cluster = ClusterTool(num_workers=1, method='quantizationcluster', levels=self.levels)
        # self.quantizer = Quantizer(cluster)                             # 输入序列量化器

        # 计算注意力矩阵
        self.scale = self.dim_quantization ** -0.25
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
        
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, nWh, nWw):
        """
        Args:
            x (tensor)                      : input features with shape of [B, num_windows, Wh*Ww, C]
            mask (tensor | None, Optional)  : mask with shape of (num_windows, Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, num_windows, N, C = x.shape

        # # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 旧代码 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # # [B, num_windows, Wh*Ww, C] -> [B, num_windows, Wh*Ww, 3*C] -> 
        # # [B, num_windows, Wh*Ww, 3, num_heads, C//num_heads] -> [3, B, num_windows, num_heads, Wh*Ww, C//num_heads]
        # qkv = self.qkv(x).reshape(B, num_windows, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        # # make torchscript happy (cannot use tensor as tuple)
        # q, k, v = qkv[0], qkv[1], qkv[2]    # [B, num_windows, num_heads, Wh*Ww, C//num_heads]

        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))    # [B, num_windows, num_heads, Wh*Ww, Wh*Ww]

        # # 相对位置编码, [Wh*Ww, Wh*Ww, num_heads]
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #         self.window_size[0] * self.window_size[1],
        #         self.window_size[0] * self.window_size[1],
        #         -1
        # )
        # # [Wh*Ww, Wh*Ww, num_heads] -> [num_heads, Wh*Ww, Wh*Ww]
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # # 添加相对位置编码, [B, num_windows, num_heads, Wh*Ww, Wh*Ww]
        # attn = attn + relative_position_bias.unsqueeze(0).unsqueeze(0)

        # # 添加掩膜, 屏蔽不相邻元素间的注意力(由cycleshift造成的不相邻像素被划分到同一个window)
        # if mask is not None:
        #     nW = mask.shape[0]
        #     # [B, num_windows, num_heads, Wh*Ww, Wh*Ww] + [1, num_windows, 1, Wh*Ww, Wh*Ww]
        #     attn = attn + mask.unsqueeze(1).unsqueeze(0)
        
        # # softmax
        # attn = self.softmax(attn)
        # attn = self.attn_drop(attn)
        # # [B, num_windows, num_heads, Wh*Ww, C//num_heads] -> [B, num_windows, Wh*Ww, num_heads, C//num_heads] -> 
        # # [B, num_windows, Wh*Ww, C]
        # x = (attn @ v).transpose(1, 2).reshape(B, num_windows, N, C)
        # # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑


        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 新增内容 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        z = self.proj_z(x)  # [B, num_windows, N, embed_dims]

        # [B, num_windows, N, num_heads*self.dim_quantization] -> [B, num_windows, N, num_heads, self.dim_quantization] -> 
        # [B, num_heads, num_windows, N, self.dim_quantization]
        q = self.proj_q(z).view(
            B, num_windows, N, self.num_heads, self.dim_quantization
        ).permute(0, 3, 1, 2, 4).contiguous()
        # q *= self.scale
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q_norm[q_norm==0] = 1               # 防止除以0
        q = q / q_norm                      # 单位化，将点积相似度转换为余弦相似度

        # 形状同q
        k = self.proj_k(z).view(
            B, num_windows, N, self.num_heads, self.dim_quantization
        ).permute(0, 3, 1, 2, 4).contiguous()
        # k *= self.scale
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        k_norm[k_norm==0] = 1               # 防止除以0
        k = k / k_norm                      # 单位化，将点积相似度转换为余弦相似度

        # [B, num_windows, N, embed_dims] -> [B, num_heads, num_windows, N, embed_dims]
        v = self.proj_v(x).view(
            B, num_windows, N, self.num_heads, C // self.num_heads
        ).permute(0, 3, 1, 2, 4).contiguous()

        # 窗口注意力
        qkT = torch.einsum('bhrck,bhrgk->bhrcg', q, k)                  # [B, num_heads, num_windows, N, N]

        # # 量化k矩阵
        # k_sequence = k.view(B*self.num_heads, num_windows*N, self.dim_quantization)
        # # [B*self.num_heads, num_windows*N, S], [B*self.num_heads, S, self.dim_quantization]
        # delta_onehot, c = self.quantizer(k_sequence)
        # # [B*self.num_heads, num_windows*N, self.dim_quantization]
        # k_hat_sequence = torch.einsum('bls,bsk->blk', delta_onehot, c)
        # k_hat = k_hat_sequence.view(B, self.num_heads, num_windows, N, self.dim_quantization)
        # c = c.view(B, self.num_heads, -1, self.dim_quantization)
        # delta_onehot = delta_onehot.view(B, self.num_heads, num_windows, N, -1)

        # # 量化后的窗口注意力
        # qk_hatT = torch.einsum('bhrck,bhrgk->bhrcg', q, k_hat)          # [B, num_heads, num_windows, N, N]
        # qcT = torch.einsum('bhrck,bhsk->bhrcs', q, c)                   # [B, num_heads, num_windows, N, S]
        
        # # 防溢出处理
        # qkT_max = qkT.max(dim=-1, keepdim=True)[0]                                      # [B, num_heads, num_windows, N, 1]
        # qcT_max = qcT.max(dim=-1, keepdim=True)[0]                                      # [B, num_heads, num_windows, N, 1]
        # qxT_max = torch.cat((qkT_max, qcT_max), dim=-1).max(dim=-1, keepdim=True)[0]    # [B, num_heads, num_windows, N, 1]
        # qkT -= qxT_max
        # qk_hatT -= qxT_max
        # qcT -= qxT_max

        # 局部注意力增强
        if self.shift_size == 0:
            # [B, num_heads, num_windows, N, N] + [1, 1, 1, N, N]
            qkTm = qkT + self.mask.unsqueeze(0)
        else:
            # [B, num_heads, num_windows, N, N] + [N, N]，总体先加一个主要的局部增强，后面再对特殊位置进行修正
            qkTm = qkT + self.mask_main
            # [B, num_heads, 1, N, N] + [N, N]
            qkTm[:, :, 0] += (self.mask0-self.mask_main)
            # [B, num_heads, nWw-1, N, N] + [N, N]
            qkTm[:, :, 1:nWw] += (self.mask1-self.mask_main)
            # [B, num_heads, nWh-1, N, N] + [N, N]
            qkTm[:, :, nWw::nWw] += (self.mask2-self.mask_main)

        # # exp
        # qkTm_exp = torch.exp(qkTm)                                                  # [B, num_heads, num_windows, N, N]
        # qk_hatT_exp = torch.exp(qk_hatT)                                            # [B, num_heads, num_windows, N, N]
        # qcT_exp = torch.exp(qcT)                                                    # [B, num_heads, num_windows, N, S]

        # # - 计算softmax分子 -
        # numerator1 = torch.einsum('bhrcg,bhrgv->bhrcv', qkTm_exp - qk_hatT_exp, v)  # [B, num_heads, num_windows, N, C // self.num_heads]
        # deltaTv = torch.einsum('bhrcs,bhrcv->bhsv', delta_onehot, v)                # [B, self.num_heads, S, C // self.num_heads]
        # numerator2 = torch.einsum('bhrcs,bhsv->bhrcv', qcT_exp, deltaTv)            # [B, num_heads, num_windows, N, C // self.num_heads]
        # numerator = numerator1 + numerator2                                         # [B, num_heads, num_windows, N, C // self.num_heads]

        # # - 计算softmax分母 -
        # denominator1 = torch.einsum('bhrcg->bhrc', qkTm_exp - qk_hatT_exp)          # [B, num_heads, num_windows, N]
        # deltaT1 = torch.einsum('bhrcs->bhs', delta_onehot)                          # [B, num_heads, S]
        # denominator2 = torch.einsum('bhrcs,bhs->bhrc', qcT_exp, deltaT1)            # [B, num_heads, num_windows, N]
        # denominator = (denominator1 + denominator2).unsqueeze(-1)                   # [B, num_heads, num_windows, N, 1]
        # denominator[denominator==0] = 1e-6                                          # 防止除以0

        # # - 计算最终结果 -
        # x = numerator / denominator                                                 # [B, num_heads, num_windows, N, C // self.num_heads]

        attn = self.softmax(qkTm)                                                   # [B, num_heads, num_windows, N, N]
        attn = self.attn_drop(attn)
        x = torch.einsum('bhrcg,bhrgv->bhrcv', attn, v)                             # [B, num_heads, num_windows, N, C // self.num_heads]

        # [B, num_heads, num_windows, N, C // self.num_heads] -> [B, num_windows, N, num_heads, C // self.num_heads] -> 
        # [B, num_windows, N, C]
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(
            B, num_windows, N, C
        )
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

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
        init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size, 
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

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
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )

        #     # calculate attention mask for SW-MSA
        #     img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size,
        #                       -self.shift_size), slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size,
        #                       -self.shift_size), slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1

        #     # 1, nW, window_size, window_size, 1
        #     mask_windows = self.window_partition(img_mask)

        #     # nW, window_size*window_size
        #     mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        #     # [nW, 1, window_size*window_size] - [nW, window_size*window_size, 1] -> 
        #     # [nW, window_size*window_size, window_size*window_size]
        #     attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        #     # [nW, window_size*window_size, window_size*window_size]
        #     attn_mask = attn_mask.masked_fill(
        #         attn_mask != 0, float(-100.0)
        #     ).masked_fill(
        #         attn_mask == 0, float(0.0)
        #     )
        else:
            shifted_query = query
            attn_mask = None

        # B, nW, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # B, nW, window_size*window_size, C
        query_windows = query_windows.view(B, -1, self.window_size**2, C)

        # W-MSA/SW-MSA [B, nW, window_size*window_size, C]
        attn_windows = self.w_msa(query_windows, H_pad//self.window_size, W_pad//self.window_size)

        # merge windows, [B*nW, window_size, window_size, C]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        # 如果之前为了使得特征图尺寸能够整除window_size添加了padding, 这里需要去掉padding
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

        # [B, W, W, C] -> [B, H // window_size, window_size, W // window_size, window_size, C]
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

        # [B, H // window_size, window_size, W // window_size, window_size, C] -> 
        # [B, H // window_size, W // window_size, window_size, window_size, C]
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        # [B, H // window_size, W // window_size, window_size, window_size, C] -> 
        # 这是原本的代码, [B*num_windows, , window_size, window_size, C]
        # windows = windows.view(-1, window_size, window_size, C)

        # 这是我修改后的, [B, num_windows, window_size, window_size, C]
        windows = windows.view(B, -1, window_size, window_size, C)

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

    def __init__(self,
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
                 init_cfg=None):

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
            init_cfg=None)

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
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


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

    def __init__(self,
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
                 init_cfg=None):
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
                init_cfg=None)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class VQSwinTransformer(BaseModule):
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
                init_cfg=None)
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
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs
