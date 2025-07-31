'''
使用EMA构造固定码表, 量化过程采用余弦相似度(如果k矩阵已经单位化, 可直接采用点积相似度).
保证attention方法的一致性(qkT 与 kcT的一致性)
'''
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
from ..utils.vqt_tools import ClusterTool, getMask, reduce_sum

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
    def forward(ctx, x, c):
        # 前向传播（基于余弦相似度/单位化点积）
        cosSim = torch.einsum('bhld,hsd->bhls', x, c)                       # 相似度矩阵, [B, heads, L, S]
        delta_index = cosSim.argmax(dim=-1)                                 # 索引矩阵, [B, heads, L]
        delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()          # one-hot索引矩阵, [B, heads, L, S]

        # # 前向传播（基于欧式距离）
        # x2 = torch.sum(torch.square(x), dim=-1).unsqueeze(-1)               # [B, heads, L, d] -> [B, heads, L] -> [B, heads, L, 1]
        # xc = torch.einsum('bhld,hsd->bhls', x, c)                           # [B, heads, L, S]
        # c2 = torch.sum(torch.square(c), dim=-1).unsqueeze(1).unsqueeze(0)   # [heads, S, d] -> [heads, S] -> [heads, 1, S] -> [1, heads, 1, S]
        # distance2 = x2 - 2*xc + c2                                          # 待量化序列中每一个向量与码表中每一个向量的欧氏距离的平方, [B, L, S]
        # delta_index = distance2.argmin(dim=-1)                              # 索引矩阵, [B, heads, L]
        # delta_onehot = F.one_hot(delta_index, c.shape[-2]).float()           # one-hot索引矩阵, [B, heads, L, S]

        # 保存必要的中间变量用于梯度反向传播
        ctx.save_for_backward(delta_onehot)

        return delta_onehot, c                                              # forward的输出个数与backward的输入个数相同
  
    @staticmethod  
    def backward(ctx, grad_output_delta_onehot, grad_output_c):  
        # 获取中间变量
        (delta_onehot, ) = ctx.saved_tensors
        # 梯度反传
        grad_x = torch.einsum('bhls,hsd->bhld', delta_onehot, grad_output_c)# 来自码表c的梯度
        return grad_x, None                                                 # backward的输出个数与forward的输入个数相同，如果某个输入变量不需要梯度，则对应返回None

# 向量量化器
class Quantizer(nn.Module):
    '''
    序列量化器
    1. 保存一个可训练的量化码表C
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
        c_init = torch.randn(heads, codes, dim)
        c_init_norm = torch.norm(c_init, dim=-1, keepdim=True)
        c_init = c_init / c_init_norm                                           # 单位化的码表

        if ema:
            # 使用EMA更新的码表
            self.c = nn.Parameter(c_init, requires_grad=False) # 汇总的码表
            # # 也可以考虑使用register_buffer方法定义上述两个不需要梯度更新的参数
            # self.register_buffer('c', c_init)
            c_sum_new = torch.zeros(heads, codes, dim)                          # 用于累计更新量
            self.c_sum_new = nn.Parameter(c_sum_new, requires_grad=False)
            c_count_new = torch.zeros(heads, codes)                             # 用于归一化更新量
            self.c_count_new = nn.Parameter(c_count_new, requires_grad=False)

            self.update_count = 1                                               # 更新量累计次数统计
            self.update_interval = 5                                            # 码表更新间隔(码表两次更新间的更新量累计次数)
        else:
            # 使用梯度更新的码表
            self.c = nn.Parameter(c_init)

        self.gamma = 0.9                                                        # EMA超参数（历史成分占比）
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

    def getCodebook(self):
        '''
        返回归一化的码表
        c: [heads, codes, dim]
        '''
        c = self.c.data                                                     # 获取码表
        return c

    def emaAccumulate(self, delta_onehot, x):
        '''
        累计码表更新量
        更新量累计在量化之后，码表更新在量化之前，由此才不会影响梯度反向传播
        '''
        c_sum_new = torch.einsum('bhls,bhld->hsd', delta_onehot, x)         # [heads, S, dim]
        c_count_new = torch.einsum('bhls->hs', delta_onehot)                # [heads, S]
        reduce_sum(c_sum_new)                                               # 多卡同步
        reduce_sum(c_count_new)                                             # 多卡同步
        self.c_sum_new.data += c_sum_new.detach()                           # 累计
        self.c_count_new.data += c_count_new.detach()                       # 累计

    def updateCodebook(self):
        '''
        EMA更新码表

        delta_onehot: tensor, [batch size, heads, L, S], 量化K的one-hot索引矩阵Δ
        x           : tensor, [batch size, heads, L, dim], K矩阵
        '''
        # 计算更新量（单位化）
        c_sum_new = self.c_sum_new.data                                     # [heads, S, dim]
        c_count_new = self.c_count_new.data.unsqueeze(-1)                   # [heads, S, 1]
        c_count_new[c_count_new==0] = 1                                     # 防止除以0
        c_new = c_sum_new / c_count_new                                     # 计算平均更新量
        c_new_norm = torch.norm(c_new, dim=-1, keepdim=True)                # 计算码表更新量中每个向量的二范数，用于单位化
        c_new_norm[c_new_norm==0] = 1                                       # 防止除以0
        c_new = c_new / c_new_norm                                          # 单位化更新量
        c = self.gamma * self.c + (1 - self.gamma) * c_new                  # EMA更新码表
        c_norm = torch.norm(c, dim=-1, keepdim=True)                        # 计算更新后码表中每个向量的二范数，用于单位化
        c_norm[c_norm==0] = 1                                               # 防止除以0
        c = c / c_norm                                                      # 单位化码表

        # # 计算更新量（非单位化，配置基于欧氏距离的向量量化方案）
        # c_sum_new = self.c_sum_new.data                                     # [heads, S, dim]
        # index_notzero = self.c_count_new.data > 0                           # 用于决定更新哪些码表向量
        # c_count_new = self.c_count_new.data.unsqueeze(-1)                   # [heads, S, 1]
        # c_count_new[c_count_new==0] = 1                                     # 防止除以0
        # c_new = c_sum_new / c_count_new                                     # 计算平均更新量
        # c = self.c.data
        # c[index_notzero] = self.gamma * c[index_notzero] + (1 - self.gamma) * c_new[index_notzero]  # EMA更新码表

        # 更新码表
        self.c.data = c                                                     # 更新码表
        self.c_sum_new.data.fill_(.0)                                       # 累积量清零
        self.c_count_new.data.fill_(.0)                                     # 累积量清零

    # # 保存日志信息
    # def saveLog(self, path_log):
    #     with open(path_log, 'a', encoding='utf-8') as f:
    #         line_time = time.strftime("%Y-%m-%d %H:%M:%S")
    #         if self.ema:
    #             line_var = ', id_block:{}, id_layer:{}, id_GAU:{}, c_sum_max:{:9.3f}, c_sum_min:{:9.3f}, c_count_max:{:9.3f}, c_count_min:{:9.3f}\n'.format(
    #                 self.id_block, 
    #                 self.id_layer, 
    #                 self.id_GAU, 
    #                 self.c_sum.max().detach().cpu(), 
    #                 self.c_sum.min().detach().cpu(), 
    #                 self.c_count.max().detach().cpu(), 
    #                 self.c_count.min().detach().cpu()
    #             )
    #         else:
    #             line_var = ', id_block:{}, id_layer:{}, id_GAU:{}, c_max:{:9.3f}, c_min:{:9.3f}\n'.format(
    #                 self.id_block, 
    #                 self.id_layer, 
    #                 self.id_GAU, 
    #                 self.c.max().detach().cpu(), 
    #                 self.c.min().detach().cpu()
    #             )
    #         f.writelines(line_time + line_var)

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

        # 累计码表更新量(量化操作之后进行)
        if self.ema and self.training:
            self.update_count += 1
            self.emaAccumulate(delta_onehot, x)

        return delta_onehot, c

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
        self.window_size = window_size              # Ws
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads   # 每个头的特征维度
        self.scale = qk_scale or head_embed_dims**-0.5
        self.temperature = 1                # 余弦相似度降低了注意力的差异，因此添加温度系数增大差异

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
        self.dim_quantization = embed_dims // num_heads                 # 量化聚类的特征维数
        self.dim_qk = num_heads * self.dim_quantization
        self.dim_v = embed_dims
        # self.proj_z = nn.Sequential(
        #     nn.Linear(self.dim_base, self.dim_z, bias = False),         # 通过一个线性映射构造Z矩阵
        #     nn.SiLU(),
        #     nn.Linear(self.dim_z, self.dim_qk, bias = False),           # 通过一个线性映射构造Z矩阵
        # )
        # self.proj_q = nn.Sequential(
        #     ScaleOffset(self.dim_qk), 
        # )
        # self.proj_k = nn.Sequential(
        #     ScaleOffset(self.dim_qk), 
        # )
        # self.proj_v = nn.Sequential(
        #     nn.Linear(self.dim_base, self.dim_v, bias = False),         # 通过一个线性映射构造V矩阵
        #     nn.SiLU(),
        # )
        self.proj_qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        # # k矩阵量化: 分头量化, [B*num_heads, num_windows, Wh*Ww, C//num_heads]
        # # 获取量化矩阵k_hat, 码表C
        # self.levels = 4
        # cluster = ClusterTool(num_workers=1, method='quantizationcluster', levels=self.levels)
        # self.quantizer = Quantizer(cluster)                             # 输入序列量化器
        codes = 512
        self.quantizer = Quantizer(
            heads=num_heads, 
            codes=codes, 
            dim=self.dim_quantization
        )                                                               # 输入序列量化器

        # 计算注意力矩阵
        self.scale = self.dim_quantization ** -0.25
        
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

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 新增内容 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
        # z = self.proj_z(x)                  # [B, num_windows, N, embed_dims]

        # # [B, num_windows, N, num_heads*self.dim_quantization] -> [B, num_windows, N, num_heads, self.dim_quantization] -> 
        # # [B, num_heads, num_windows, N, self.dim_quantization]
        # q = self.proj_q(z).view(
        #     B, num_windows, N, self.num_heads, self.dim_quantization
        # ).permute(0, 3, 1, 2, 4).contiguous()
        # # q *= self.scale
        # q_norm = torch.norm(q, dim=-1, keepdim=True)
        # q_norm[q_norm==0] = 1               # 防止除以0
        # q = q / q_norm                      # 单位化，将点积相似度转换为余弦相似度

        # # 形状同q
        # k = self.proj_k(z).view(
        #     B, num_windows, N, self.num_heads, self.dim_quantization
        # ).permute(0, 3, 1, 2, 4).contiguous()
        # # k *= self.scale
        # k_norm = torch.norm(k, dim=-1, keepdim=True)
        # k_norm[k_norm==0] = 1               # 防止除以0
        # k = k / k_norm                      # 单位化，将点积相似度转换为余弦相似度

        # # [B, num_windows, N, embed_dims] -> [B, num_heads, num_windows, N, embed_dims]
        # v = self.proj_v(x).view(
        #     B, num_windows, N, self.num_heads, C // self.num_heads
        # ).permute(0, 3, 1, 2, 4).contiguous()


        # --- 原始的qkv构造 ---
        # [B, num_windows, Wh*Ww, C] -> [B, num_windows, Wh*Ww, 3*C] -> 
        # [B, num_windows, Wh*Ww, 3, num_heads, C//num_heads] -> [3, B, num_heads, num_windows, Wh*Ww, C//num_heads]
        qkv = self.proj_qkv(x).reshape(B, num_windows, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5).contiguous()
        # [B, num_heads, num_windows, N, self.dim_quantization]
        q, k, v = qkv[0], qkv[1], qkv[2]
        q_norm = torch.norm(q, dim=-1, keepdim=True)
        q_norm[q_norm==0] = 1               # 防止除以0
        q = q / q_norm                      # 单位化，将点积相似度转换为余弦相似度
        k_norm = torch.norm(k, dim=-1, keepdim=True)
        k_norm[k_norm==0] = 1               # 防止除以0
        k = k / k_norm                      # 单位化，将点积相似度转换为余弦相似度

        # # 窗口注意力, [B, num_heads, num_windows, N, N]
        # qkT = torch.einsum('bhrck,bhrgk->bhrcg', q, k) * self.temperature

        # # 局部注意力增强
        # if self.shift_size == 0:
        #     # [B, num_heads, num_windows, N, N] + [1, 1, 1, N, N]
        #     qkTm = qkT + self.mask.unsqueeze(0)
        # else:
        #     # [B, num_heads, num_windows, N, N] + [N, N]，总体先加一个主要的局部增强，后面再对特殊位置进行修正
        #     qkTm = qkT + self.mask_main
        #     # [B, num_heads, 1, N, N] + [N, N]
        #     qkTm[:, :, 0] += (self.mask0-self.mask_main)
        #     # [B, num_heads, nWw-1, N, N] + [N, N]
        #     qkTm[:, :, 1:nWw] += (self.mask1-self.mask_main)
        #     # [B, num_heads, nWh-1, N, N] + [N, N]
        #     qkTm[:, :, nWw::nWw] += (self.mask2-self.mask_main)

        # 量化k矩阵
        k_sequence = k.view(B, self.num_heads, num_windows*N, self.dim_quantization)
        # [B, self.num_heads, num_windows*N, S], [self.num_heads, S, self.dim_quantization]
        delta_onehot, c = self.quantizer(k_sequence)
        # [B, self.num_heads, num_windows*N, self.dim_quantization]
        k_hat_sequence = torch.einsum('bhls,hsk->bhlk', delta_onehot, c)
        k_hat = k_hat_sequence.view(B, self.num_heads, num_windows, N, self.dim_quantization)
        delta_onehot = delta_onehot.view(B, self.num_heads, num_windows, N, -1)

        # 量化后的窗口注意力
        # [B, num_heads, num_windows, N, N]
        qk_hatT = torch.einsum('bhrck,bhrgk->bhrcg', q, k_hat) * self.temperature

        # ↓↓↓↓↓↓ 临时增加的，用以测试量化对于局部注意力造成的影响 ↓↓↓↓↓↓
        # 局部注意力增强
        if self.shift_size == 0:
            # [B, num_heads, num_windows, N, N] + [1, 1, 1, N, N]
            qkTm = qk_hatT + self.mask.unsqueeze(0)
        else:
            # [B, num_heads, num_windows, N, N] + [N, N]，总体先加一个主要的局部增强，后面再对特殊位置进行修正
            qkTm = qk_hatT + self.mask_main
            # [B, num_heads, 1, N, N] + [N, N]
            qkTm[:, :, 0] += (self.mask0-self.mask_main)
            # [B, num_heads, nWw-1, N, N] + [N, N]
            qkTm[:, :, 1:nWw] += (self.mask1-self.mask_main)
            # [B, num_heads, nWh-1, N, N] + [N, N]
            qkTm[:, :, nWw::nWw] += (self.mask2-self.mask_main)
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

        # # [B, num_heads, num_windows, N, S]
        # qcT = torch.einsum('bhrck,hsk->bhrcs', q, c) * self.temperature
        
        # # 防溢出处理
        # qkT_max = qkT.max(dim=-1, keepdim=True)[0]                                      # [B, num_heads, num_windows, N, 1]
        # qcT_max = qcT.max(dim=-1, keepdim=True)[0]                                      # [B, num_heads, num_windows, N, 1]
        # qxT_max = torch.cat((qkT_max, qcT_max), dim=-1).max(dim=-1, keepdim=True)[0]    # [B, num_heads, num_windows, N, 1]
        # qkT -= qxT_max
        # qk_hatT -= qxT_max
        # qcT -= qxT_max

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

        # - swin窗口注意力 -
        attn = self.softmax(qkTm)                                                   # [B, num_heads, num_windows, N, N]
        attn = self.attn_drop(attn)
        x = torch.einsum('bhrcg,bhrgv->bhrcv', attn, v)                             # [B, num_heads, num_windows, N, C // self.num_heads]

        # [B, num_heads, num_windows, N, C // self.num_heads] -> [B, num_windows, N, num_heads, C // self.num_heads] -> 
        # [B, num_windows, N, C]
        x = x.permute(0, 2, 3, 1, 4).contiguous().view(
            B, num_windows, N, C
        )

        # 量化损失（二范数损失，也可以考虑余弦相似度损失），考虑到不同self-attention层的序列长度、头数不一致，因此需要求均值，最后需要对每一层的量化损失求和
        error_quantization = torch.norm(k - k_hat.detach(), dim=-1).square().mean()
        # error_quantization = torch.square(k - k_hat.detach()).mean()
        # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

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
        attn_windows, error_quantization = self.w_msa(query_windows, H_pad//self.window_size, W_pad//self.window_size)

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
