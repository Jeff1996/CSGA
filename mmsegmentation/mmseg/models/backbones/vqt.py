'''
基于量化聚类的线性Transformer
'''
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.autograd import Function
from torch.nn import init

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_, trunc_normal_init)
from mmengine.runner import CheckpointLoader
from mmengine.utils import to_2tuple

from mmseg.registry import MODELS
from ..utils.embed import PatchEmbed, PatchMerging
from ..utils.vqt_tools import ClusterTool, LogOut, Flag_debug, getBlocks, getFeaturemap, getMask 

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
 
    def _norm(self,x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
 
    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerNorm(nn.Module):
    "Construct a layernorm module (similar to torch.nn.LayerNorm)."
    '''
    在视觉领域，归一化一般用BatchNorm，但是在NLP领域，归一化一般用LayerNorm。
    这是由于NLP领域的输入常常是不等长的Sequence，使用BatchNorm会让较长的Sequence输入的后面特征能够使用的参与归一化的样本数太少，让输入变得不稳定。
    同时同一个Sequence的被PADDING填充的特征也会因BatchNorm获得不同的非零值，这对模型非常不友好。
    相比之下，LayerNorm总是对一个样本自己的特征进行归一化，没有上述问题。
    '''
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias

class BatchNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(features, eps=eps)
    def forward(self, x):
        B, L, dim = x.shape
        size = int(L ** 0.5)
        x_2d = x.permute(0, 2, 1).view(B, dim, size, size)
        x_norm = self.bn(x_2d)
        out = x_norm.view(B, dim, L).permute(0, 2, 1)
        return out

class InstanceNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(InstanceNorm, self).__init__()
        self.norm = nn.InstanceNorm2d(features, eps=eps)
    def forward(self, x):
        '''
        x: [B, dim, H, W]
        '''
        out = self.norm(x)      # [B, dim, H, W]
        return out

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

# 深度可扩展卷积
class Conv2d(nn.Module):
    def __init__(self, dim_in, dim_expand, dim_out, kernel_size, stride, nolinear):
        '''
        dim_in      : 特征图输入通道数
        dim_expand  : 扩展通道数
        dim_out     : 输出通道数
        kernel_size : 卷积核尺寸
        stride      : 步长
        nolinear    : 非线性层
        '''
        super(Conv2d, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(dim_in, dim_expand, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(dim_expand)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(dim_expand, dim_expand, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=dim_expand, bias=False)
        self.bn2 = nn.BatchNorm2d(dim_expand)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(dim_expand, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(dim_out)

        self.shortcut = nn.Sequential()
        if stride == 1 and dim_in != dim_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim_out),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

# GAU
class GAU(nn.Module):
    '''
    基于GAU框架的自注意力块

    初始化参数：
    dim             : token特征维数

    运行参数: 
    x               : 待处理的序列数据, [batch_size, tokens, dim]

    输出参数: 
    x               : 处理后的序列数据, [batch_size, tokens, dim]
    '''
    def __init__(self, 
        dim: int,                                                   # token的基础特征维度
        cluster: ClusterTool, 
        drop_compress: float=0.1, 
        id_block: int=0, 
        id_layer: int=0, 
        id_GAU: int=0, 
        stride: int=4, 
        gain: float=1.0, 
        shifted: bool=False, 
    ):
        super(GAU, self).__init__()
        self.id_block = id_block
        self.id_layer = id_layer
        self.id_GAU = id_GAU

        self.dim_base = dim                                         # 输入序列的特征维度

        # # 方案1：原生GAU
        # self.dim_z = dim
        # self.dim_qk = dim // 8                                      # 标准GAU框架中的Q、K矩阵的特征维度
        # # 方案2：二分聚类
        # self.dim_z = dim
        # self.dim_qk = 10 - 2 * id_block                             # 二分聚类算法中的Q、K矩阵的特征维度, 这是为了配合二分聚类, 4个block对应的特征维度分别为10, 8, 6, 4
        # 方案3：量化聚类
        self.dim_z = dim
        self.dim_qk = 2                                             # 量化聚类中采用固定的2维特征

        self.dim_vg = dim * 2                                       # V、G矩阵的特征维度
        self.scale_qk = self.dim_qk ** -0.25                        # 提前、分别对Q、K矩阵进行缩放

        # self.prenorm = RMSNorm(self.dim_base)
        # self.prenorm = nn.LayerNorm(self.dim_base)
        # self.prenorm = BatchNorm(self.dim_base)
        self.prenorm = InstanceNorm(self.dim_base)
        
        self.proj_z = nn.Sequential(
            nn.Linear(self.dim_base, self.dim_z, bias = False),     # 通过一个线性映射构造Z矩阵
            nn.SiLU(),
            nn.Linear(self.dim_z, self.dim_qk, bias = False),       # 通过一个线性映射构造Z矩阵
        )
        self.proj_q = nn.Sequential(
            ScaleOffset(self.dim_qk), 
        )
        self.proj_k = nn.Sequential(
            ScaleOffset(self.dim_qk), 
        )
        self.proj_v = nn.Sequential(
            nn.Linear(self.dim_base, self.dim_vg, bias = False),    # 通过一个线性映射构造V矩阵
            nn.SiLU(),
        )
        self.proj_g = nn.Sequential(
            nn.Linear(self.dim_base, self.dim_vg, bias = False),    # 通过一个线性映射构造G矩阵
            nn.SiLU(),
        )
        self.compress = nn.Sequential(
            nn.Linear(self.dim_vg, self.dim_base, bias = False),    # 将门控后的特征序列进行特征压缩
            # nn.Dropout(drop_compress),
        )

        self.quantizer = Quantizer(cluster)                         # 输入序列量化器

        self.stride = stride                                                    # 分块大小
        self.shifted = shifted                                                  # 是否平移特征图
        if not shifted:
            mask = getMask((stride, stride), gain)                              # [1, 1, stride**2, stride**2]
            self.register_buffer('mask', mask)                                  # 局部注意力增强掩膜（加性）
        else:
            # 左下角大片区域的mask，与unshifted时的mask一样
            mask_main = getMask((stride, stride), gain).squeeze()               # [stride**2, stride**2]
            self.register_buffer('mask_main', mask_main)                        # [stride**2, stride**2]

            stride_half = stride // 2
            # mask0, 左上角的一个mask，该mask包含四个不相邻的区域
            mask0_temp = getMask((stride_half, stride_half), gain)              # [1, 1, stride_half**2, stride_half**2]
            mask0_temp = mask0_temp.view(*mask0_temp.shape[-2:])                # [stride_half**2, stride_half**2], 不能使用squeeze方法，否则当stride_half=1时会出bug

            mask0 = torch.zeros_like(mask_main)
            for index_h in range(stride_half):
                for index_w in range(stride_half):
                    mask0[index_h:stride_half*stride:stride, index_w:stride_half*stride:stride] \
                        = mask0_temp[index_h::stride_half, index_w::stride_half]
                    mask0[index_h+stride_half:stride_half*stride+stride_half:stride, index_w+stride_half:stride_half*stride+stride_half:stride] \
                        = mask0_temp[index_h::stride_half, index_w::stride_half]
            mask0[stride_half*stride:, stride_half*stride:] = mask0[:stride_half*stride, :stride_half*stride]
            self.register_buffer('mask0', mask0)                                # [stride**2, stride**2]

            # mask1, 右上方的mask，该mask包含上下两个不相邻的区域
            mask1_temp = getMask((stride_half, stride), gain).squeeze()         # [stride_half*stride, stride_half*stride]
            mask1 = torch.zeros_like(mask_main)
            mask1[:stride_half*stride, :stride_half*stride] = mask1_temp
            mask1[stride_half*stride:, stride_half*stride:] = mask1_temp
            self.register_buffer('mask1', mask1)                                # [stride**2, stride**2]

            # mask2, 左下方的mask，该mask包含左右两个不相邻的区域
            mask2_temp = getMask((stride, stride_half), gain).squeeze()         # [tride*stride_half, stride*stride_half]
            mask2 = torch.zeros_like(mask_main)
            for index_h in range(stride_half):
                for index_w in range(stride_half):
                    mask2[index_h::stride, index_w::stride] = mask2_temp[index_h::stride_half, index_w::stride_half]
                    mask2[index_h+stride_half::stride, index_w+stride_half::stride] = mask2_temp[index_h::stride_half, index_w::stride_half]
            self.register_buffer('mask2', mask2)                                # [stride**2, stride**2]

        self.gradients = {}                                             # 调试用变量，用于存储中间变量梯度

    def hook_getGradient_q(self, grad: torch.Tensor):
        key = 'q'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_k(self, grad: torch.Tensor):
        key = 'k'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_delta_onehot(self, grad: torch.Tensor):
        key = 'delta_onehot'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_c(self, grad: torch.Tensor):
        key = 'c'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_qcT(self, grad: torch.Tensor):
        key = 'qcT'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_qct_exp(self, grad: torch.Tensor):
        key = 'qcT_exp'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_deltaTv(self, grad: torch.Tensor):
        key = 'deltaTv'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_deltaT1(self, grad: torch.Tensor):
        key = 'deltaT1'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_numerator(self, grad: torch.Tensor):
        key = 'numerator'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_denominator(self, grad: torch.Tensor):
        key = 'denominator'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_wv(self, grad: torch.Tensor):
        key = 'wv'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    def hook_getGradient_temp(self, grad: torch.Tensor):
        key = 'temp'
        if not key in self.gradients.keys():
            self.gradients[key] = []    
        self.gradients[key].append(grad)

    # 对特征图进行循环移位(所有像素向右下方移动)
    def shift(self, x):
        '''
        x: [B, dim, H, W]
        '''
        stride_half = self.stride // 2
        x = torch.roll(x, shifts=(stride_half, stride_half), dims=(-2, -1))
        return x
    
    # 恢复循环移位前的特征图(所有像素向左上方移动)
    def unshift(self, x):
        '''
        x: [B, dim, H, W]
        '''
        stride_half = self.stride // 2
        x = torch.roll(x, shifts=(-stride_half, -stride_half), dims=(-2, -1))
        return x

    def attention(self, q_blocks, k_blocks, v_blocks, H: int, W: int):
        '''
        输入数据特征维度：
        q_blocks/k_blocks   : [B, R, C, K]
        v_blocks            : [B, R, C, V]
        '''
        # if int(os.environ['RANK']) == 0 and Flag_debug:
        #     # print('--- id_block: {}, id_layer: {}, id_GAU: {} ---'.format(self.id_block, self.id_layer, self.id_GAU))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('q_blocks', q_blocks.max(), q_blocks.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('k_blocks', k_blocks.max(), k_blocks.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('v_blocks', v_blocks.max(), v_blocks.min()))

        # scale
        q_blocks = q_blocks * self.scale_qk                             # 乘以缩放系数
        k_blocks = k_blocks * self.scale_qk

        # 量化k矩阵
        k_2d = getFeaturemap(k_blocks, (H, W), self.stride)             # [B, K, H, W]
        B, K, H, W = k_2d.shape
        k_sequence = k_2d.flatten(-2).transpose(-2, -1)                 # [B, L, K]
        delta_onehot, c = self.quantizer(k_sequence)                    # [B, L, S], [B, S, K], 需要将聚类函数的输入数据形状修改为[B, dim, H, W]
        k_hat_sequence = torch.einsum('bls,bsk->blk', delta_onehot, c)  # [B, L, K]
        delta_onehot = delta_onehot.transpose(-2, -1).view(B, -1, H, W) # [B, S, H, W]
        k_hat = k_hat_sequence.transpose(-2, -1).view(*k_2d.shape)      # [B, K, H, W]

        # 矩阵分块
        delta_onehot_blocks = getBlocks(delta_onehot, self.stride)      # [B, R, C, S]
        k_hat_blocks = getBlocks(k_hat, self.stride)                    # [B, R, C, K]

        if torch.isnan(q_blocks).any() or torch.isnan(k_blocks).any() or torch.isnan(v_blocks).any():
            print('--- id_block: {}, id_layer: {}, id_GAU: {} ---'.format(self.id_block, self.id_layer, self.id_GAU))
            print('{:30s}, max: {:.9f}, min: {:.9f}'.format('q_blocks', q_blocks.max(), q_blocks.min()))
            print('{:30s}, max: {:.9f}, min: {:.9f}'.format('k_blocks', k_blocks.max(), k_blocks.min()))
            print('{:30s}, max: {:.9f}, min: {:.9f}'.format('v_blocks', v_blocks.max(), v_blocks.min()))

        assert not torch.isnan(q_blocks).any(), 'q出现NaN'
        assert not torch.isnan(k_blocks).any(), 'k出现NaN'
        assert not torch.isnan(v_blocks).any(), 'v出现NaN'
        assert not torch.isnan(c).any(), 'codebooks出现NaN'

        # 计算softmax分子
        B, R, C, K = q_blocks.shape
        *_, V = v_blocks.shape
        *_, S = delta_onehot_blocks.shape

        # qkT = torch.einsum('brck,brgk->brcg', q_blocks, k_hat_blocks)                   # [B, R, C, C], 方案1
        qk_hatT = torch.einsum('brck,brgk->brcg', q_blocks, k_hat_blocks)                   # [B, R, C, C], 方案2
        qkT = torch.einsum('brck,brgk->brcg', q_blocks, k_blocks)                           # [B, R, C, C], 方案2, 局部注意力使用未量化的k进行计算也未尝不可

        # # 不使用局部增强
        # qkTm = qkT
        # 使用局部增强
        if not self.shifted:
            qkTm = qkT + self.mask                                                                      # [B, R, C, C] + [1, 1, C, C]
        else:
            qkTm = qkT + self.mask_main.unsqueeze(0).unsqueeze(0)                                       # [B, R, C, C]，总体先加一个主要的局部增强，后面再对特殊位置进行修正
            qkTm[:, 0] += (self.mask0-self.mask_main).unsqueeze(0)                                      # [B, 1, C, C]
            qkTm[:, 1:int(R**0.5)] += (self.mask1-self.mask_main).unsqueeze(0).unsqueeze(0)             # [B, (R**0.5)-1, C, C]
            qkTm[:, int(R**0.5)::int(R**0.5)] += (self.mask2-self.mask_main).unsqueeze(0).unsqueeze(0)  # [B, (R**0.5)-1, C, C] 

        qcT = torch.einsum('brck,bsk->brcs', q_blocks, c)                               # [B, R, C, S]
        deltaTv = torch.einsum('brcs,brcv->bsv', delta_onehot_blocks, v_blocks)         # 在R维度进行了求和，[B, S, V]

        # 防溢出处理
        qkTm_max = qkTm.max(dim=-1, keepdim=True)[0]                                    # [B, R, C, 1]
        qcT_max = qcT.max(dim=-1, keepdim=True)[0]                                      # [B, R, C, 1]
        qxT_max = torch.cat((qkTm_max, qcT_max), dim=-1).max(dim=-1, keepdim=True)[0]   # [B, R, C, 1]
        qkTm -= qxT_max
        # qkT -= qxT_max                                                          # 方案1
        qkT = qk_hatT - qxT_max                                                 # 方案2
        qcT -= qxT_max

        # exp
        qkTm_exp = torch.exp(qkTm)                                              # [B, R, C, C]
        qkT_exp = torch.exp(qkT)                                                # [B, R, C, C]
        qcT_exp = torch.exp(qcT)

        # - 计算softmax分子 -
        numerator1 = torch.einsum('brcg,brgv->brcv', qkTm_exp - qkT_exp, v_blocks)
        deltaTv = torch.einsum('brcs,brcv->bsv', delta_onehot_blocks, v_blocks) # 在R维度进行了求和，[B, S, V]
        numerator2 = torch.einsum('brcs,bsv->brcv', qcT_exp, deltaTv)
        numerator_blocks = numerator1 + numerator2                              # [B, R, C, V]

        # - 计算softmax分母 -
        denominator1 = torch.einsum('brcg->brc', qkTm_exp - qkT_exp)
        deltaT1 = torch.einsum('brcs->bs', delta_onehot_blocks)                 # [B, S, 1]
        denominator2 = torch.einsum('brcs,bs->brc', qcT_exp, deltaT1)
        denominator_blocks = (denominator1 + denominator2).unsqueeze(-1)        # [B, R, C, 1]
        denominator_blocks[denominator_blocks==0] = 1e-6                        # 防止除以0
        # - 计算最终结果 -
        wv_blocks = numerator_blocks / denominator_blocks                       # [B, R, C, V]

        assert not torch.isnan(qkTm_exp).any(), 'qkTm_exp出现NaN'
        assert not torch.isnan(qkT_exp).any(), 'qkT_exp出现NaN'
        assert not torch.isnan(qcT_exp).any(), 'qcT_exp出现NaN'
        assert not torch.isnan(numerator_blocks).any(), 'numerator出现NaN'
        assert not torch.isnan(denominator_blocks).any(), 'denominator出现NaN'
        assert not torch.isnan(wv_blocks).any(), 'wv出现NaN, 推测分子或分母出现inf'

        # 统计梯度信息
        # if int(os.environ['RANK']) == 0 and Flag_debug:
        # if int(os.environ['RANK']) == 0 and self.training and False:
        #     q_blocks.register_hook(self.hook_getGradient_q)                         # 
        #     k_blocks.register_hook(self.hook_getGradient_k)                         # 
        #     delta_onehot_blocks.register_hook(self.hook_getGradient_delta_onehot)   # 
        #     c.register_hook(self.hook_getGradient_c)                                # 
        #     qcT.register_hook(self.hook_getGradient_qcT)                            # [B, R, C, C]
        #     qcT_exp.register_hook(self.hook_getGradient_qct_exp)                    # [B, R, C, C]
        #     deltaTv.register_hook(self.hook_getGradient_deltaTv)                    # 
        #     deltaT1.register_hook(self.hook_getGradient_deltaT1)                    # 
        #     numerator_blocks.register_hook(self.hook_getGradient_numerator)         # [B, R, C, V]
        #     denominator_blocks.register_hook(self.hook_getGradient_denominator)     # [B, R, C, 1]
        #     wv_blocks.register_hook(self.hook_getGradient_wv)                       # [B, R, C, V]

        # # 打印中间值
        # if int(os.environ['RANK']) == 0 and Flag_debug:
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('codebook', c.max(), c.min()))                  # [B, S, K]
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('qkTm', qkTm.max(), qkTm.min()))                # [B, R, C, C]
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('qkT', qkT.max(), qkT.min()))                   # [B, R, C, C]
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('qcT', qcT.max(), qcT.min()))                   # [B, R, C, S]
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('qkTm_exp', qkTm_exp.max(), qkTm_exp.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('qkt_exp', qkT_exp.max(), qkT_exp.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('qct_exp', qcT_exp.max(), qcT_exp.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('deltaTv', deltaTv.max(), deltaTv.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('deltaT1', deltaT1.max(), deltaT1.min()))
        #     print('sum of {}: {:.4f}'.format('deltaT1', deltaT1.sum()/deltaT1.shape[0]))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('numerator_blocks', numerator_blocks.max(), numerator_blocks.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('denominator_blocks', denominator_blocks.max(), denominator_blocks.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('wv_blocks', wv_blocks.max(), wv_blocks.min()))

        # 增加一个量化损失计算
        error = k_blocks - k_hat_blocks
        return wv_blocks, error

    def forward(self, x):
        '''
        维度符号说明：
        B: batch size
        L: tokens
        H: height of feature map
        W: width of feature map
        R: num blocks
        C/G: block length
        D/dim: d_base
        S: number of vectors in codebooks
        K: d_qk
        V: d_vg

        输入数据特征维度：
        x: [B, dim, H, W]
        '''
        if self.shifted:
            x = self.shift(x)                                           # shift, [B, dim, H, W]
        shortcut, x = x, self.prenorm(x)                                # skip connection, [B, dim, H, W]; prenorm, [B, dim, H, W]

        # # 打印中间值
        # if int(os.environ['RANK']) == 0 and Flag_debug:
        #     print('--- id_block: {}, id_layer: {}, id_GAU: {} ---'.format(self.id_block, self.id_layer, self.id_GAU))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('shortcut', shortcut.max(), shortcut.min()))
        #     print('{:30s}, max: {:.9f}, min: {:.9f}'.format('x', x.max(), x.min()))

        x_blocks = getBlocks(x, self.stride)                                            # [B, R, C, dim]
        z_blocks = self.proj_z(x_blocks)                                                # [B, R, C, K]
        q_blocks = self.proj_q(z_blocks)                                                # [B, R, C, K]
        k_blocks = self.proj_k(z_blocks)                                                # [B, R, C, K]
        v_blocks = self.proj_v(x_blocks)                                                # V矩阵, [B, R, C, V]
        g_blocks = self.proj_g(x_blocks)                                                # G门控矩阵, [B, R, C, V]
        wv_blocks, error = self.attention(q_blocks, k_blocks, v_blocks, *x.shape[-2:])  # 加权结果, [B, R, C, dim]; 量化损失, []
        out_blocks = self.compress(wv_blocks * g_blocks)                                # 特征门控与特征压缩, [B, R, C, dim]
        out = getFeaturemap(out_blocks, x.shape[-2:], self.stride) + shortcut           # 残差连接, [B, dim, H, W]
        
        # unshift
        if self.shifted:
            out = self.unshift(out)                                     # [B, dim, H, W]

        assert not torch.isnan(out).any(), 'out出现NaN'
        return out, error

# TransformerLayer
class TransformerLayer(nn.Module):
    def __init__(
        self, 
        dim: int, 
        window_size: int, 
        cluster: ClusterTool, 
        drop_compress=0.1, 
        drop_path_rate=0., 
        id_block: int = 0, 
        id_layer: int = 0
    ):
        super(TransformerLayer, self).__init__()
        self.id_block = id_block
        self.id_layer = id_layer
        self.GAU0 = GAU(
            dim=dim, 
            cluster=cluster, 
            drop_compress=drop_compress, 
            id_block=id_block, 
            id_layer=id_layer, 
            id_GAU=0, 
            stride=window_size, 
            gain=1, 
            shifted=False
        )
        self.GAU1 = GAU(
            dim=dim, 
            cluster=cluster, 
            drop_compress=drop_compress, 
            id_block=id_block, 
            id_layer=id_layer, 
            id_GAU=1, 
            stride=window_size, 
            gain=1, 
            shifted=True
        )

    def forward(self, x):
        '''
        x: [B, dim, H, W]
        '''
        x, error0 = self.GAU0(x)
        x, error1 = self.GAU1(x)
        error_mean = ((error0**2).mean() + (error1**2).mean()) / 2

        return x, error_mean

# TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        dim_in: int, 
        dim_out: int, 
        num_workers: int, 
        window_size: int, 
        layers: int, 
        drop_compress: float=0.1, 
        drop_path_rate=0.,
        id_block: int=0
    ):
        '''
        特征提取块：一个下采样层+L个Transformer层

        dim_in  : 模块输入特征维数
        dim_out : 模块输出特征维数
        layers  : Transformer层数
        '''
        super(TransformerBlock, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == layers
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(layers)]

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.id_block = id_block
        self.levels = 2 ** (5 - id_block)                               # 量化聚类中量化等级数, 4个block对应的量化等级分别是: 2^5, 2^4, 2^3, 2^2
        self.cluster = ClusterTool(num_workers=num_workers, method='quantizationcluster', levels=self.levels)
        # self.cluster = ClusterTool(num_workers=num_workers, method='binarycluster')

        self.conv = nn.Sequential(
            Conv2d(dim_in, dim_in*2, dim_out, kernel_size=3, stride=1, nolinear=nn.ReLU(inplace=True)),     # 通道调整（扩展）
            Conv2d(dim_out, dim_out*2, dim_out, kernel_size=3, stride=2, nolinear=nn.ReLU(inplace=True)),   # 尺寸压缩（/2）
        )
        self.TransformerLayers = nn.ModuleList()
        for id_layer in range(layers):
            layer = TransformerLayer(
                dim=dim_out, 
                window_size=window_size,
                cluster=self.cluster, 
                drop_compress=drop_compress, 
                drop_path_rate=drop_path_rates[id_layer], 
                id_block=id_block, 
                id_layer=id_layer
            )
            self.TransformerLayers.append(layer)
  
    def downSample(self, x: torch.tensor):
        '''
        将送入block的二维特征图下采样
        x: tensor, [B, dim, H, W]
        '''
        assert x.ndim >= 4, 'TransformerBlock输入特征图不是二维的, 请检查程序'
        x = self.conv(x)                                                # [B, dim, H, W] -> [B, dim_new, H/2, W/2]
        return x
    
    def forward(self, x):
        # 下采样
        x = self.downSample(x)                                          # [B, dim, H, W] -> [B, dim_new, H/2, W/2]

        # 特征提取
        error_sum = 0
        for TransformerLayer in self.TransformerLayers:
            x, error_mean = TransformerLayer(x)                         # [B, dim_new, H/2, W/2], []
            error_sum += error_mean

        return x, error_sum

# backbone
@MODELS.register_module()
class VQTransformer(BaseModule):
    """VQ Transformer backbone.

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
            Default: dict(type='GELU').
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
        in_channels=3,
        dims: list = [96, 96, 192, 384, 768],   # 特征嵌入维度
        num_workers: int = 2,                   # 聚类器进程数
        window_size: int = 8,                   # 局部注意力的分块大小
        blocks: list = [1, 1, 3, 1],            # 各级TransformerBlock中包含的Transformer Layer数
        drop_compress: float = 0.1,             # 

        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.1,

        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN'),
        with_cp=False,
        pretrained=None,
        frozen_stages=-1,
        init_cfg=None
    ):
        self.frozen_stages = frozen_stages

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

        self.dims = dims
        self.out_indices = out_indices
        # 图像编码 
        self.ebd = nn.Sequential(
            nn.Conv2d(in_channels, self.dims[0], kernel_size=2, stride=2),      # 原图尺寸/2
        )

        # set stochastic depth decay rule
        total_depth = sum(blocks)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        # Transformer特征提取器
        self.TransformerBlocks = nn.ModuleList()
        for index in range(blocks.__len__()):
            block = TransformerBlock(
                dim_in = self.dims[index],
                dim_out = self.dims[index+1], 
                num_workers = num_workers, 
                window_size = window_size, 
                layers = blocks[index], 
                drop_compress = drop_compress, 
                drop_path_rate=dpr[sum(blocks[:index]):sum(blocks[:index+1])],
                id_block = index
            )
            self.TransformerBlocks.append(block)

        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.dims[i+1])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.ebd.eval()
            for param in self.ebd.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.TransformerBlocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        if self.init_cfg is None:
            print_log(f'No pre-trained weights for '
                      f'{self.__class__.__name__}, '
                      f'training start from scratch')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # init.normal_(m.weight, std=0.001)
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    # init.normal_(m.weight, std=0.001)
                    # init.normal_(m.weight, std=0.01)
                    # init.uniform_(m.weight, -0.5, 0.5)
                    init.xavier_normal_(m.weight, gain=0.5)                 # 针对sigmoid或tanh等S型曲线激活函数
                    # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 针对ReLU激活函数
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                else:
                    pass
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

    def forward(self, imgs):
        x = self.ebd(imgs)                                              # 仅改变特征通道数，[b, c, h, w] -> [b, dim_ebd, h, w]
        fms = []
        error_blocks = []                                               # 每个TransformerBlock的K矩阵量化损失
        for i, TransformerBlock in enumerate(self.TransformerBlocks):
            x, error_layers = TransformerBlock(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                fm = x.permute(0, 2, 3, 1).contiguous()                 # [b, h, w, dim_ebd]
                fm = norm_layer(fm)
                fm = fm.permute(0, 3, 1, 2).contiguous()                # [b, dim_ebd, h, w]
                fms.append(fm)                                          # 收集每个TransformerBlock输出的特征图
            error_blocks.append(error_layers)                           # 收集每个TransformerBlock输出的量化误差
        return fms, torch.stack(error_blocks)                           # tensor, [b, dim, h, w]; tensor, [error_block1, error_block2, ...]
