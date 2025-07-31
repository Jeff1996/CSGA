import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model.weight_init import (constant_init, normal_init, trunc_normal_init)
from mmengine.utils import to_2tuple
from mmengine.model import BaseModule, ModuleList

from mmpretrain.registry import MODELS

iterations = 1      # k-means聚类次数

# 下采样
class PointReducer(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size = patch_size,
            stride = stride, 
            padding = padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

# norm
class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

# 余弦相似度
def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
    x1 = F.normalize(x1,dim=-1)
    x2 = F.normalize(x2,dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

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
        self.qkv = nn.Conv2d(embed_dims, embed_dims * 3, kernel_size=1, bias=qkv_bias)

        # 实例化聚类器
        self.cluster = Cluster(iterations, head_embed_dims, qk_scale)
        
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Conv2d(embed_dims, embed_dims, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        # 慢速注意力需要这两项
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
        
        self.out_drop = build_dropout(dropout_layer)
    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        """
        Args:
            x (tensor)      : [B, C, H, W]
            delta_onehot_x  : [batch_size*num_heads, S, H, W]
        输出:
            x (tensor)      : [B, C, H, W]
            delta_onehot_x  : [batch_size*num_heads, S, H, W]
        """
        batch_size, C, H, W = x.shape
        L = H * W
        L_ = delta_onehot_x.shape[1]
        # [3, batch_size, num_heads, L, head_embed_dims]
        qkv = self.qkv(x).reshape(
            batch_size, 3, self.num_heads, C // self.num_heads, L
        ).permute(1, 0, 2, 4, 3)
        # [batch_size, num_heads, L, head_embed_dims]
        q, k, v = qkv[0], qkv[1], qkv[2]

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
        # [batch_size, num_heads, L, head_embed_dims]
        x = (attn @ c_v).transpose(2, 3).reshape(batch_size, C, H, W)

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
        # ).reshape(batch_size, H, W, C).float()

        # # 使用Flash-Attention 2.x的API
        # q = q.transpose(1, 2).half()       # [batch_size, L, num_heads, head_embed_dims]
        # c_k = c_k.transpose(1, 2).half()   # [batch_size, L_, num_heads, head_embed_dims]
        # c_v = c_v.transpose(1, 2).half()   # [batch_size, L_, num_heads, head_embed_dims]
        # x = flash_attn_func(q, c_k, c_v, dropout_p=self.attn_drop_rate if self.training else 0.0, softmax_scale=1.0)  # [batch_size, L, num_heads, head_embed_dims]
        # x = x.reshape(batch_size, H, W, C).float()

        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.out_drop(x)

        # return x, delta_onehot_x, error_quantization, affinity, scale
        return x, delta_onehot_x

# ffn
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_init(m, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# layer
class ClusterBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        mlp_ratio = 4.,
        act_layer = nn.GELU, 
        norm_layer = GroupNorm,
        qkv_bias=True,
        qk_scale=None,
        drop = 0., 
        attn_drop=0.0,
        drop_path = 0.,
        use_layer_scale = False, 
        layer_scale_init_value = 1e-5,
        heads = 4, 
        return_center = False,
    ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        
        self.token_mixer = ClusterAttn(
            embed_dims=dim, 
            num_heads=heads, 
            attn_drop_rate=attn_drop,
            proj_drop_rate=drop,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features = dim, 
            hidden_features = mlp_hidden_dim,
            act_layer = act_layer, 
            drop = drop
        )

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path))
        self.use_layer_scale = use_layer_scale
        self.return_center = return_center
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        # attn
        identity = x
        x, delta_onehot_x = self.token_mixer(self.norm1(x), delta_onehot_x)
        x = identity + self.drop_path(x)
        # ffn
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, delta_onehot_x

# stage
class ClusterStage(nn.Module):
    def __init__(
        self, 
        dim, 
        num_layers,
        mlp_ratio = 4.,
        act_layer = nn.GELU, 
        norm_layer = GroupNorm,
        qkv_bias=True,
        qk_scale=None,
        drop_rate = .0, 
        attn_drop_rate=0.0,
        drop_path_rate = 0.,
        use_layer_scale = False, 
        layer_scale_init_value = 1e-5,
        heads = 4, 
        return_center = False, 
        downsampler = None          # block首的下采样器
    ):

        super().__init__()

        # 下采样器，位于stage的开头处
        self.downsampler = downsampler

        self.blocks = ModuleList()
        for i in range(num_layers):
            block = ClusterBlock(
                dim = dim, 
                mlp_ratio = mlp_ratio,
                act_layer = act_layer, 
                norm_layer = norm_layer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop = drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i],
                use_layer_scale = use_layer_scale,
                layer_scale_init_value = layer_scale_init_value,
                heads = heads, 
                return_center = return_center, 
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        '''
        x: [B, C, H, W]
        '''
        if self.downsampler:
            x = self.downsampler(x)
            delta_onehot_x = F.interpolate(delta_onehot_x, x.shape[-2:], mode='nearest')
        for block in self.blocks:
            x, delta_onehot_x = block(x, delta_onehot_x)
        return x, delta_onehot_x

# 主体
class ClusterFormer(nn.Module):
    def __init__(
        self, 
        layers = [2, 2, 6, 2], 
        in_channels = 3,
        embed_dims = [96, 192, 384, 768],
        mlp_ratios = [8, 8, 4, 4],              # FFN倍率，前两个stage的倍率比swin的高一倍
        downsamples = [False, True, True, True],
        norm_layer = GroupNorm, 
        act_layer = nn.GELU,
        in_patch_size = 4, 
        in_stride = 4, 
        in_pad = 0,
        down_patch_size = 3, 
        down_stride = 2, 
        down_pad = 0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate = 0., 
        attn_drop_rate = 0., 
        drop_path_rate = 0.,
        use_layer_scale = False, 
        layer_scale_init_value = 1e-5,
        out_indices = (0, 1, 2, 3),
        init_cfg = None,
        pretrained = None,
        heads = [3,6,12,24], 
        return_center = False, 
        **kwargs
    ):
        super().__init__()
        self.out_indices = out_indices

        self.patch_embed = PointReducer(
            patch_size = in_patch_size, 
            stride = in_stride, 
            padding = in_pad,
            in_chans = in_channels + 2,     # 加2是因为输入图片还增加了二维位置索引
            embed_dim = embed_dims[0]
        )

        # drop path rate
        total_depth = sum(layers)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        # set the main block in network
        self.stages = ModuleList()
        for i in range(len(layers)):
            if downsamples[i]:
                downsampler = PointReducer(
                    patch_size = down_patch_size, 
                    stride = down_stride,
                    padding = down_pad,
                    in_chans = embed_dims[i-1], 
                    embed_dim = embed_dims[i]
                )
            else:
                downsampler = None
            stage = ClusterStage(
                dim = embed_dims[i],                                # 嵌入特征维度
                num_layers = layers[i],                             # 每个stage的layer数
                mlp_ratio = mlp_ratios[i],                          # mlp倍率
                act_layer = act_layer,                              # 激活函数，默认GELU
                norm_layer = norm_layer,                            # GroupNorm
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate = drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate = dpr[sum(layers[:i]):sum(layers[:i+1])],
                use_layer_scale = use_layer_scale,                  # False
                layer_scale_init_value = layer_scale_init_value,    # 1e-5
                heads = heads[i], 
                return_center = return_center,                      # False
                downsampler = downsampler                           # 每个stage开头的下采样, stage 0开头不需要下采样
            )
            self.stages.append(stage)

        # Add a norm layer for each output
        for i in out_indices:
            layer = norm_layer(embed_dims[i])
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # 初始化
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.init_cfg is not None or pretrained is not None:
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading imagenet pre-trained weights
    def init_weights(self, pretrained = None):
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            logger.warn('权重载入程序还没有修改完成！')
            # ckpt = _load_checkpoint(
            #     ckpt_path, logger = logger, map_location='cpu')
            # if 'state_dict' in ckpt:
            #     _state_dict = ckpt['state_dict']
            # elif 'model' in ckpt:
            #     _state_dict = ckpt['model']
            # else:
            #     _state_dict = ckpt

            # state_dict = _state_dict
            # missing_keys, unexpected_keys = \
            #     self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

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


    # 初始特征嵌入
    def forward_embeddings(self, x: torch.Tensor):
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1)/(img_w-1.0)
        range_h = torch.arange(0, img_h, step=1)/(img_h-1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing = 'ij'), dim = -1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos-0.5
        pos = fea_pos.permute(2,0,1).unsqueeze(dim=0).expand(x.shape[0],-1,-1,-1)
        x = self.patch_embed(torch.cat([x,pos], dim=1))
        return x

    # stages
    def forward_tokens(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        outs = []
        for i, block in enumerate(self.stages):
            x, delta_onehot_x = block(x, delta_onehot_x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                outs.append(out)
        return outs

    def forward(self, x: torch.Tensor):
        '''
        x: tensor, [B, 3, H, W]
        '''
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        hw_shape = x.shape[-2:]
        # delta_onehot_x = self.initIndex(hw_shape, shape_c=(16, 16), device=x.device).repeat(x.shape[0], 1, 1, 1)
        delta_onehot_x = self.initIndex(hw_shape, stride=(4, 4), device=x.device).repeat(x.shape[0], 1, 1, 1)
        outs = self.forward_tokens(x, delta_onehot_x)
        return tuple(outs)

@MODELS.register_module()
class cluster_tinyMod(ClusterFormer):
    def __init__(
        self, 
        layers = [2, 2, 6, 2],
        in_channels = 3,
        embed_dims = [96, 192, 384, 768],
        norm_layer=GroupNorm,
        mlp_ratios = [8, 8, 4, 4],
        downsamples = [False, True, True, True],
        qkv_bias=True,
        qk_scale=15.0,
        drop_rate = 0., 
        attn_drop_rate = 0., 
        drop_path_rate = 0.,
        heads = [3,6,12,24],
        down_patch_size = 3,
        down_pad = 1,
        out_indices=(0, 1, 2, 3), 
        return_center = False,          # 用于可视化中间过程
        **kwargs
    ):
        super().__init__(
            layers=layers, 
            in_channels=in_channels,
            embed_dims=embed_dims, 
            mlp_ratios=mlp_ratios, 
            downsamples=downsamples,
            norm_layer=norm_layer,
            down_patch_size=down_patch_size, 
            down_pad=down_pad,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate = drop_rate, 
            attn_drop_rate = attn_drop_rate, 
            drop_path_rate = drop_path_rate,
            heads=heads, 
            out_indices=out_indices,
            return_center=return_center, 
            **kwargs
        )

@MODELS.register_module()
class cluster_smallMod(ClusterFormer):
    def __init__(
        self,
        layers = [2, 2, 18, 2],
        norm_layer=GroupNorm,
        embed_dims = [96, 192, 384, 768],
        mlp_ratios = [8, 8, 4, 4],
        downsamples = [False, True, True, True],
        heads=[3,6,12,24],
        head_dim=[32,32,32,32],
        down_patch_size=3,
        down_pad = 1, 
        out_indices=(3,), 
        **kwargs
    ):
        super().__init__(
            layers=layers, 
            embed_dims=embed_dims, 
            norm_layer=norm_layer,
            mlp_ratios=mlp_ratios, 
            downsamples=downsamples,
            down_patch_size = down_patch_size, 
            down_pad=down_pad,
            heads=heads, 
            head_dim=head_dim,
            out_indices=out_indices,
            **kwargs
        )
