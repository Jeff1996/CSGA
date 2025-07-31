import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from functools import partial

from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.utils import to_2tuple
from mmengine.model.weight_init import (constant_init, trunc_normal_, trunc_normal_init)
from mmseg.registry import MODELS

# from mmseg.utils import get_root_logger
# from mmcv.runner import load_checkpoint

iterations = 1      # k-means聚类次数

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

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

        # --------------------------- *_debug2.py的特色改动 ---------------------------
        # relative_position_bias = self.getMask((H, W), delta_onehot)
        relative_position_bias = None

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


class Block(nn.Module):
    def __init__(
        self, 
        dim, 
        num_heads, 
        mlp_ratio=4., 
        qkv_bias=False, 
        qk_scale=None, 
        attn_type='origin',
        drop=0., 
        attn_drop=0.,
        drop_path=0., 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1,
        with_cp=False,
    ):
        super().__init__()
        self.attn_type=attn_type
        self.with_cp = with_cp

        self.norm1 = norm_layer(dim)
        if attn_type == 'origin':
            self.attn = Attention(
                dim=dim,
                num_heads=num_heads, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                attn_drop=attn_drop, 
                proj_drop=drop, 
                sr_ratio=sr_ratio
            )
        elif attn_type == 'clusterattn':
            self.attn = ClusterAttn(
                embed_dims=dim,
                num_heads=num_heads,
                attn_drop_rate=attn_drop,
                proj_drop_rate=drop,
                dropout_layer=dict(type='DropPath', drop_prob=0.0), # 因为外层已经启用了drop path，所以内部就不需要了
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
            )
        else:
            raise NotImplementedError

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path))
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        '''
        x               : [B, L, dims]
        delta_onehot_x  : [B, S, H_x, W_x]
        '''
        # # 完全checkpoints
        # def _inner_forward(x: torch.Tensor, delta_onehot_x: torch.Tensor):
        #     # attn
        #     identity = x
        #     x = self.norm1(x)
        #     if self.attn_type == 'clusterattn':
        #         x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)    # [B, L, dims], [B, S, H_x, W_x]
        #     else:
        #         H, W = delta_onehot_x.shape[-2:]
        #         x = self.attn(x, H, W)                                  # [B, L, dims]
        #         delta_onehot_x_dst = delta_onehot_x
        #     x = identity + self.drop_path(x)
        #     # ffn
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     return x, delta_onehot_x_dst
        # if self.with_cp and x.requires_grad:
        #     x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        # else:
        #     x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)
        # return x, delta_onehot_x_dst

        # 仅attn部分checkpoints
        def _inner_forward(x: torch.Tensor, delta_onehot_x: torch.Tensor):
            # attn
            identity = x
            x = self.norm1(x)
            if self.attn_type == 'clusterattn':
                x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)    # [B, L, dims], [B, S, H_x, W_x]
            else:
                H, W = delta_onehot_x.shape[-2:]
                x = self.attn(x, H, W)                                  # [B, L, dims]
                delta_onehot_x_dst = delta_onehot_x
            x = identity + self.drop_path(x)
            return x, delta_onehot_x_dst
        
        if self.with_cp and x.requires_grad:
            x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        else:
            x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)
        # ffn
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, delta_onehot_x_dst

        # # 仅ffn部分checkpoints
        # def _inner_forward(x: torch.Tensor):
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     return x
        # # attn
        # identity = x
        # x = self.norm1(x)
        # if self.attn_type == 'clusterattn':
        #     x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)    # [B, L, dims], [B, S, H_x, W_x]
        # else:
        #     H, W = delta_onehot_x.shape[-2:]
        #     x = self.attn(x, H, W)                                  # [B, L, dims]
        #     delta_onehot_x_dst = delta_onehot_x
        # x = identity + self.drop_path(x)
        # # ffn
        # if self.with_cp and x.requires_grad:
        #     x = cp.checkpoint(_inner_forward, x)
        # else:
        #     x = _inner_forward(x)
        # return x, delta_onehot_x_dst


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        num_classes=1000, 
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8], 
        mlp_ratios=[4, 4, 4, 4], 
        out_indices=(0, 1, 2, 3),
        qkv_bias=False, 
        qk_scale=None, 
        stride_cluster=(8, 8),      # 构造初始聚类划分的步长，决定了聚类中心的数量
        drop_rate=0., 
        attn_drop_rate=0., 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm, 
        depths=[3, 4, 6, 3], 
        sr_ratios=[8, 4, 2, 1], 
        num_stages=4, 
        with_cp=False,
    ):
        super().__init__()
        self.stride_cluster = stride_cluster
        self.num_classes = num_classes
        self.depths = depths
        self.out_indices = out_indices
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=patch_size if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i]
            )
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i], 
                        num_heads=num_heads[i], 
                        mlp_ratio=mlp_ratios[i], 
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale, 
                        # attn_type='origin',
                        attn_type='clusterattn',
                        drop=drop_rate, 
                        attn_drop=attn_drop_rate, 
                        drop_path=dpr[cur + j],
                        norm_layer=norm_layer, 
                        sr_ratio=sr_ratios[i],
                        with_cp=with_cp,
                    ) for j in range(depths[i])
                ]
            )
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

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
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            if i == 0:  # 仅在第一个stage构造初始聚类分布
                # 获取初始聚类索引值, [B, S, H_x, W_x], pvt网络中, 初始特征图尺寸为输入图片的1/4, 所以对于512*512的输入图片尺寸, stride设置为(8, 8)
                # delta_onehot_x = self.initIndex((H, W), shape_c=(16, 16), device=x.device).repeat(x.shape[0], 1, 1, 1)
                delta_onehot_x = self.initIndex((H, W), stride=self.stride_cluster, device=x.device).repeat(x.shape[0], 1, 1, 1)
            else:       # 其余stage沿用之前的聚类分布（下采样）
                delta_onehot_x = F.interpolate(delta_onehot_x, (H, W), mode='nearest')

            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                '''
                x               : [B, L, C]
                delta_onehot_x  : [B, S, H, W]
                '''
                x, delta_onehot_x = blk(x, delta_onehot_x)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)

        return outs

# def _conv_filter(state_dict, patch_size=16):
#     """ convert patch embedding weight from manual patchify + linear proj to conv"""
#     out_dict = {}
#     for k, v in state_dict.items():
#         if 'patch_embed.proj.weight' in k:
#             v = v.reshape((v.shape[0], 3, patch_size, patch_size))
#         out_dict[k] = v
#     return out_dict


@MODELS.register_module()
class pvt_tinyMod2(PyramidVisionTransformer):
    def __init__(
        self, 
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True, 
        qk_scale=None,
        stride_cluster=(8, 8),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1], 
        drop_rate=0.0, 
        drop_path_rate=0.1,
        with_cp=False,
        **kwargs
    ):
        super(pvt_tinyMod2, self).__init__(
            img_size=img_size,
            patch_size=patch_size, 
            embed_dims=embed_dims, 
            num_heads=num_heads, 
            mlp_ratios=mlp_ratios,
            out_indices=out_indices,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            stride_cluster=stride_cluster,
            norm_layer=norm_layer, 
            depths=depths,
            sr_ratios=sr_ratios, 
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate,
            with_cp=with_cp,
        )


@MODELS.register_module()
class pvt_smallMod2(PyramidVisionTransformer):
    def __init__(
        self, 
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True, 
        qk_scale=None,
        stride_cluster=(8, 8),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1], 
        drop_rate=0.0, 
        drop_path_rate=0.1,
        with_cp=False,
        **kwargs
    ):
        super(pvt_smallMod2, self).__init__(
            img_size=img_size,
            patch_size=patch_size, 
            embed_dims=embed_dims, 
            num_heads=num_heads, 
            mlp_ratios=mlp_ratios,
            out_indices=out_indices,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            stride_cluster=stride_cluster,
            norm_layer=norm_layer, 
            depths=depths,
            sr_ratios=sr_ratios, 
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate,
            with_cp=with_cp,
        )


@MODELS.register_module()
class pvt_mediumMod2(PyramidVisionTransformer):
    def __init__(
        self, 
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True, 
        qk_scale=None,
        stride_cluster=(8, 8),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1], 
        drop_rate=0.0, 
        drop_path_rate=0.1,
        with_cp=False,
        **kwargs
    ):
        super(pvt_mediumMod2, self).__init__(
            img_size=img_size,
            patch_size=patch_size, 
            embed_dims=embed_dims, 
            num_heads=num_heads, 
            mlp_ratios=mlp_ratios,
            out_indices=out_indices,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            stride_cluster=stride_cluster,
            norm_layer=norm_layer, 
            depths=depths,
            sr_ratios=sr_ratios, 
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate,
            with_cp=with_cp,
        )


@MODELS.register_module()
class pvt_largeMod2(PyramidVisionTransformer):
    def __init__(
        self, 
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True, 
        qk_scale=None,
        stride_cluster=(8, 8),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1], 
        drop_rate=0.0, 
        drop_path_rate=0.1,
        with_cp=False,
        **kwargs
    ):
        super(pvt_largeMod2, self).__init__(
            img_size=img_size,
            patch_size=patch_size, 
            embed_dims=embed_dims, 
            num_heads=num_heads, 
            mlp_ratios=mlp_ratios,
            out_indices=out_indices,
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale,
            stride_cluster=stride_cluster,
            norm_layer=norm_layer, 
            depths=depths,
            sr_ratios=sr_ratios, 
            drop_rate=drop_rate, 
            drop_path_rate=drop_path_rate,
            with_cp=with_cp,
        )

