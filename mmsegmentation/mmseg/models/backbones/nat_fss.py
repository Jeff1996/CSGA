"""
NAT+CSGA实现的小样本图像分割框架
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.runner import CheckpointLoader

from mmseg.registry import MODELS
from natten import NeighborhoodAttention2D as NeighborhoodAttention

iterations = 1      # k-means聚类次数

class ConvTokenizer(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ConvDownsampler(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Conv2d(
            dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        x = self.reduction(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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
        patch_number=16,                        # 分块数量
    ):
        super(ClusterAttn, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.patch_number = patch_number

        # 可学习的缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 构造常规qkv
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)

        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
        
        self.out_drop = build_dropout(dropout_layer)
    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)

    # # Quiet Attention
    # def softmax(self, x, dim=-1, eps=1.0):
    #     """
    #     Args:
    #         x: input tensor (shape: [..., seq_len, seq_len])
    #         dim: softmax 计算维度
    #         eps: 分母加的常数（默认为 1）
    #     Returns:
    #         softmax 结果
    #     """
    #     exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)  # 数值稳定性优化
    #     sum_exp = exp_x.sum(dim=dim, keepdim=True)
    #     return exp_x / (sum_exp + eps)  # 分母加 1

    def forward(self, x: torch.Tensor, delta_onehot_x: torch.Tensor):
        """
        Args:
            x (tensor)      : [B, H, W, C], 除第一个CSGA模块外, 后续CSGA模块收到的都是含有分支的x
                分支配列规则(训练阶段): q-q, s-s, ^s-^s, q-s, s-q, q-^s, ^s-q
                分支配列规则(验证阶段): q-s, s-s

            delta_onehot_x  : [batch_size, 1, H, W]
        输出:
            x (tensor)      : [B, H, W, C]
            delta_onehot_x  : [batch_size, 1, H, W]
        """
        B, H, W, C = x.shape
        batch_size = delta_onehot_x.shape[0]
        L = H * W

        # [3, B, num_heads, L, head_embed_dims]
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, num_heads, L, head_embed_dims]
        if batch_size == 2 or batch_size == 3:  # 训练阶段, 输入为q-s或q-s-^s及其标签
            q, k, v = qkv[0], qkv[1][:batch_size], qkv[2][:batch_size]
        else:                                   # 推理阶段，输入为q-supports及supports的标签, 此时batch_size = 1或5，分别对应1-shot和5-shot
            q, k, v = qkv[0], qkv[1][1:], qkv[2][1:]

        # 试试单位化的q/k矩阵，即以余弦相似度计算注意力
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q * self.scale * self.scale_base    # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

        # - 掩码分块 -
        H_patch = H // self.patch_number
        W_patch = W // self.patch_number
        # 前景掩码
        delta_onehot_patches_fg = delta_onehot_x.reshape(
            batch_size, self.patch_number, H_patch, self.patch_number, W_patch
        ).permute(
            0, 1, 3, 2, 4
        ).reshape(
            batch_size, -1, H_patch*W_patch
        )                                       # 前景掩膜, [batch_size, S, l]
        delta_onehot_patches_fg_sum = torch.einsum(
            'bsl->bs', delta_onehot_patches_fg
        ).reshape(
            batch_size, -1
        )                                       # [batch_size, S]计算每个块内的前景数量，待会儿要用于c_v前景的归一化
        delta_onehot_patches_fg_sum[delta_onehot_patches_fg_sum < 1e-6] = 1e-6
        # 背景掩码
        delta_onehot_patches_bg = 1 - delta_onehot_patches_fg
                                                # 背景掩膜, [batch_size, S, l]
        delta_onehot_patches_bg_sum = torch.einsum(
            'bsl->bs', delta_onehot_patches_bg
        ).reshape(
            batch_size, -1
        )                                       # [batch_size, S]计算每个块内的背景数量，待会儿要用于c_v背景的归一化
        delta_onehot_patches_bg_sum[delta_onehot_patches_bg_sum < 1e-6] = 1e-6
        # 为前景与背景合并做准备
        delta_onehot_patches_fg_sum[delta_onehot_patches_fg_sum<delta_onehot_patches_bg_sum] = 1e-6
        delta_onehot_patches_bg_sum[delta_onehot_patches_fg_sum>=delta_onehot_patches_bg_sum] = 1e-6
        delta_onehot_patches_fg[delta_onehot_patches_fg_sum<1] *= 0
        delta_onehot_patches_bg[delta_onehot_patches_bg_sum<1] *= 0
                                                # 注意，这里有可能导致梯度中断
        # - k矩阵分块 -
        k_patches = k.reshape(
            batch_size, self.num_heads, H, W, C // self.num_heads
        ).reshape(
            batch_size, self.num_heads, self.patch_number, H_patch, self.patch_number, W_patch, C // self.num_heads
        ).permute(
            0, 1, 2, 4, 3, 5, 6
        ).reshape(
            batch_size, self.num_heads, -1, H_patch*W_patch, C // self.num_heads
        )                                       # [batch_size, num_heads, S, l, head_embed_dims]

        # - v矩阵分块 -
        v_patches = v.reshape(
            batch_size, self.num_heads, H, W, C // self.num_heads
        ).reshape(
            batch_size, self.num_heads, self.patch_number, H_patch, self.patch_number, W_patch, C // self.num_heads
        ).permute(
            0, 1, 2, 4, 3, 5, 6
        ).reshape(
            batch_size, self.num_heads, -1, H_patch*W_patch, C // self.num_heads
        )                                       # [batch_size, num_heads, S, l, head_embed_dims]

        # - 获取原型 -
        # 前景原型
        c_k_fg = F.normalize(
            torch.einsum('bsl,bhsld->bhsd', delta_onehot_patches_fg, k_patches), 
            dim=-1
        )                                       # [batch_size, num_heads, S, head_embed_dims]
        c_v_fg = torch.einsum('bsl,bhsld->bhsd', delta_onehot_patches_fg, v_patches) / delta_onehot_patches_fg_sum.reshape(batch_size, 1, -1, 1) 
        # 背景原型
        c_k_bg = F.normalize(
            torch.einsum('bsl,bhsld->bhsd', delta_onehot_patches_bg, k_patches), 
            dim=-1
        )                                       # [batch_size, num_heads, S, head_embed_dims]
        c_v_bg = torch.einsum('bsl,bhsld->bhsd', delta_onehot_patches_bg, v_patches) / delta_onehot_patches_bg_sum.reshape(batch_size, 1, -1, 1) 
        # 前景、背景原型拼接
        c_k = c_k_fg + c_k_bg                   # [batch_size, num_heads, S, head_embed_dims]
        c_v = c_v_fg + c_v_bg                   # [batch_size, num_heads, S, head_embed_dims]

        # print('c_k, c_v尺寸：{}, {}'.format(c_k.shape, c_v.shape))
        
        # - 计算注意力 -
        if batch_size == 3:
            # 训练阶段，有3张图片（注意图片顺序为q-s-^s），会产生7个分支
            if B == batch_size:                 # 第一个接触q-s-^s的CSGA模块
                q = torch.stack([q[0], q[1], q[2], q[0], q[1], q[0], q[2]], dim=0)

            c_k = torch.stack([c_k[0], c_k[1], c_k[2], c_k[1], c_k[0], c_k[2], c_k[0]], dim=0)
            c_v = torch.stack([c_v[0], c_v[1], c_v[2], c_v[1], c_v[0], c_v[2], c_v[0]], dim=0)
            attn = torch.einsum('bhld,bhsd->bhls', q, c_k)
                                                # [B, num_heads, L, S]
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = torch.einsum('bhls,bhsd->bhld', attn, c_v).transpose(1, 2).reshape(-1, H, W, C)
        elif batch_size == 2:
            # 训练阶段，有2张图片（注意图片顺序为q-s），会产生4个分支
            if B == batch_size:                 # 第一个接触q-s的CSGA模块
                q = torch.stack([q[0], q[1], q[0], q[1]], dim=0)

            c_k = torch.stack([c_k[0], c_k[1], c_k[1], c_k[0]], dim=0)
            c_v = torch.stack([c_v[0], c_v[1], c_v[1], c_v[0]], dim=0)
            attn = torch.einsum('bhld,bhsd->bhls', q, c_k)
                                                # [B, num_heads, L, S]
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = torch.einsum('bhls,bhsd->bhld', attn, c_v).transpose(1, 2).reshape(-1, H, W, C)
        else:
            # 推理阶段，有2/6张图片（注意图片顺序为q-ss），不会产生分支
            # s-s自交
            q_s = q[1:]                         # [K, num_heads, L, head_embed_dims]
            attn_ss = torch.einsum('khld,khsd->khls', q_s, c_k)
                                                # [K, num_heads, L, S]
            attn_ss = self.softmax(attn_ss)
            attn_ss = self.attn_drop(attn_ss)
            ss = torch.einsum('khls,khsd->khld', attn_ss, c_v).transpose(-3, -2).reshape(-1, H, W, C)
                                                # [K, num_heads, L, head_embed_dims]

            # q-ss交互
            q_q = q[0]                          # [num_heads, L, head_embed_dims]
            c_k = c_k.permute(
                1, 0, 2, 3
            ).reshape(
                self.num_heads, -1, C//self.num_heads
            )                                   # [num_heads, batch_size*S, head_embed_dims]
            c_v = c_v.permute(
                1, 0, 2, 3
            ).reshape(
                self.num_heads, -1, C//self.num_heads
            )                                   # [num_heads, batch_size*S, head_embed_dims]
            attn_qs = torch.einsum('hld,hsd->hls', q_q, c_k)
                                                # [num_heads, L, batch_size*S]
            attn_qs = self.softmax(attn_qs)
            attn_qs = self.attn_drop(attn_qs)
            qs = torch.einsum('hls,hsd->hld', attn_qs, c_v).transpose(-3, -2).reshape(-1, H, W, C)

            x = torch.cat((qs, ss), dim=0)      # [B, H, W, C]

        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.out_drop(x)

        return x, delta_onehot_x

# 在这里切换nat和聚类稀疏全局注意力
class NATLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=7,
        dilation=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        patch_number=16,
        gla=False,          # Fasle-NAT, True-聚类稀疏全局注意力
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        with_cp=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.gla = gla
        self.with_cp = with_cp

        self.norm1 = norm_layer(dim)
        if self.gla:        # 聚类稀疏全局注意力
            self.attn = ClusterAttn(
                embed_dims=dim,
                num_heads=num_heads,
                attn_drop_rate=attn_drop,
                proj_drop_rate=drop,
                dropout_layer=dict(type='DropPath', drop_prob=0.0), # 因为外层已经有drop path了，内部的drop path需要取消
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                patch_number=patch_number,
            )
        else:               # NAT
            self.attn = NeighborhoodAttention(
                dim,
                kernel_size=kernel_size,
                dilation=dilation,
                num_heads=num_heads,
                rel_pos_bias=True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path))

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(dim), requires_grad=True
            )

    def forward(self, x, delta_onehot_x=None):
        '''
        输入
        x               : tensor, [B, H, W, C]
        delta_onehot_x  : [B, S, H_x, W_x], S是聚类中心的数量
        
        输出
        x               : tensor, [B, H, W, C]
        delta_onehot_x  : [B, S, H_x, W_x], S是聚类中心的数量
        '''
        # # 全部checkpoints
        # def _inner_forward(x, delta_onehot_x: torch.Tensor):
        #     # attn
        #     shortcut = x
        #     x = self.norm1(x)
        #     if self.gla:
        #         x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)
        #     else:
        #         x = self.attn(x)
        #         delta_onehot_x_dst = delta_onehot_x
        #     # ffn
        #     if not self.layer_scale:
        #         x = shortcut + self.drop_path(x)
        #         x = x + self.drop_path(self.mlp(self.norm2(x)))
        #     else:
        #         x = shortcut + self.drop_path(self.gamma1 * x)
        #         x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        #     return x, delta_onehot_x_dst

        # if self.with_cp and x.requires_grad:
        #     x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        # else:
        #     x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)
        # return x, delta_onehot_x_dst

        # # attn部分checkpoints
        # def _inner_forward(x, delta_onehot_x: torch.Tensor):
        #     # attn
        #     shortcut = x
        #     x = self.norm1(x)
        #     if self.gla:
        #         x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)
        #     else:
        #         x = self.attn(x)
        #         delta_onehot_x_dst = delta_onehot_x
        #     # ffn
        #     if not self.layer_scale:
        #         x = shortcut + self.drop_path(x)
        #     else:
        #         x = shortcut + self.drop_path(self.gamma1 * x)
        #     return x, delta_onehot_x_dst

        # if self.with_cp and x.requires_grad:
        #     x, delta_onehot_x_dst = cp.checkpoint(_inner_forward, x, delta_onehot_x)
        # else:
        #     x, delta_onehot_x_dst = _inner_forward(x, delta_onehot_x)

        # if not self.layer_scale:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        # else:
        #     x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        # return x, delta_onehot_x_dst

        # ffn部分checkpoints
        def _inner_forward(x):
            if not self.layer_scale:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x
        # attn
        shortcut = x
        x = self.norm1(x)
        if self.gla:
            x, delta_onehot_x_dst = self.attn(x, delta_onehot_x)
        else:
            x = self.attn(x)
            delta_onehot_x_dst = delta_onehot_x
        if not self.layer_scale:
            if not shortcut.shape[0] == x.shape[0]:
                if shortcut.shape[0] == 3:
                    shortcut = torch.stack([shortcut[0], shortcut[1], shortcut[2], shortcut[0], shortcut[1], shortcut[0], shortcut[2]], dim=0)
                elif shortcut.shape[0] == 2:
                    shortcut = torch.stack([shortcut[0], shortcut[1], shortcut[0], shortcut[1]], dim=0)
                else:
                    pass
            x = shortcut + self.drop_path(x)
        else:
            if not shortcut.shape[0] == x.shape[0]:
                if shortcut.shape[0] == 3:
                    shortcut = torch.stack([shortcut[0], shortcut[1], shortcut[2], shortcut[0], shortcut[1], shortcut[0], shortcut[2]], dim=0)
                elif shortcut.shape[0] == 2:
                    shortcut = torch.stack([shortcut[0], shortcut[1], shortcut[0], shortcut[1]], dim=0)
                else:
                    pass
            x = shortcut + self.drop_path(self.gamma1 * x)
        # ffn
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x, delta_onehot_x_dst

class NATBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        kernel_size,
        dilations=None,
        downsample=True,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        patch_number=16,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        layer_scale=None,
        with_cp=False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                NATLayer(
                    dim=dim,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    dilation=None if dilations is None else dilations[i],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=None if i % 2 == 0 else qk_scale,      # 需要匹配聚类稀疏全局注意力
                    patch_number=patch_number,
                    gla=False if i % 2 == 0 else True,              # 在这里控制聚类稀疏全局注意力的堆叠
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    layer_scale=layer_scale,
                    with_cp=with_cp,
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            None if not downsample else ConvDownsampler(dim=dim, norm_layer=norm_layer)
        )

    def forward(self, x, delta_onehot_x=None):
        outs = []
        for blk in self.blocks:
            x, delta_onehot_x = blk(x, delta_onehot_x)
            outs.append(x)
        if self.downsample:
            x = self.downsample(x)
            down_hw_shape = x.shape[1:3]
            delta_onehot_x = F.interpolate(delta_onehot_x, down_hw_shape, mode='nearest')
        return x, outs, delta_onehot_x

@MODELS.register_module()
class NATFSS(nn.Module):
    def __init__(
        self,
        embed_dim,
        mlp_ratio,
        depths,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        kernel_size=7,
        dilations=None,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_number=16,            # 分块数量
        # stride_cluster=(8, 8),      # 构造初始聚类划分的步长，决定了聚类中心的数量
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        frozen_stages=-1,
        pretrained=None,
        layer_scale=None,
        with_cp=False,
        **kwargs,
    ):
        super().__init__()
        # self.stride_cluster = stride_cluster
        self.num_levels = len(depths)
        self.embed_dim = embed_dim
        self.num_features = [int(embed_dim * 2**i) for i in range(self.num_levels)]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = ConvTokenizer(
            in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if dilations is None else dilations[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                patch_number=patch_number,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                with_cp=with_cp,
            )
            self.levels.append(level)

        # # add a norm layer for each output
        # self.out_indices = out_indices
        # for i_layer in self.out_indices:
        #     layer = norm_layer(self.num_features[i_layer])
        #     layer_name = f"norm{i_layer}"
        #     self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # if pretrained is not None:
        #     self.init_weights(pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages):
                m = self.levels[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(NATFSS, self).train(mode)
        self._freeze_stages()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
            pass
        elif pretrained is None:
            pass
        else:
            raise TypeError("pretrained must be a str or None")

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

    # 堆叠分割标签+下采样
    @torch.no_grad
    def getLabel(self, data_samples, hw_shape):
        delta_onehot_x = []
        for data_sample in data_samples:
            label = data_sample.gt_sem_seg.data.unsqueeze(dim=0).float()
                                                            # [1, 1, 512, 512]
            label_scaled = F.interpolate(label, hw_shape, mode='nearest')
                                                            # [1, 1, H_x, W_x]
            label_scaled[label_scaled == 255] = 0           # 注意一定要将padding置为0
            delta_onehot_x.append(label_scaled)
        delta_onehot_x = torch.cat(delta_onehot_x, dim=0)   # [B, 1, H_x, W_x]
        return delta_onehot_x

    # 模型推理
    def forward(self, x, data_samples = None):
        
        '''
        输入
        x: tensor, [B, H, W, C]
        '''
        x = self.patch_embed(x)

        # 获取标注
        hw_shape = x.shape[1:3]                             # [128, 128]
        delta_onehot_x = self.getLabel(data_samples, hw_shape)
                                                            # [B, 1, H_x, W_x], [3, 1, 128, 128]
        # print('delta_onehot_x: ', delta_onehot_x.shape, delta_onehot_x.unique())

        outs = []
        for idx, level in enumerate(self.levels):
            x, xo, delta_onehot_x = level(x, delta_onehot_x)
            # if idx in self.out_indices:
            #     norm_layer = getattr(self, f"norm{idx}")
            #     x_out = norm_layer(xo)
            #     outs.append(x_out.permute(0, 3, 1, 2).contiguous())
            for fm in xo:
                B, H, W, C = fm.shape
                outs.append(fm.reshape(B, -1, C))
        return outs
