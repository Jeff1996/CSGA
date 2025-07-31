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

# attn
class Clustering(nn.Module):
    def __init__(
        self, 
        dim, 
        out_dim, 
        center_w, 
        center_h, 
        window_w, 
        window_h, 
        heads, 
        head_dim, 
        qkv_bias=True, 
        return_center=False, 
        num_clustering=1
    ):
        super().__init__()
        assert dim == heads*head_dim, 'dim != heads*head_dim'
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.qkv = nn.Conv2d(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, out_dim, kernel_size=1)                 # 输出

        # 余弦相似度仿射变换
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        # 聚类中心构造？？
        self.centers_proposal = nn.AdaptiveAvgPool2d((center_w, center_h))

        self.window_w = int(window_w)
        self.window_h = int(window_h)
        self.return_center = return_center
        self.softmax = nn.Softmax(dim=-2)
        self.num_clustering = num_clustering

    def forward(self, x: torch.Tensor): 
        '''
        x: tensor, [B, C, H, W]
        '''
        b, c, h, w = x.shape
        # [B, 3*C, H, W] -> [b, 3, heads, head_dim, h, w] -> [3, b, heads, head_dim, h, w]
        x, value, feature = self.qkv(x).reshape(b, 3, self.heads, self.head_dim, h, w).permute(1, 0, 2, 3, 4, 5)

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
            out = out.permute(0, 2, 1).reshape(-1, c, h, w)
            out = self.proj(out)
            return out

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
        drop = 0., 
        drop_path = 0.,
        use_layer_scale = False, 
        layer_scale_init_value = 1e-5,
        center_w = 2, 
        center_h = 2, 
        window_w = 2, 
        window_h = 2, 
        heads = 4, 
        head_dim = 24, 
        return_center = False,
        num_clustering = 1
    ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Clustering(
            dim=dim, 
            out_dim=dim, 
            center_w=center_w, 
            center_h=center_h, 
            window_w=window_w, 
            window_h=window_h, 
            heads=heads, 
            head_dim=head_dim, 
            return_center=return_center,
            num_clustering=num_clustering
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

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# stage
class ClusterStage(nn.Module):
    def __init__(
        self, 
        dim, 
        num_layers,
        mlp_ratio = 4.,
        act_layer = nn.GELU, 
        norm_layer = GroupNorm,
        drop_rate = .0, 
        drop_path_rate = 0.,
        use_layer_scale = False, 
        layer_scale_init_value = 1e-5,
        center_w = 5, 
        center_h = 5, 
        window_w = 5, 
        window_h = 5, 
        heads = 4, 
        head_dim = 24, 
        return_center = False, 
        num_clustering = 1,
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
                drop = drop_rate, 
                drop_path = drop_path_rate[i],
                use_layer_scale = use_layer_scale,
                layer_scale_init_value = layer_scale_init_value,
                center_w = center_w, 
                center_h = center_h, 
                window_w = window_w, 
                window_h = window_h,
                heads = heads, 
                head_dim = head_dim, 
                return_center = return_center, 
                num_clustering = num_clustering
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor):
        '''
        x: [B, C, H, W]
        '''
        if self.downsampler:
            x = self.downsampler(x)
        for block in self.blocks:
            x = block(x)
        return x

# 主体
class Cluster(nn.Module):
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
        drop_rate = 0., 
        drop_path_rate = 0.,
        use_layer_scale = False, 
        layer_scale_init_value = 1e-5,
        out_indices = (0, 1, 2, 3),
        init_cfg = None,
        pretrained = None,
        center_w = [10, 10, 10, 10], 
        center_h = [10, 10, 10, 10], 
        window_w = [32, 16, 8, 4], 
        window_h = [32, 16, 8, 4],
        heads = [3,6,12,24], 
        head_dim = [32,32,32,32], 
        return_center = False, 
        num_clustering = 3,
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
                drop_rate = drop_rate,
                drop_path_rate = dpr[sum(layers[:i]):sum(layers[:i+1])],
                use_layer_scale = use_layer_scale,                  # False
                layer_scale_init_value = layer_scale_init_value,    # 1e-5
                center_w = center_w[i],                             # 聚类中心数
                center_h = center_h[i],
                window_w = window_w[i],                             # 划分小块数量（局部注意力）
                window_h = window_h[i], 
                heads = heads[i], 
                head_dim = head_dim[i],                             # head_dim = embed_dims / head_dim
                return_center = return_center,                      # False
                num_clustering = num_clustering,                    # 聚类次数，3
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
    def forward_tokens(self, x: torch.Tensor):
        outs = []
        for i, block in enumerate(self.stages):
            x = block(x)
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
        outs = self.forward_tokens(x)
        return tuple(outs)

@MODELS.register_module()
class cluster_tiny(Cluster):
    def __init__(
        self, 
        layers = [2, 2, 6, 2],
        in_channels = 3,
        embed_dims = [96, 192, 384, 768],
        norm_layer=GroupNorm,
        mlp_ratios = [8, 8, 4, 4],
        downsamples = [False, True, True, True],
        center_w=[10, 10, 10, 10],
        center_h=[10, 10, 10, 10],
        window_w=[32, 16, 8, 4],        # 划分窗口没有意义，特别是对于高层小尺寸特征图（7*7），每个窗口只有4*4大小
        window_h=[32, 16, 8, 4],
        heads=[3,6,12,24],
        head_dim=[32,32,32,32],
        down_patch_size=3,
        down_pad = 1,
        out_indices=(0, 1, 2, 3), 
        return_center = False,          # 用于可视化中间过程
        num_clustering = 3,
        **kwargs
    ):
        super().__init__(
            layers=layers, 
            in_channels=in_channels,
            embed_dims=embed_dims, 
            norm_layer=norm_layer,
            mlp_ratios=mlp_ratios, 
            downsamples=downsamples,
            down_patch_size = down_patch_size, 
            down_pad=down_pad,
            center_w=center_w, 
            center_h=center_h, 
            window_w=window_w, 
            window_h=window_h,
            heads=heads, 
            head_dim=head_dim,
            out_indices=out_indices,
            return_center = return_center, 
            num_clustering = num_clustering,
            **kwargs
        )

@MODELS.register_module()
class cluster_small(Cluster):
    def __init__(
        self,
        layers = [2, 2, 18, 2],
        norm_layer=GroupNorm,
        embed_dims = [96, 192, 384, 768],
        mlp_ratios = [8, 8, 4, 4],
        downsamples = [False, True, True, True],
        center_w=[10, 10, 10, 10],
        center_h=[10, 10, 10, 10],
        window_w=[32, 16, 8, 4],
        window_h=[32, 16, 8, 4],
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
            center_w=center_w, 
            center_h=center_h, 
            window_w=window_w, 
            window_h=window_h,
            heads=heads, 
            head_dim=head_dim,
            out_indices=out_indices,
            **kwargs
        )
