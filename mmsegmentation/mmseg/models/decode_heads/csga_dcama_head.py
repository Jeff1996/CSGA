'''
基于CSGA的少样本分割头: 参考了DCAMA的设计, 并且采用了多尺度、多层（每个尺度）特征融合
'''
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

from torch import Tensor
import torch.nn.functional as F
from typing import List, Tuple
from mmseg.utils import ConfigType, SampleList
from mmseg.structures import SegDataSample
from mmengine.model.weight_init import (constant_init, trunc_normal_, trunc_normal_init)

from functools import reduce
from operator import add
from PIL import Image
import numpy as np
import math, copy
from torch.autograd import Variable

def build_conv_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
    r""" bulid conv blocks """
    assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

    building_block_layers = []
    for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
        inch = in_channel if idx == 0 else out_channels[idx - 1]
        pad = ksz // 2

        building_block_layers.append(nn.Conv2d(in_channels=inch, out_channels=outch, kernel_size=ksz, stride=stride, padding=pad))
        building_block_layers.append(nn.GroupNorm(group, outch))
        building_block_layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*building_block_layers)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class MultiHeadCSGA(nn.Module):
    def __init__(
        self, 
        idx,
        embed_dims,
        num_heads,
        nlayers,
        attn_drop_rate=0.,
        bias = True,
        qk_scale=None,
        patch_number=16,                        # 分块数量
        pe = False,                             # 是否使用位置编码
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.patch_number = patch_number

        # 可学习的缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        self.scale = nn.Parameter(torch.tensor(1.0)) if qk_scale else 1.0
        self.pe = PositionalEncoding(d_model=embed_dims, dropout=0.5) if pe else nn.Identity()

        self.proj_q = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.proj_s = nn.Linear(embed_dims, embed_dims, bias=bias)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.idx_scale = idx                    # 临时变量，用于调试

        if idx == 0:                            # 最大尺度的特征图96*96(对应输入图片尺寸384*384)
            self.proj_out = build_conv_block(nlayers, [16, 64, 128], [5, 5, 5], [1, 1, 1])
        elif idx == 1:
            self.proj_out = build_conv_block(nlayers, [16, 64, 128], [5, 5, 3], [1, 1, 1])
        elif idx == 2:
            self.proj_out = build_conv_block(nlayers, [16, 64, 128], [5, 3, 3], [1, 1, 1])
        else:                                   # 最小尺度的特征图12*12(对应输入图片尺寸384*384)
            self.proj_out = build_conv_block(nlayers, [16, 64, 128], [3, 3, 3], [1, 1, 1])

    def forward(self, x_pairs_singlescale: list[torch.Tensor], delta_onehot_x:torch.Tensor):
        '''
        x_pairs_singlescale: [fm1, fm2, ..., fmn]; fm = tensor, [1+K, C, H, W]
        delta_onehot_x: tensor, [K, 1, H_original, W_original], 支持样本分割标签
        '''
        coarse_masks = []
        for idx_layer, x_pairs in enumerate(x_pairs_singlescale):
            coarse_mask = self.singleScaleForward(x_pairs, delta_onehot_x)
                                                # [1, 1, H, W]
            # coarse_mask = self.dcamaForward(x_pairs, delta_onehot_x)
            #                                     # 为了找出导致模型过拟合的根本原因，这里直接引入DCAMA的注意力模块
            coarse_masks.append(coarse_mask)

            # path_save = '/home/hjf/workspace/mmsegmentation/work_dirs/aaa/' + 'heatmap_scale{}_layer{}.png'.format(self.idx_scale, idx_layer)
            # self.save_heatmap_pil(coarse_mask, path_save)
            # print('info of coarse_masks in scale {} layer {}: min {}, max {}'.format(self.idx_scale, idx_layer, coarse_mask.min(), coarse_mask.max()))

        coarse_masks = torch.cat(coarse_masks, dim=1)
                                                # [1, layers, H, W]
        out = self.proj_out(coarse_masks)
                                                # [1, 128, H, W]
        
        # print('特征图尺寸: {}, self.scale = {}'.format(x_pairs.shape, self.scale))
        return out

    def save_heatmap_pil(self, heatmap, path='heatmap_gray.png'):
        """
        将热图保存为灰度或彩色图片（PIL）
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        # 转换为0-255的uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        img = Image.fromarray(heatmap_uint8)
        img.save(path)

    # 为了找出导致模型过拟合的根本原因，这里直接引入DCAMA的注意力模块
    def dcamaForward(self, x: torch.Tensor, delta_onehot_x:torch.Tensor):
        '''
        x: tensor, [1+K, C, H, W], 单一尺度的查询样本+支持样本特征图
        delta_onehot_x: tensor, [K, 1, H_original, W_original], 支持样本分割标签
        '''
        q = x[:1]
        s = x[1:]
        delta_onehot_x = F.interpolate(delta_onehot_x, x.shape[-2:], mode='nearest')

        K, C, H, W = s.shape
        L = H * W

        q = q.permute(0, 2, 3, 1).reshape(1, L, C)
                                                # [1, L, C]
        s = s.permute(0, 2, 3, 1).reshape(K, L, C)
                                                # [K, L, C]
        
        if self.pe:
            q = self.pe(q)                      # 添加位置编码
            s = self.pe(s)

        # - 特征映射 -
        q = self.proj_q(
            q
        ).reshape(
            L, self.num_heads, C // self.num_heads
        ).permute(
            1, 0, 2
        )                                       # [num_heads, L, head_embed_dims]
        s = self.proj_s(
            s
        ).reshape(
            -1, self.num_heads, C // self.num_heads
        ).permute(
            1, 0, 2
        )                                       # [num_heads, KL, head_embed_dims]

        # - value -
        value = delta_onehot_x.reshape(-1)      # 支持图片分割掩膜, [KL]

        # - 计算注意力 -
        # q: [num_heads, H, W, head_embed_dims]
        # c_s: [num_heads, 2KS, head_embed_dims]
        # attn: [num_heads, H, W, 2KS]
        # c_v: [num_heads, 2KS]
        attn = torch.einsum('hld,hsd->hls', q, s) * self.scale_base * self.scale
                                                # [num_heads, L, KL]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = torch.einsum('hls,s->hl', attn, value)
                                                # [num_heads, L]
        x = torch.mean(
            x, dim=0, keepdim=True
        ).reshape(
            1, 1, H, W
        )                                       # [1, 1, H, W]

        return x

    # 处理单一尺度、单层特征图输入
    def singleScaleForward(self, x: torch.Tensor, delta_onehot_x:torch.Tensor):
        '''
        x: tensor, [1+K, C, H, W], 单一尺度的查询样本+支持样本特征图
        delta_onehot_x: tensor, [K, 1, H_original, W_original], 支持样本分割标签
        '''
        q = x[:1]
        s = x[1:]
        delta_onehot_x = F.interpolate(delta_onehot_x, x.shape[-2:], mode='nearest')

        K, C, H, W = s.shape
        L = H * W

        q = q.permute(0, 2, 3, 1)               # [1, H, W, C]
        s = s.permute(0, 2, 3, 1)               # [K, H, W, C]
        
        # - 特征映射 -
        q = self.proj_q(
            q
        ).reshape(
            H, W, self.num_heads, C // self.num_heads
        ).permute(
            2, 0, 1, 3
        )                                       # [num_heads, H, W, head_embed_dims]
        s = self.proj_s(
            s
        ).reshape(
            K, H, W, self.num_heads, C // self.num_heads
        ).permute(
            0, 3, 1, 2, 4
        )                                       # [K, num_heads, H, W, head_embed_dims]

        # # - 单位化的s/q矩阵，即以余弦相似度计算注意力 -
        # q = F.normalize(q, dim=-1)
        # s = F.normalize(s, dim=-1)
        # q = q * self.scale * self.scale_base    # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

        # - 掩码分块 -
        H_patch = H // self.patch_number
        W_patch = W // self.patch_number
        # 前景掩码
        delta_onehot_patches_fg = delta_onehot_x.reshape(
            K, self.patch_number, H_patch, self.patch_number, W_patch
        ).permute(
            0, 1, 3, 2, 4
        ).reshape(
            K, -1, H_patch*W_patch
        )                                       # 前景掩膜, [K, S, l]
        delta_onehot_patches_fg_sum = torch.einsum(
            'ksl->ks', delta_onehot_patches_fg
        ).reshape(
            K, 1, -1, 1
        )                                       # [K, 1, S, 1]计算每个块内的前景数量，待会儿要用于c_v前景的归一化
        delta_onehot_patches_fg_sum[delta_onehot_patches_fg_sum < 1e-6] = 1e-6
        # 背景掩码
        delta_onehot_patches_bg = 1 - delta_onehot_patches_fg
                                                # 背景掩膜, [K, S, l]
        delta_onehot_patches_bg_sum = torch.einsum(
            'ksl->ks', delta_onehot_patches_bg
        ).reshape(
            K, 1, -1, 1
        )                                       # [K, 1, S, 1]计算每个块内的背景数量，待会儿要用于c_v背景的归一化
        delta_onehot_patches_bg_sum[delta_onehot_patches_bg_sum < 1e-6] = 1e-6

        # print('掩码情况(scale{}): delta_onehot_patches_fg_sum = {}, shape of delta_onehot_patches_fg = {}'.format(
        #     self.idx_scale, delta_onehot_patches_fg_sum.unique(), delta_onehot_patches_fg.shape
        # ))

        # - s矩阵分块 -
        s_patches = s.reshape(
            K, self.num_heads, self.patch_number, H_patch, self.patch_number, W_patch, C // self.num_heads
        ).permute(
            0, 1, 2, 4, 3, 5, 6
        ).reshape(
            K, self.num_heads, -1, H_patch*W_patch, C // self.num_heads
        )                                       # [K, num_heads, S, l, head_embed_dims]

        # - 获取原型 -
        # 前景原型
        c_s_fg = torch.einsum('ksl,khsld->khsd', delta_onehot_patches_fg, s_patches) / delta_onehot_patches_fg_sum
                                                # [K, num_heads, S, head_embed_dims]
        c_v_fg = torch.ones((*c_s_fg.shape[:3], ), device=c_s_fg.device)
                                                # [K, num_heads, S]
        # 背景原型
        c_s_bg = torch.einsum('ksl,khsld->khsd', delta_onehot_patches_bg, s_patches) / delta_onehot_patches_bg_sum
                                                # [K, num_heads, S, head_embed_dims]
        c_v_bg = torch.zeros((*c_s_bg.shape[:3], ), device=c_s_bg.device)
                                                # [K, num_heads, S]
        # 前景、背景原型拼接->多个支持图片拼接
        c_s = torch.cat(
            (c_s_fg, c_s_bg), dim=2
        ).permute(                              # [K, num_heads, S+S, head_embed_dims]
            1, 0, 2, 3
        ).reshape(                              # [num_heads, K, S+S, head_embed_dims]
            self.num_heads, -1, C // self.num_heads
        )                                       # [num_heads, 2KS, head_embed_dims]
        c_v = torch.cat(
            (c_v_fg, c_v_bg), dim=2
        ).permute(                              # [K, num_heads, S+S]
            1, 0, 2
        ).reshape(                              # [num_heads, K, S+S]
            self.num_heads, -1
        )                                       # [num_heads, 2KS]

        # - 构造注意力掩膜: 对于前景、背景聚类中没有元素的聚类中心（零向量），将在softmax之前对注意力矩阵施加-inf掩膜，避免此类聚类中心分走注意力 -
        # 前景掩膜
        mask_inf_fg = torch.zeros_like(delta_onehot_patches_fg_sum, requires_grad=False)
        mask_inf_fg[delta_onehot_patches_fg_sum < 1] = -torch.inf
                                                # [K, 1, S, 1]
        # 背景掩膜
        mask_inf_bg = torch.zeros_like(delta_onehot_patches_bg_sum, requires_grad=False)
        mask_inf_bg[delta_onehot_patches_bg_sum < 1] = -torch.inf
                                                # [K, 1, S, 1]
        # 全掩膜
        mask_inf = torch.cat(
            (mask_inf_fg, mask_inf_bg), dim=2
        ).permute(                              # [K, 1, S+S, 1]
            1, 0, 2, 3
        ).reshape(                              # [1, K, S+S, 1]
            1, 1, 1, -1
        )                                       # [1, 1, 1, 2KS]

        # print('------------------------ {} ----------------------'.format(self.idx_scale))
        # print('delta_onehot_patches_fg_sum:', delta_onehot_patches_fg_sum.reshape(K, self.patch_number, self.patch_number).long())
        # print('mask_inf_fg:', mask_inf_fg.reshape(K, self.patch_number, self.patch_number))
        # print('delta_onehot_patches_bg_sum:', delta_onehot_patches_bg_sum.reshape(K, self.patch_number, self.patch_number).long())
        # print('mask_inf_bg:', mask_inf_bg.reshape(K, self.patch_number, self.patch_number))

        # - 计算注意力 -
        # q: [num_heads, H, W, head_embed_dims]
        # c_s: [num_heads, 2KS, head_embed_dims]
        # attn: [num_heads, H, W, 2KS]
        # c_v: [num_heads, 2KS]
        attn = torch.einsum('hyxd,hsd->hyxs', q, c_s) + mask_inf
                                                # [num_heads, H, W, S+S]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = torch.einsum('hyxs,hs->hyx', attn, c_v)
                                                # [num_heads, H, W]
        x = torch.mean(
            x, dim=0, keepdim=True
        ).unsqueeze(0)                          # [1, 1, H, W]

        return x

@MODELS.register_module()
class CSGADCAMAHead(BaseDecodeHead):
    """
    
    """
    def __init__(
        self, 
        nlayers,
        num_heads,
        patch_number,
        **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)
        # 多尺度特征融合
        '''
        embed_dims: 128, 256, 512, 1024
        num_heads: 4, 8, 16, 32
        特征图尺寸: 96*96, 48*48, 24*24, 12*12
        '''
        self.embed_dims = self.in_channels      # [128, 256, 512, 1024]
        self.num_heads = num_heads              # [4, 8, 16, 32]
        self.nlayers = nlayers                  # [2, 2, 18, 2]
        
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
                                                # 特征图索引, [2,  4, 22, 24]

        # 多尺度分割掩膜构造
        self.csgas = nn.ModuleList()
        for idx, (embed_dim, num_head, nlayer) in enumerate(zip(self.embed_dims, num_heads, nlayers)):
            if idx == 0:
                continue                        # 尺寸最大的特征图不进行分割掩膜构造
                                                # 仅对后三个尺度的特征图构造分割掩膜
            # csga = MultiHeadCSGA(idx, embed_dim, num_head, nlayer, attn_drop_rate=0.5, qk_scale=15.0, patch_number=patch_number)
            csga = MultiHeadCSGA(idx, embed_dim, 8, nlayer, attn_drop_rate=0.5, patch_number=patch_number, pe=True)
            self.csgas.append(csga)

        # 掩膜融合后处理
        self.fusion1 = build_conv_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
                                                # 12*12 -> 24*24
        self.fusion2 = build_conv_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
                                                # 24*24 -> 48*48

        # 上采样（无参数），掩膜融合时使用
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 最后的特征融合
        self.mixer = nn.Sequential(
            nn.Conv2d(128+2*self.embed_dims[-3]+2*self.embed_dims[-4], 128, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(128+2*256+2*128, 128, (3, 3), padding=(1, 1), bias=True)
            # nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
            #                                     # 不进行特征拼接，直接使用粗糙分割掩膜进行分割
            nn.ReLU(),
            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Conv2d(64, 16, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(64, 16, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            nn.Conv2d(16, 16, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(16, 16, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Conv2d(16, 2, (3, 3), padding=(1, 1), bias=True)
                                                # nn.Conv2d(16, 2, (3, 3), padding=(1, 1), bias=True)
        )

        # 删除mmsegmentation自带最后一级分类头
        self.conv_seg = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 因为输入包含多个tasks，所以需要重写loss方法
    def loss(
        self, 
        x_pairs_multiscales: list[Tuple[Tensor]], 
        data_samples_pairs: list[SampleList], 
        train_cfg: ConfigType
    ) -> dict:
        '''
        输入参数:
        x_pairs_multiscales: list[pair1, ..., pairn]; pair = list[scale1, ..., scale4]; scale = tensor[1+K, C, H, W]
        data_samples_pairs: list[pair1, ..., pairn]; pair = list[data_sample1, data_sample2]

        输出参数:
        loss: dict{'loss name': value, ...}
        '''
        # 计算所有pairs的输出，构造对应的标签batch
        seg_logits = []
        batch_data_samples = []
        for x_pair_multiscales, data_samples_pair in zip(x_pairs_multiscales, data_samples_pairs):
            '''
            x_pair_multiscales: list[scale1, ..., scale4]; scale = tensor[1+K, C, H, W]
            data_samples_pair: [data_sample_q, data_sample_s]
            推理过程只需要support样本的分割标签data_samples_pair[-1]
            '''
            # 获取s样本标签
            label_s = []
            for data_sample in data_samples_pair[1:]:
                label = data_sample.gt_sem_seg.data.float() 
                                                # [1, 384, 384]
                label[label == 255] = 0         # 将padding置为0
                label_s.append(label)
            label_s = torch.stack(label_s, dim=0)
                                                # [K, 1, H, W]

            # 推理
            seg_logit = self.forward(x_pair_multiscales, label_s)
                                                # [1, C, H, W], H/W为最大输入特征图尺寸
            
            # 保存结果
            seg_logits.append(seg_logit)        # 保存s样本的处理结果
            batch_data_samples.append(data_samples_pair[0])
                                                # 保存s样本的标签，用于后续计算损失
        seg_logits = torch.cat(seg_logits, dim=0)
                                                # [b, C, H, W], b为输入的pair数量
        # print(seg_logits.shape)
        
        # 统一计算损失
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        # print(losses)
        # exit(0)
        return losses

    # 模型性能测试
    def predict(
        self, 
        x_pairs_multiscales: list[Tuple[Tensor]], 
        data_samples_pairs: list[SampleList], 
        batch_img_metas: List[dict],
        test_cfg: ConfigType
    ) -> Tensor:
        """Forward function for prediction.

        输入参数:
            x_pairs_multiscales: list[pair1, ..., pairn]; pair = list[scale1, ..., scale4]; scale = tensor[1+K, C, H, W]
            data_samples_pairs: list[pair1, ..., pairn]; pair = list[data_sample1, data_sample2]
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # 计算所有pairs的输出，构造对应的标签batch
        seg_logits = []
        for x_pair_multiscales, data_samples_pair in zip(x_pairs_multiscales, data_samples_pairs):
            '''
            x_pair_multiscales: list[scale1, ..., scale4]; scale = tensor[1+K, C, H, W]
            data_samples_pair: [data_sample_q, data_sample_s]
            推理过程只需要support样本的分割标签data_samples_pair[-1]
            '''
            # 获取s样本标签
            label_s = []
            for data_sample in data_samples_pair[1:]:
                label = data_sample.gt_sem_seg.data.float() 
                                                # [1, 384, 384]
                label[label == 255] = 0         # 将padding置为0
                label_s.append(label)
            label_s = torch.stack(label_s, dim=0)
                                                # [K, 1, H, W]
            # 推理
            seg_logit = self.forward(x_pair_multiscales, label_s)
                                                # [1, C, H, W], H/W为最大输入特征图尺寸
            # 保存结果
            seg_logits.append(seg_logit)        # 保存s样本的处理结果
        seg_logits = torch.cat(seg_logits, dim=0)
                                                # [b, C, H, W], b为输入的pair数量

        # return self.predict_by_feat(seg_logits, batch_img_metas)
        return seg_logits

    # 将输入的特征图按照尺寸进行分组
    def _transform_inputs(self, x_pair_multiscales):
        # 将特征序列转换为二维特征图
        x_pair_multiscales_temp = []
        for x_pair in x_pair_multiscales:       # [b, L, C]
            B, L, C = x_pair.shape
            H = int(L**0.5)
            W = H
            x_pair = x_pair.reshape(B, H, -1, C).permute(0, 3, 1, 2).contiguous()

            x_pair_multiscales_temp.append(x_pair)

        # 将特征图按尺度分组
        x_pair_multiscales = []
        for i, idx_end in enumerate(self.stack_ids):
            idx_start = self.stack_ids[i-1] if i >= 1 else 0
            x_pair_singlescale = x_pair_multiscales_temp[idx_start:idx_end]
            x_pair_multiscales.append(x_pair_singlescale)
        return x_pair_multiscales

    # 推理(无需查询样本的标签)
    def forward(
        self, 
        x_pair_multiscales: Tuple[Tensor],
        label_s: Tensor
    ):
        """
        Forward function.

        输入参数:
        x_pair_multiscales: 多尺度特征图, list[scale1, ..., scale4]; scale = tensor[1+K, C, H, W]
        label_s: 支持样本的二值分割标签, [K, 1, H, W]
        """
        '''
        query_feats/support_feats: list[
            shape of x: torch.Size([2, 9216, 128])
            shape of x: torch.Size([2, 9216, 128])
            shape of x: torch.Size([2, 2304, 256])
            shape of x: torch.Size([2, 2304, 256])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 576, 512])
            shape of x: torch.Size([2, 144, 1024])
            shape of x: torch.Size([2, 144, 1024])
        ]

        self.stack_ids: tensor([ 2,  4, 22, 24])
        '''

        x_pair_multiscales = self._transform_inputs(x_pair_multiscales)
                                                # 输入预处理
        # 构造三个尺度的分割掩膜
        outs = []
        for idx, x in enumerate(x_pair_multiscales[1:]):
                                                # 96*96尺度不进行掩膜构造
            # print('shape of x:', x.__len__())
            out = self.csgas[idx](x, label_s)   # 尺度顺序: 48*48 -> 24*24 -> 12*12
                                                # [1, 128, H, W]
            # print('shape of out: ', out.shape)
            outs.append(out)
        # exit(0)

        # 融合三个尺度的分割掩膜
        fusion1 = self.fusion1(
            self.upsample(outs[-1]) + outs[-2]
        )                                       # [1, 128, 24, 24]
        fusion2 = self.fusion2(
            self.upsample(fusion1) + outs[-3]
        )                                       # [1, 128, 48, 48]
        # print(fusion2.shape)
        # exit(0)
        
        # 特征拼接
        qs = x_pair_multiscales[-3][-1]         # 第2尺度的最后一个特征图, [1+K, 128, 48, 48]
        s_max = torch.max(qs[1:], dim=0, keepdim=True).values
                                                # [1, 128, 48, 48]
        output = torch.cat(
            (fusion2, qs[:1], s_max), dim=1
        )                                       # [1, 128+128+128, 48, 48]
        output = self.upsample(output)          # [1, 384, 96, 96]

        qs = x_pair_multiscales[-4][-1]         # 第1尺度的最后一个特征图, [1+K, 64, 96, 96]
        s_max = torch.max(qs[1:], dim=0, keepdim=True).values
                                                # [1, 64, 96, 96]
        output = torch.cat(
            (output, qs[:1], s_max), dim=1
        )                                       # [1, 384+64+64, 96, 96]

        # 不进行特征拼接
        # output = self.upsample(fusion2)         # [1, 384, 96, 96]

        # 特征融合
        output = self.mixer(output)             # [1, 512, 96, 96] -> [1, num_classes, 384, 384]
        # print(output.shape)

        # path_save = '/home/hjf/workspace/mmsegmentation/work_dirs/aaa/' + 'segmentation.png'
        # self.save_heatmap_pil(output.argmax(dim=1), path_save)
        # exit(0)

        # output = self.cls_seg(output)           # 像素分类, [1, num_classes, H, W], H/W为最大输入特征图尺寸

        return output


    def save_heatmap_pil(self, heatmap, path='heatmap_gray.png'):
        """
        将热图保存为灰度或彩色图片（PIL）
        """
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        # 转换为0-255的uint8
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        img = Image.fromarray(heatmap_uint8)
        img.save(path)