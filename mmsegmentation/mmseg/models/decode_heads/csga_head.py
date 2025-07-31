'''
基于CSGA的少样本分割头: 参考了DCAMA的设计，但是只使用了每个stage最后一个特征图
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

class MultiHeadCSGA(nn.Module):
    def __init__(
        self, 
        embed_dims,
        num_heads,
        attn_drop_rate=0.,
        bias = True,
        qk_scale=None,
        patch_number=16,                        # 分块数量
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.patch_number = patch_number

        # 可学习的缩放系数
        self.scale_base = qk_scale or head_embed_dims**-0.5
        self.scale = nn.Parameter(torch.tensor(1.0))

        self.proj_q = nn.Linear(embed_dims, embed_dims, bias=bias)
        self.proj_s = nn.Linear(embed_dims, embed_dims, bias=bias)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop_rate)

        if num_heads == 2:
            self.proj_out = build_conv_block(num_heads, [16, 64, 128], [5, 5, 5], [1, 1, 1])
        elif num_heads == 4:
            self.proj_out = build_conv_block(num_heads, [16, 64, 128], [5, 5, 3], [1, 1, 1])
        elif num_heads == 8:
            self.proj_out = build_conv_block(num_heads, [16, 64, 128], [5, 3, 3], [1, 1, 1])
        else:
            self.proj_out = build_conv_block(num_heads, [16, 64, 128], [3, 3, 3], [1, 1, 1])

    def forward(self, x: torch.Tensor, delta_onehot_x:torch.Tensor):
        '''
        q: tensor, [1+K, C, H, W], 单一尺度的查询样本+支持样本特征图
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

        # - 单位化的s/q矩阵，即以余弦相似度计算注意力 -
        q = F.normalize(q, dim=-1)
        s = F.normalize(s, dim=-1)
        q = q * self.scale * self.scale_base    # 让神经网络在1.0左右进行调优(类似于Faster-RCNN中，让网络预测bbox的相对尺寸、位置，而不是预测绝对值)

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
        c_s_fg = F.normalize(
            torch.einsum('ksl,khsld->khsd', delta_onehot_patches_fg, s_patches), 
            dim=-1
        )                                       # [K, num_heads, S, head_embed_dims]
        c_v_fg = torch.ones((*c_s_fg.shape[:3], ), device=c_s_fg.device)
                                                # [K, num_heads, S]
        # 背景原型
        c_s_bg = F.normalize(
            torch.einsum('ksl,khsld->khsd', delta_onehot_patches_bg, s_patches), 
            dim=-1
        )                                       # [K, num_heads, S, head_embed_dims]
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

        # - 计算注意力 -
        # q: [num_heads, H, W, head_embed_dims]
        # c_s: [num_heads, 2KS, head_embed_dims]
        # attn: [num_heads, H, W, 2KS]
        # c_v: [num_heads, 2KS]
        attn = torch.einsum('hyxd,hsd->hyxs', q, c_s) + mask_inf
                                                # [num_heads, H, W, S+S]
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = torch.einsum('hyxs,hs->hyx', attn, c_v).unsqueeze(0)
                                                # [1, num_heads, H, W]
        # x = torch.mean(
        #     x, dim=0, keepdim=True
        # ).unsqueeze(0)                          # [1, 1, H, W]

        out = self.proj_out(x)                  # [1, 1, H, W]
        return out

@MODELS.register_module()
class CSGAHead(BaseDecodeHead):
    """
    
    """
    def __init__(
        self, 
        num_heads,
        **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)
        # 多尺度特征融合
        '''
        embed_dims: 64, 128, 256, 512
        num_heads: 2, 4, 8, 16
        特征图尺寸: 128*128, 64*64, 32*32, 16*16
        '''
        self.embed_dims = self.in_channels      # [64, 128, 256, 512]
        self.num_heads = num_heads              # [2, 4, 8, 16]

        # 多尺度分割掩膜构造
        self.csgas = nn.ModuleList()
        for embed_dim, num_head in zip(self.embed_dims[1:], num_heads[1:]):
                                                # 仅对后三个尺度的特征图构造分割掩膜
            csga = MultiHeadCSGA(embed_dim, num_head, attn_drop_rate=0.5, qk_scale=15.0)
            self.csgas.append(csga)

        # 掩膜融合后处理
        self.fusion1 = build_conv_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
                                                # 16*16 -> 32*32
        self.fusion2 = build_conv_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
                                                # 32*32 -> 64*64

        # 上采样（无参数），掩膜融合时使用
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 最后的特征融合
        self.mixer = nn.Sequential(
            nn.Conv2d(128+2*self.embed_dims[-3]+2*self.embed_dims[-4], 128, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(128+2*256+2*128, 128, (3, 3), padding=(1, 1), bias=True)
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
            # nn.Conv2d(16, 2, (3, 3), padding=(1, 1), bias=True)
            #                                     # nn.Conv2d(16, 2, (3, 3), padding=(1, 1), bias=True)
        )

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
                                                # [1, 512, 512]
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
                                                # [1, 512, 512]
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
        x_pair_multiscales = self._transform_inputs(x_pair_multiscales)
                                                # 输入预处理

        # 构造三个尺度的分割掩膜
        outs = []
        for idx, x in enumerate(x_pair_multiscales[1:]):
                                                # 128*128尺度不进行掩膜构造
            out = self.csgas[idx](x, label_s)   # 尺度顺序: 64*64 -> 32*32 -> 16*16
            # print('shape of out: ', out.shape)
            outs.append(out)

        # 融合三个尺度的分割掩膜
        fusion1 = self.fusion1(
            self.upsample(outs[-1]) + outs[-2]
        )                                       # [1, 128, 32, 32]
        fusion2 = self.fusion2(
            self.upsample(fusion1) + outs[-3]
        )                                       # [1, 128, 64, 64]
        # print(fusion2.shape)

        # 特征拼接
        qs = x_pair_multiscales[-3]             # [1+K, 128, 64, 64]
        s_max = torch.max(qs[1:], dim=0, keepdim=True).values
                                                # [1, 128, 64, 64]
        output = torch.cat(
            (fusion2, qs[:1], s_max), dim=1
        )                                       # [1, 128+128+128, 64, 64]
        output = self.upsample(output)          # [1, 384, 128, 128]

        qs = x_pair_multiscales[-4]             # [1+K, 64, 128, 128]
        s_max = torch.max(qs[1:], dim=0, keepdim=True).values
                                                # [1, 64, 128, 128]
        output = torch.cat(
            (output, qs[:1], s_max), dim=1
        )                                       # [1, 384+64+64, 128, 128]

        # 特征融合
        output = self.mixer(output)             # [1, 512, 128, 128] -> [1, 16, 512, 512]
        # print(output.shape)

        output = self.cls_seg(output)           # 像素分类, [1, num_classes, H, W], H/W为最大输入特征图尺寸
        return output
