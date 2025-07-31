'''
复现的DCAMA分割头
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


import math, copy
from torch.autograd import Variable
from functools import reduce
from operator import add
from PIL import Image
import numpy as np

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        query: [1, L, C]
        key: [K, L, C]
        value: [K, L]
        '''
        # print('输入MultiHeadedAttention的token尺寸: q, {}; k, {}, v, {}'.format(query.shape, key.shape, value.shape))

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)                # 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key))
                                                # [1, num_heads, L, d], [1, num_heads, K*L, d]
        ]
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
                                                # [1, num_heads, K*L, 1]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
                                                # [1, num_heads, L, 1]
        
        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)                # [1, L, 1]

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

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DCAMA_model(nn.Module):
    def __init__(self, in_channels, stack_ids):
        super(DCAMA_model, self).__init__()

        self.stack_ids = stack_ids

        # DCAMA blocks
        self.DCAMA_blocks = nn.ModuleList()
        self.pe = nn.ModuleList()
        for inch in in_channels[1:]:
            self.DCAMA_blocks.append(MultiHeadedAttention(h=8, d_model=inch, dropout=0.5))
            self.pe.append(PositionalEncoding(d_model=inch, dropout=0.5))

        outch1, outch2, outch3 = 16, 64, 128

        # conv blocks
        self.conv1 = self.build_conv_block(stack_ids[3]-stack_ids[2], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1]) # 1/32
        self.conv2 = self.build_conv_block(stack_ids[2]-stack_ids[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1]) # 1/16
        self.conv3 = self.build_conv_block(stack_ids[1]-stack_ids[0], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1]) # 1/8

        self.conv4 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/32 + 1/16
        self.conv5 = self.build_conv_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1]) # 1/16 + 1/8

        # mixer blocks
        self.mixer1 = nn.Sequential(
            nn.Conv2d(outch3+2*in_channels[1]+2*in_channels[0], outch3, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(128+2*256+2*128, 128, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU()
        )

        self.mixer2 = nn.Sequential(
            nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(64, 16, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU()
        )

        self.mixer3 = nn.Sequential(
            nn.Conv2d(outch1, outch1, (3, 3), padding=(1, 1), bias=True),
                                                # nn.Conv2d(16, 16, (3, 3), padding=(1, 1), bias=True)
            nn.ReLU(),
            nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True)
                                                # nn.Conv2d(16, 2, (3, 3), padding=(1, 1), bias=True)
        )

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

    def forward(self, query_feats, support_feats, support_mask, nshot=1):
        '''
        query_feats/support_feats: list[
            torch.Size([1或K, 128, 96, 96])
            torch.Size([1或K, 128, 96, 96])
            torch.Size([1或K, 256, 48, 48])
            torch.Size([1或K, 256, 48, 48])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 512, 24, 24])
            torch.Size([1或K, 1024, 12, 12])
            torch.Size([1或K, 1024, 12, 12])
        ]

        self.stack_ids: tensor([ 2,  4, 22, 24])
        '''
        coarse_masks = []
                                                # 对于swin-b，2-23层的每个Transformer Layer层产生的特征图都会用来构造分割掩膜
        for idx, query_feat in enumerate(query_feats):
            # 1/4 scale feature only used in skip connect
            if idx < self.stack_ids[0]: continue

            bsz, ch, ha, wa = query_feat.size()

            # reshape the input feature and mask
            query = query_feat.view(bsz, ch, -1).permute(0, 2, 1).contiguous()
                                                # [1, L, C]
            if nshot == 1:                      # 只有一张支持图片
                support_feat = support_feats[idx]
                                                # [1, C, H, W]
                mask = F.interpolate(
                    support_mask.unsqueeze(1).float(), 
                                                # [1, 1, H, W]
                    support_feat.size()[2:], 
                    mode='bilinear',
                    align_corners=True
                ).view(support_feat.size()[0], -1)
                                                # 支持图片分割标签（缩放至当前特征图尺寸）, [1, L]
                support_feat = support_feat.view(
                    support_feat.size()[0], support_feat.size()[1], -1
                ).permute(0, 2, 1).contiguous() # [1, L, C]
            else:
                support_feat = torch.stack([support_feats[k][idx] for k in range(nshot)])
                                                # 提取所有支持图片在当前尺度下的特征图, [K, C, H, W]
                support_feat = support_feat.view(-1, ch, ha * wa).permute(0, 2, 1).contiguous()
                                                # [K, L, C]
                mask = torch.stack(
                    [F.interpolate(k.unsqueeze(1).float(), (ha, wa), mode='bilinear', align_corners=True) for k in support_mask]
                )                               # [K, 1, H, W]
                mask = mask.view(bsz, -1)       # [1, K*L]

            # print('输入MultiHeadedAttention的token尺寸: q, {}; k, {}, v, {}'.format(query.shape, support_feat.shape, support_mask.shape))
            # exit(0)

            # DCAMA blocks forward
            if idx < self.stack_ids[1]:
                coarse_mask = self.DCAMA_blocks[0](self.pe[0](query), self.pe[0](support_feat), mask)
                                                # [1, L1, 1]
            elif idx < self.stack_ids[2]:
                coarse_mask = self.DCAMA_blocks[1](self.pe[1](query), self.pe[1](support_feat), mask)
                                                # [1, L2, 1]
            else:
                coarse_mask = self.DCAMA_blocks[2](self.pe[2](query), self.pe[2](support_feat), mask)
                                                # [1, L3, 1]
            coarse_masks.append(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa))
                                                # [1, 1, H, W]
            
            # path_save = '/home/hjf/workspace/mmsegmentation/work_dirs/aaa/' + 'heatmap_idx{}.png'.format(idx)
            # self.save_heatmap_pil(coarse_mask.permute(0, 2, 1).contiguous().view(bsz, 1, ha, wa), path_save)

        # multi-scale conv blocks forward
        bsz, ch, ha, wa = coarse_masks[self.stack_ids[3]-1-self.stack_ids[0]].size()
                                                # coarse_masks[21], 因为1/4尺度的2个特征图没有参与掩膜构造, 所以coarse_masks里边一共有22个掩膜
                                                # [1, 1, H1, W1]
        coarse_masks1 = torch.stack(
            coarse_masks[self.stack_ids[2]-self.stack_ids[0]:self.stack_ids[3]-self.stack_ids[0]]
                                                # coarse_masks[20:22], 1/32尺度的两个掩膜
        ).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
                                                # [1, 2, H1, W1]

        bsz, ch, ha, wa = coarse_masks[self.stack_ids[2]-1-self.stack_ids[0]].size()
        coarse_masks2 = torch.stack(
            coarse_masks[self.stack_ids[1]-self.stack_ids[0]:self.stack_ids[2]-self.stack_ids[0]]
                                                # coarse_masks[2:20], 1/16尺度的18个掩膜
        ).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
                                                # [1, 18, H2, W2]

        bsz, ch, ha, wa = coarse_masks[self.stack_ids[1]-1-self.stack_ids[0]].size()
        coarse_masks3 = torch.stack(
            coarse_masks[0:self.stack_ids[1]-self.stack_ids[0]]
                                                # coarse_masks[0:2], 1/8尺度的2个掩膜
        ).transpose(0, 1).contiguous().view(bsz, -1, ha, wa)
                                                # [1, 18, H3, W3]

        coarse_masks1 = self.conv1(coarse_masks1)
                                                # [1, 128, H1, W1]
        coarse_masks2 = self.conv2(coarse_masks2)
                                                # [1, 128, H2, W2]
        coarse_masks3 = self.conv3(coarse_masks3)
                                                # [1, 128, H3, W3]

        # multi-scale cascade (pixel-wise addition)
        coarse_masks1 = F.interpolate(coarse_masks1, coarse_masks2.size()[-2:], mode='bilinear', align_corners=True)
                                                # [1, 128, H2, W2]
        mix = coarse_masks1 + coarse_masks2     # [1, 128, H2, W2]
        mix = self.conv4(mix)                   # [1, 128, H2, W2]

        mix = F.interpolate(mix, coarse_masks3.size()[-2:], mode='bilinear', align_corners=True)
                                                # [1, 128, H3, W3]
        mix = mix + coarse_masks3               # [1, 128, H3, W3]
        mix = self.conv5(mix)                   # [1, 128, H3, W3]

        # 特征拼接看这里
        # skip connect 1/8 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[1] - 1]
                                                # support_feats[3], 1/8尺度最后一个特征图, [1, 256, H3, W3]
        else:
            support_feat = torch.stack(
                [support_feats[k][self.stack_ids[1] - 1] for k in range(nshot)]
                                                # [[1, 256, H3, W3], ...] -> [K, 1, 256, H3, W3]
            ).max(dim=0).values                 # [1, 256, H3, W3]

        mix = torch.cat((mix, query_feats[self.stack_ids[1] - 1], support_feat), 1)
                                                # [1, 128, H3, W3] cat query_feats[3] cat support_feats[3]
                                                # [1, 128, H3, W3] cat [1, 256, H3, W3] cat [1, 256, H3, W3]
                                                # -> [1, 640, H3, W3]

        upsample_size = (mix.size(-1) * 2,) * 2 # (H4, W4)
        mix = F.interpolate(mix, upsample_size, mode='bilinear', align_corners=True)
                                                # [1, 640, H4, W4]

        # skip connect 1/4 features (concatenation)
        if nshot == 1:
            support_feat = support_feats[self.stack_ids[0] - 1]
                                                # [1, 128, H4, W4]
        else:
            support_feat = torch.stack([support_feats[k][self.stack_ids[0] - 1] for k in range(nshot)]).max(dim=0).values
                                                # [1, 128, H4, W4]
        mix = torch.cat((mix, query_feats[self.stack_ids[0] - 1], support_feat), 1)
                                                # [1, 896, H4, W4]

        # mixer blocks forward
        out = self.mixer1(mix)                  # [1, 64, H4, W4]
        upsample_size = (out.size(-1) * 2,) * 2 # (H5, W5)
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
                                                # [1, 64, H5, W5]
        
        out = self.mixer2(out)                  # [1, 16, H5, W5]
        upsample_size = (out.size(-1) * 2,) * 2 # (H6, W6)
        out = F.interpolate(out, upsample_size, mode='bilinear', align_corners=True)
                                                # [1, 16, H6, W6]
        
        logit_mask = self.mixer3(out)           # [1, 2, H6, W6]

        return logit_mask

    def build_conv_block(self, in_channel, out_channels, kernel_sizes, spt_strides, group=4):
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

@MODELS.register_module()
class DCAMAHead(BaseDecodeHead):
    """
    
    """
    def __init__(
        self, 
        use_original_imgsize,
        nlayers,
        **kwargs
    ):
        super().__init__(input_transform='multiple_select', **kwargs)
        # 多尺度特征融合
        '''
        embed_dims: 64, 128, 256, 512
        num_heads: 2, 4, 8, 16
        特征图尺寸: 128*128, 64*64, 32*32, 16*16
        '''
        
        self.use_original_imgsize = use_original_imgsize
        self.feat_channels = self.in_channels   # [128, 256, 512, 1024]
        self.nlayers = nlayers                  # [2, 2, 18, 2]

        # define model
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(self.nlayers)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-4:].cumsum(dim=0)
        self.model = DCAMA_model(in_channels=self.feat_channels, stack_ids=self.stack_ids)

        # 删除mmsegmentation自带最后一级分类头
        self.conv_seg = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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
                                                # [1, 512, 512]
                label[label == 255] = 0         # 将padding置为0
                label_s.append(label)
            label_s = torch.cat(label_s, dim=0)
                                                # [K, H, W]
            # print(label_s.shape)
            # exit(0)

            # 推理
            seg_logit = self.forward(x_pair_multiscales, label_s)
                                                # [1, C, H, W], H/W为最大输入特征图尺寸
            
            # print('shape of seg_logit: ', seg_logit.shape)
            # exit(0)

            # 保存结果
            seg_logits.append(seg_logit)        # 保存q样本的处理结果
            batch_data_samples.append(data_samples_pair[0])
                                                # 保存q样本的标签，用于后续计算损失
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
            label_s = torch.cat(label_s, dim=0)
                                                # [K, H, W]
            # 推理
            seg_logit = self.forward(x_pair_multiscales, label_s)
                                                # [1, C, H, W], H/W为最大输入特征图尺寸

            # print('shape of seg_logit: ', seg_logit.shape)
            # exit(0)

            # 保存结果
            seg_logits.append(seg_logit)        # 保存s样本的处理结果
        seg_logits = torch.cat(seg_logits, dim=0)
                                                # [b, C, H, W], b为输入的pair数量

        # return self.predict_by_feat(seg_logits, batch_img_metas)
        return seg_logit

    # 将骨干网络提取的多尺度特征(q+supports)分开
    def extract_feats(self, x_pair_multiscales: Tuple[Tensor]):
        query_feats = []
        support_feats = []
        for feat in x_pair_multiscales:
            B, L, C = feat.shape
            H = int(L ** 0.5)
            W = H
            feat = feat.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            query_feats.append(feat[:1])
            support_feats.append(feat[1:])
        return query_feats, support_feats

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
        label_s: 支持样本的二值分割标签, [K, H, W]
        """
        # x_pair_multiscales = self._transform_inputs(x_pair_multiscales)
        #                                         # 输入预处理

        query_feats, support_feats = self.extract_feats(x_pair_multiscales)
        # for query_feat, support_feat in zip(query_feats, support_feats):
        #     print('输入分割头的特征图尺寸: q, {}; s, {}'.format(query_feat.shape, support_feat.shape))

        logit_mask = self.model(query_feats, support_feats, label_s.clone())

        # path_save = '/home/hjf/workspace/mmsegmentation/work_dirs/aaa/' + 'segmentation.png'
        # self.save_heatmap_pil(logit_mask.argmax(dim=1), path_save)

        # exit(0)

        # output = self.cls_seg(output)           # 像素分类, [1, num_classes, H, W], H/W为最大输入特征图尺寸
        return logit_mask
