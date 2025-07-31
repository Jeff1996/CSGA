'''
用于统计码表向量利用率
'''
# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from ..utils.vqt_tools import LogOut, GradVis, ParamsDisVis
import time
import torch

@MODELS.register_module()
class EncoderDecoderSt(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    # 1. loss方法(loss模式)
    (1) backbone提取特征
    (2) 分割头前向传播+计算损失
    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: 
    (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.
    # 对应代码: 
    loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
    _decode_head_forward_train(): decode_head.loss()
    _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    # 2. predict方法(predict模式)
    (1) 调用inference方法获取未softmax的分割图
    (2) 调用post-processing方法对分割图进行后处理
    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: 
    (1) Run inference function to obtain the list of seg_logits 
    (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.
    # 对应代码: 
    predict(): inference() -> postprocess_result()
    inference(): whole_inference()/slide_inference()
    whole_inference()/slide_inference(): encoder_decoder()
    encoder_decoder(): extract_feat() -> decode_head.predict()

    # 3. _forward方法(tensor模式)
    (1) backbone提取特征
    (2) 解码头前向传播
    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: 
    (1) Extracts features to obtain the feature maps
    (2) Call the decode head forward function to forward decode head model.
    # 对应代码: 
    _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType)           : The config for the backnone of segmentor.
        decode_head (ConfigType)        : The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor. Defaults to None.
        auxiliary_head (OptConfigType)  : The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType)       : The config for training. Defaults to None.
        test_cfg (OptConfigType)        : The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional)      : The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional)       : The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(                               # 直接在框架里添加量化损失处理函数
        self,
        backbone: ConfigType,                   # 需要处理多尺度特征图和量化损失
        decode_head: ConfigType,                # 
        neck: OptConfigType = None,             # 
        auxiliary_head: OptConfigType = None,   # 可以有多个辅助头
        train_cfg: OptConfigType = None,        # 
        test_cfg: OptConfigType = None,         # 
        data_preprocessor: OptConfigType = None,# 
        pretrained: Optional[str] = None,       # 
        init_cfg: OptMultiConfig = None         # 
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, 'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)      # 构建骨干网络
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)         # 构建主分割头
        self._init_auxiliary_head(auxiliary_head)   # 构建辅助分割头

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # --- 用于网络权重参数导出的临时变量 ---
        self.images_count = 0
        self.interval_save = 1000   # 每1000张图片保存一次权重信息

        # --- affinity辅助损失函数
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean', log_target=False)

        assert self.with_decode_head

        self.num_images_total = 2000    # ADE20K
        self.num_images_current = 0
        self.statistic = {
            1:{}, 3:{}, 5:{}, 
            10:{}, 30:{}, 50:{}, 
            100:{}, 300:{}, 500:{},
            1000:{}, 3000:{}, 5000:{}
        } # 将阈值作为顶级键

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)


    # ------ loss模式相关函数 ------
    def extract_feat(self, inputs: Tensor, data_samples = None) -> List[Tensor]:    # backbone特征提取
        """Extract features from images."""
        # x = self.backbone(inputs)   # 原始结构
        # x, loss_quantization, affinity_stages, scale_stages = self.backbone(inputs, data_samples)
        x, loss_quantization, codebook_activation_stages = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x, loss_quantization, codebook_activation_stages
        # return x

    def _decode_head_forward_train(                                                 # 主分割头损失
        self, inputs: List[Tensor],
        data_samples: SampleList
    ) -> dict:                                                                      # loss模式
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses


    def _auxiliary_head_forward_train(                                              # 辅助分割头损失
        self, inputs: List[Tensor],
        data_samples: SampleList
    ) -> dict:                                                                      # loss模式
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))
        return losses
    

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

    # 获取初始相对位置编码(与初始化聚类中心构造方法有关), 如果图片尺寸不变化，则只需要构造一次
    @torch.no_grad
    def initRPE(self, shape_x, shape_c, device):
        '''
        输入：
        shape_x: [H, W]
        shape_c: [H_, W_]
        device
        输出：
        relative_position_bias_init
        '''
        H, W = shape_x
        L = H * W
        H_, W_ = shape_c
        L_ = H_ * W_
        N = (H // H_) * (W // W_)

        delta_index = torch.arange(L_, device=device).reshape(1, 1, L_, 1).repeat(1, 1, 1, N)
        delta_index = delta_index.reshape(
            1, 1, H_, W_, H // H_, W // W_
        ).permute(
            0, 1, 2, 4, 3, 5
        ).reshape(
            1, 1, H, W
        ).reshape(
            1, 1, L
        )                                                                   # [B, num_heads, L]
        delta_onehot = F.one_hot(delta_index, L_).float()                   # [B, num_heads, L, L']
        relative_position_bias_init = self.getMask((H, W), delta_onehot)    # [B, num_heads, L, L']
        return relative_position_bias_init

    @torch.no_grad
    def getAffinity_label(self, labels: torch.Tensor, stride: int = 4, num_classes: int = 151):
        '''
        labels: [B, 1, H, W]
        stride: 下采样倍率
        num_classes: 分割类别数
        '''
        B, dim, H, W = labels.shape
        assert dim == 1

        # 分块，模拟下采样的过程，更加精细
        bs = stride ** 2                                    # block size
        nb = int(H*W / bs)                                  # num blocks
        labels_unfold = F.unfold(                           # [B, nb, bs, dim]
            labels.float(), 
            kernel_size=stride, 
            padding=0, 
            stride=stride
        ).reshape(B, bs, nb).transpose(-2, -1).long()       # [B, dim*bs, nb] -> [B, nb, bs]

        # 用于区分聚类中心每一行的编码
        index_add = torch.arange(0, B*nb, device=labels.device).reshape(B, nb, 1) * num_classes
        labels_unfold_encode = labels_unfold + index_add

        # 统计聚类中心的类别成分占比
        labels_unfold_encode_flatten = labels_unfold_encode.flatten()
        counts = torch.bincount(labels_unfold_encode_flatten)
        padding = B * nb * num_classes - counts.shape[0]    # 最后一行特殊处理
        counts = torch.cat([counts, torch.zeros(padding, dtype=torch.long, device=labels.device)], dim=0)
        frequency = counts.reshape(B, nb, num_classes) / bs # [B, nb, num_classes]

        # 使用labels索引从聚类中心占比中索引出一个亲和力矩阵，亲和力矩阵的每个元素为labels元素在聚类中心分块中同类元素的占比
        labels = labels.flatten(1).long()                   # [B, L]
        affinity = []
        for i in range(B):
            affinity.append(frequency[i][:, labels[i]].transpose(-2, -1))
        affinity = torch.stack(affinity).unsqueeze(1)       # [B, 1, L, nb]
        return affinity

    # 针对k-means聚类过程中的affinity矩阵进行监督
    def _auxiliary_affinity_loss(
        self, 
        affinity_stages: list,
        scale_stages: list,
        data_samples: SampleList, 
        num_classes: int = 150,
        loss_weight: float = 1.0
    ):
        '''
        输入:
        affinity_stages : list, 一个包含stages个元素的列表, 每个元素为包含blocks/2个affinity矩阵的列表
            以swin-t架构为例: 
            [
                [None, affinity],                                   # stage0
                [None, affinity],                                   # stage1
                [None, affinity, None, affinity, None, affinity],   # stage2
                [None, None]                                        # stage3
            ]
            其中, affinity的形状为 [batch_size, num_heads, L, L']
        scale_stages    : list, 一个包含stages个元素的列表, 每个元素为包含blocks/2个缩放系数的列表
            以swin-t架构为例: 
            [
                [None, scale],                                      # stage0
                [None, scale],                                      # stage1
                [None, scale, None, scale, None, scale],            # stage2
                [None, None]                                        # stage3
            ]
            其中, scale的形状为 [1, ]
        data_samples    : SampleList, 分割标签访问方法: data_samples[i].gt_sem_seg.data, [1, H, W]

        '''
        '''
        添加以下代码前, batch_size = 4 * 2时, 训练速度为:
            [    50/160000] eta: 1 day, 4:13:53  time: 0.5460  data_time: 0.0083
            [   100/160000] eta: 1 day, 2:16:24  time: 0.5474  data_time: 0.0084
            [   150/160000] eta: 1 day, 1:39:45  time: 0.5540  data_time: 0.0094
            [   200/160000] eta: 1 day, 1:21:34  time: 0.5505  data_time: 0.0086
            [   250/160000] eta: 1 day, 1:10:31  time: 0.5554  data_time: 0.0095
            [   300/160000] eta: 1 day, 1:03:34  time: 0.5533  data_time: 0.0094
            [   350/160000] eta: 1 day, 0:57:22  time: 0.5494  data_time: 0.0088
            [   400/160000] eta: 1 day, 0:52:15  time: 0.5466  data_time: 0.0089
            [   450/160000] eta: 1 day, 0:48:24  time: 0.5476  data_time: 0.0087
            [   500/160000] eta: 1 day, 0:44:40  time: 0.5484  data_time: 0.0084
        '''
        # 1. 获取各stage的affinity和scale
        affinity_dict = {}
        size_fms = []            # 特征图尺寸
        size_cs = []             # 聚类中心尺寸
        for index_stage in range(affinity_stages.__len__()):
            affinity_blocks = affinity_stages[index_stage]
            for index_block in range(affinity_blocks.__len__()):
                affinity = affinity_blocks[index_block]
                if affinity is None:
                    continue
                else:
                    scale = scale_stages[index_stage][index_block]
                    if not index_stage in affinity_dict.keys():
                        affinity_dict[index_stage] = [[affinity, scale]]
                        size_fms.append(int(affinity.shape[-2]**0.5))
                        size_cs.append(int(affinity.shape[-1]**0.5))
                    else:
                        affinity_dict[index_stage].append([affinity, scale])

        # 2. 获取原始分割标签
        labels_original = []
        for data_sample in data_samples:
            label = data_sample.gt_sem_seg.data.float()             # [1, 512, 512]
            label[label >= num_classes] = num_classes               # 大于类别数的标签值一律置为num_classes
            labels_original.append(label)
        # 保留维数为1的通道是为了符合插值函数F.interpolate的输入数据要求[B, C, H, W]
        labels_original = torch.stack(labels_original, dim=0)       # [batch_size, 1, 512, 512]

        # 3. 构造标签在各尺度下的affinity_label
        affinity_labels = []
        for size_fm, size_c in zip(size_fms, size_cs):
            # [[B, 1, 128, 128], [B, 1, 64, 64], [B, 1, 32, 32]]
            label_subsample = F.interpolate(labels_original, (size_fm, size_fm), mode='nearest')
            stride = size_fm // size_c
            # 这个affinity是没有乘以缩放系数，没有加上相对位置编码的（因为不同block有不同的缩放系数，而相对位置编码需要缩放之后才能加上）
            # [[B, 1, 128^2, 256], [B, 1, 64^2, 256], [B, 1, 32^2, 256]]
            affinity_label = self.getAffinity_label(label_subsample, stride, num_classes+1)
            affinity_labels.append(affinity_label)
            # print(affinity_label.shape)

        # 4. 构造不同尺度的相对位置编码, [[1, 1, 128^2, 256], [1, 1, 64^2, 256], [1, 1, 32^2, 256]]
        relative_position_biases = []
        for size_fm, size_c in zip(size_fms, size_cs):
            # 使用相对位置编码
            relative_position_biases.append(
                self.initRPE((size_fm, size_fm), (size_c, size_c), labels_original.device)
            )
            # # 不使用位置编码
            # relative_position_biases.append(None)
            # print(relative_position_biases[-1].shape)

        # 5. 计算k-means的affinity(log_softmax)与affinity_label(softmax)的KL散度损失
        loss_affinity = []
        for key in affinity_dict.keys():
            affinity_label = affinity_labels[key]                       # [B, 1, L, L'], 同一个stage使用相同的label
            relative_position_bias = relative_position_biases[key]      # [1, 1, L, L'], 同一个stage使用相同的相对位置编码
            for affinity, scale in affinity_dict[key]:
                affinity_log_softmax = F.log_softmax(affinity, dim=-1)  # [batch_size, num_heads, L, L']
                if not relative_position_bias is None:
                    affinity_label_scaled = affinity_label * scale + relative_position_bias
                else:
                    affinity_label_scaled = affinity_label * scale
                affinity_label_softmax = F.softmax(                     # [batch_size, num_heads, L, L']
                    affinity_label_scaled, 
                    dim=-1
                ).repeat(1, affinity_log_softmax.shape[1], 1, 1)
                loss = self.KLDivLoss(affinity_log_softmax.flatten(0,-2), affinity_label_softmax.flatten(0,-2))
                loss_affinity.append(loss)

        if loss_affinity.__len__() == 0:
            loss_affinity = {'loss_affinity': torch.tensor(0.0)}                      # 不启用affinity loss
        else:
            loss_affinity = {'loss_affinity': sum(loss_affinity)/loss_affinity.__len__() * loss_weight}  # 各个block的affinity损失计算均值
        return loss_affinity


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # x = self.extract_feat(inputs)       # 原始结构
        x, error_quantization, *_ = self.extract_feat(inputs)  # 增加监督信号
        # x, error_quantization, affinity_stages, scale_stages = self.extract_feat(inputs, data_samples)
        losses = dict()
        # 量化损失
        loss_weight_quantization = 0.01
        loss_quantization = {'loss_quantization': error_quantization.sum() * loss_weight_quantization} # 获取骨干网络的量化损失
        losses.update(add_prefix(loss_quantization, 'backbone'))

        # # k-means损失
        # loss_affinity = self._auxiliary_affinity_loss(affinity_stages, scale_stages, data_samples, loss_weight=0.1)
        # losses.update(add_prefix(loss_affinity, 'backbone'))
        # # exit(0)

        loss_decode = self._decode_head_forward_train(x, data_samples)              # 获取主分割头损失
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)          # 获取辅助分割头损失
            losses.update(loss_aux)
        return losses


    # ------ predict模式相关函数 ------
    def slide_inference(
        self, inputs: Tensor,
        batch_img_metas: List[dict]
    ) -> Tensor:                                                                    # 
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(
                    crop_seg_logit,
                    (int(x1), int(preds.shape[3] - x2), int(y1),
                    int(preds.shape[2] - y2))
                )
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        return seg_logits

    def encode_decode(
        self, inputs: Tensor,
        batch_img_metas: List[dict],
        data_samples = None
    ) -> Tensor:                                                                    # predict模式
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, _, codebook_activation_stages = self.extract_feat(inputs)

        # --- 在这里统计码表向量使用情况（需要用一个成员变量存储所有图片在每个layer的码表使用情况） ---
        '''
        codebook_activation_stages: [
            [None, tensor[B, H, S]], 
            [None, layer1], 
            [None, layer1, None, layer3, None, layer5], 
            [None, layer1]
        ]
        '''
        # self.statistic
        for index_stage in range(codebook_activation_stages.__len__()):
            codebook_activation_layers = codebook_activation_stages[index_stage]
            for index_layer in range(codebook_activation_layers.__len__()):
                codebook_activation = codebook_activation_layers[index_layer]   # [B, heads, S]
                if not codebook_activation is None:
                    for threshold in self.statistic.keys():
                        codebook_activation_threshold = (codebook_activation >= threshold).float()      # [B, heads, S]
                        codebook_activation_norm_s = codebook_activation_threshold.mean(dim=-1)         # [B, heads]
                        codebook_activation_norm_head = codebook_activation_norm_s.mean(dim=-1)         # [B, ]

                        # 保存当前阈值下的码表利用率
                        key = 'stage_{}, layer{}'.format(index_stage, index_layer)
                        if not key in self.statistic[threshold]:
                            self.statistic[threshold][key] = []
                        self.statistic[threshold][key].append(codebook_activation_norm_head)

        # 考虑到最后一个batch的图片极大可能比设定的batchsize小，由此可以作为判断测试结束，进行信息汇总
        self.num_images_current += 1
        print('index_image: {:4d}'.format(self.num_images_current))
        if self.num_images_current >= self.num_images_total:
            # 信息汇总
            print('\n--------------------------------------------------------------------------')
            for threshold in self.statistic.keys():
                for key in self.statistic[threshold].keys():
                    activation_rates = torch.cat(self.statistic[threshold][key]).mean().data
                    self.statistic[threshold][key] = activation_rates
                print('{:4d}: {}'.format(threshold, self.statistic[threshold]))
            print('--------------------------------------------------------------------------\n')
            print(self.statistic)
        else:
            pass

        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits

    def whole_inference(
        self, inputs: Tensor,
        batch_img_metas: List[dict],
        data_samples: None
    ) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        seg_logits = self.encode_decode(inputs, batch_img_metas, data_samples)
        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict], data_samples = None) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN
            )
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas, data_samples)
        return seg_logit
    
    def predict(
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None
    ) -> SampleList:                                                                # predict模式
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        seg_logits = self.inference(inputs, batch_img_metas, data_samples)          # [batch_size, num_classes, H, W]
        return self.postprocess_result(seg_logits, data_samples)                    # 获取分割损失


    # ------ tensor模式相关函数 ------
    def _forward(                                                                   # tensor模式
        self,
        inputs: Tensor,
        data_samples: OptSampleList = None
    ) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        # x = self.extract_feat(inputs)       # 原始结构
        x, *_ = self.extract_feat(inputs)
        return self.decode_head.forward(x)


    # ------ 模型性能测试函数 ------
    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
