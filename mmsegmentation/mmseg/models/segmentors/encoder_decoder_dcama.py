# Copyright (c) OpenMMLab. All rights reserved.
'''
少样本图像分割处理框架：仅用于DCAMA模型复现
'''
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

from PIL import Image
import numpy as np
import copy

@MODELS.register_module()
class EncoderDecoderDCAMA(BaseSegmentor):
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
        self.backbone = MODELS.build(backbone)  # 构建骨干网络
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)     # 构建主分割头
        self._init_auxiliary_head(auxiliary_head)
                                                # 构建辅助分割头

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

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
        if data_samples:
            x = self.backbone(inputs, data_samples)
        else:
            x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

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

    def loss(self, inputs: list[Tensor], data_samples: list[SampleList]) -> dict:
        """
        这个阶段通常输入为3张图片（注意图片顺序为q-s-^s），会产生7个分支
        Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images, 以task为元素构成的列表, task的数量等于batch size
            data_samples (list[:obj:`SegDataSample`]): 以task为元素构成的列表. The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 特征提取
        x_pairs_multiscales = []                # 收集所有任务、所有分支的特征提取结果: list[pair1, ..., pairn]; pair = list[scale1, ..., scale4]; scale = tensor[2, C, H, W]
        data_samples_pairs = []
        for inputs_task, data_samples_task in zip(inputs, data_samples):
            inputs_task = inputs_task[:-1]      # [2, C, H, W], 训练阶段默认构造的数据为q-s-^s, 由于DCAMA不需要^s, 所以需要去除
            x = self.extract_feat(inputs_task)
                                                # 特征提取: x = list[scale1, scale2, ..., scale24]; scale = tensor[2, c, h, w]
            x_pairs_multiscales.append(x)       # 收集多尺度特征图对

            # for x_scale in x:
            #     print(x_scale.shape)

            data_samples_pairs.append(data_samples_task[:-1])
                                                # 由于DCAMA不需要^s, 所以需要去除^s的标签

        # print('骨干网络参数名称及参数形状---------------------------------------------------')
        # params = self.backbone.state_dict()
        # for key in params:
        #     # if 'feature' in key:
        #     #     continue
        #     print('{:70s}, {}'.format(key, params[key].shape))

        # print('分割头参数名称及参数形状---------------------------------------------------')
        # params = self.decode_head.state_dict()
        # for key in params:
        #     # if 'feature' in key:
        #     #     continue
        #     print('{:70s}, {}'.format(key, params[key].shape))
        # exit(0)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x_pairs_multiscales, data_samples_pairs)
                                                # 获取主分割头损失
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_pairs_multiscales, data_samples_pairs)
                                                # 获取辅助分割头损失
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
        data_samples: SampleList = None
    ) -> Tensor:                                # predict模式
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)           # 特征提取

        x_pairs_multiscales = [x]               # [[1+K, C1, H1, W1], ..., [1+K, C4, H4, W4]]
        data_samples_pairs = [data_samples]

        seg_logits = self.decode_head.predict(x_pairs_multiscales,  data_samples_pairs, batch_img_metas[:1], self.test_cfg)
                                                # batch_img_metas只需要送入q样本的
        return seg_logits

    def whole_inference(
        self, inputs: Tensor,
        batch_img_metas: List[dict],
        data_samples: SampleList = None
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

    def inference(
        self, 
        inputs: Tensor, 
        batch_img_metas: List[dict], 
        data_samples: SampleList = None
    ) -> Tensor:
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
        inputs: list[Tensor],
        data_samples: list[OptSampleList] = None
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
        inputs_task = inputs[0]
        data_samples_task = data_samples[0]

        # print('原始图片路径:', data_samples_task[0].img_path)
        # print('原始标签路径:', data_samples_task[0].seg_map_path)

        # print('支持图片路径:', data_samples_task[-1].img_path)
        # print('支持标签路径:', data_samples_task[-1].seg_map_path)

        # print(data_samples_task)
        # exit(0)

        # 模型推理
        batch_img_metas = [data_sample.metainfo for data_sample in data_samples_task]
                                                # q-supports
        seg_logits = self.inference(inputs_task, batch_img_metas, data_samples_task)
                                                # 这个输出结果已经是排除了s样本的, [1, 2, H, W]

        # - 数据还原(图片和标签在载入时被缩放+padding了, 因此需要还原) -
        # 获取元信息
        data_sample = copy.deepcopy(data_samples_task[0])
        # img_shape_scaled = data_sample.img_shape
        # img_shape_original = data_sample.ori_shape
        # data_sample.set_metainfo({
        #     'ori_shape': data_sample.pad_shape,
        # })                                      # 不还原图片尺寸
                                                
        # 标签还原
        label_pil = Image.open(data_sample.seg_map_path)
                                                # 直接从文件系统中读取标签
        label_np = np.array(label_pil)
        label_np[label_np == 255] = 1
        label = torch.from_numpy(label_np).to(seg_logits.device).unsqueeze(dim=0).long()
                                                # [1, H, W]

        # label = data_sample.gt_sem_seg.data.unsqueeze(dim=0).float()
        #                                         # [1, 1, 512, 512], 从预处理的标签进行还原
        # label = label[:,:,:img_shape_scaled[0],:img_shape_scaled[1]]
        #                                         # 去除标签的padding
        # label = F.interpolate(label, img_shape_original, mode='nearest').squeeze(dim=0).long()
        #                                         # 缩放标签
        # if label.shape[-2] <= 0 or label.shape[-1] <= 0:
        #     print('label尺寸异常: ', label.shape)
        #     print(data_sample)
        del data_sample.gt_sem_seg.data
        data_sample.gt_sem_seg.data = label

        # import numpy as np
        # label_pil = Image.fromarray(label.squeeze().cpu().numpy().astype(np.uint8)*255)
        # path_save = '/home/hjf/workspace/mmsegmentation/work_dirs/aaa/label_recover.png'
        # print(path_save)
        # label_pil.save(path_save)
        # print(label.shape)
        # exit(0)

        # 计算模型性能指标
        return self.postprocess_result(seg_logits, [data_sample])               # 获取分割损失

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
        x = self.extract_feat(inputs)       # 原始结构
        # x, *_ = self.extract_feat(inputs)
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
