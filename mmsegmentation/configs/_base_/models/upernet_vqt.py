# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='VQTransformer',
        in_channels=3,
        dims=[96, 96, 192, 384, 768],
        num_workers=2,
        window_size=8,
        blocks=[1, 1, 3, 1],
        drop_compress=0.1,
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.3,
    ),
    decode_head=dict(
        type='MyUPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    # auxiliary_head=dict(
    #     type='VQTHead',
    #     in_channels=384,
    #     in_index=2,
    #     channels=256,
    #     num_classes=150,
    #     loss_decode=dict(
    #         type='SumLoss', loss_weight=1.0)
    # ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
