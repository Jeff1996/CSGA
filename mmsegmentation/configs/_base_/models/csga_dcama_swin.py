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
    type='EncoderDecoderDCAMA',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformerDCAMA',
        img_size=224,
        embed_dim=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        # strides=(4, 2, 2, 2),
        # out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        ape=False,
        num_classes=0,                          # 作为分割模型的骨干网络不需要分类头
        # act_cfg=dict(type='GELU'),
        # norm_cfg=backbone_norm_cfg
    ),
    decode_head=dict(
        type='CSGADCAMAHead',
        in_channels=[96, 192, 384, 768],
        channels=16,
        nlayers=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_number=16,                        # 聚类中心数量(单个方向)

        in_index=[0, 1, 2, 3],                  # 此参数无用，但是得保留

        dropout_ratio=0.,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),

    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=384,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=19,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
