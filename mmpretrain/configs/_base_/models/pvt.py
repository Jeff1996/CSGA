# model settings
model = dict(
    type='ImageClassifier',
    # pretrained=None,
    backbone=dict(
        type='pvt_tiny',
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8],
        out_indices=(3,),
        qk_scale=None,
        with_cp=False,
    ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=.02),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ])
)
