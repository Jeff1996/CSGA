_base_ = [
    '../_base_/models/upernet_vqt.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/ade20k'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2, 
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1, 
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline
    )
)
test_dataloader = val_dataloader


# 模型配置
data_preprocessor = dict(
    size=crop_size
)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        dims=[96, 96, 192, 384, 768],
        num_workers=2,
        window_size=8,
        blocks=[1, 1, 3, 1],
        drop_compress=0.1,
        drop_path_rate=0.3,
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768], 
        num_classes=150, 
    ),
    # auxiliary_head=dict(
    #     loss_decode=dict(
    #         type='SumLoss', loss_weight=1.0), 
    # ),
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }
    )
)

param_scheduler = [
    dict(                               # warmup
        type='LinearLR', 
        start_factor=1e-6, 
        by_epoch=False, 
        begin=0, 
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_cfg = dict(
    val_interval=8000
)
