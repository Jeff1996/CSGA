_base_ = [
    '../_base_/models/upernet_nat.py', 
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# data setting
data_root = 'path/to/ade20k'
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=8,                               # 8 (batch size) * 2 (GPUS) = 16
    dataset=dict(
        data_root=data_root,
    )
)

val_dataloader = dict(
    batch_size=1, 
    dataset=dict(
        data_root=data_root,
    )
)

test_dataloader = val_dataloader

# model setting
data_preprocessor = dict(
    size=crop_size
)
checkpoint = 'path/to/ImageNet-1K/pre-trained/weight.pth'
model = dict(
    type='EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='NATMod',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
        qk_scale=15.0,
        with_cp=False,                          # gradient checkpoint
    ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=150
    ))

# 学习策略配置
# AdamW optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006,                             # bs = 16
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'rpb': dict(decay_mult=0.), 
            'norm': dict(decay_mult=0.),
        }
    ),
)

param_scheduler = [
    dict(
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
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    max_iters=160000,
    val_interval=16000
)

# default_hooks = dict(
#     visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
# )