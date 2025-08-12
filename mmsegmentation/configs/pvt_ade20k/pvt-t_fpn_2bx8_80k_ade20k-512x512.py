_base_ = [
    '../_base_/models/fpn_pvt.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# data setting
data_root = 'path/to/ade20k'
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=8, 
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
        type='pvt_tiny',
        patch_size=4,
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],                 # [1, 2, 5, 8] -> [1, 2, 4, 8]
        out_indices=(0, 1, 2, 3),
        qk_scale=None,
        with_cp=False,                          # gradient checkpoint
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    decode_head=dict(
        num_classes=150
    )
)

# optimization setting
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer = dict(
        type='AdamW', 
        lr=0.0001,                              # bs = 16
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False
    )
]

train_cfg = dict(
    max_iters=80000,
    val_interval=8000
)

# default_hooks = dict(
#     visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
# )
