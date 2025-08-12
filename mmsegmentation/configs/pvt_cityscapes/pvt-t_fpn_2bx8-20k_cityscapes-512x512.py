_base_ = [
    '../_base_/models/fpn_pvt.py',
    '../_base_/datasets/cityscapes_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

# data setting
data_root = 'path/to/cityscapes'
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=8, 
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='image/train', seg_map_path='label/train'
        ),
    )
)

val_dataloader = dict(
    batch_size=1, 
    dataset=dict(
        data_root=data_root,
        data_prefix=dict(
            img_path='image/val', seg_map_path='label/val'
        ),
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
        num_classes=19
    )
)

# optimization setting
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer = dict(
        type='AdamW', 
        lr=0.0001,
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=20000,
        by_epoch=False
    )
]

train_cfg = dict(
    max_iters=20000,
    val_interval=2000
)

