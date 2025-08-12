_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# data setting
data_root = 'path/to/coco/'
num_gpus = 2
batch_size_pergpu = 4

train_dataloader = dict(
    batch_size=batch_size_pergpu,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
    )
)
val_dataloader = dict(
    batch_size=batch_size_pergpu//2, 
    num_workers=4, 
    dataset=dict(
        data_root=data_root,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json',
)
test_evaluator = val_evaluator


# model setting
checkpoint = 'path/to/ImageNet-1K/pre-trained/weight.pth'
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='NAT',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
        out_indices=(1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        frozen_stages=-1,
    ),
    neck=dict(
        in_channels=[128, 256, 512],
        start_level=0, 
        num_outs=5
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)


# optimization setting
optim_wrapper = dict(
    optimizer=dict(
        lr=0.01 * num_gpus * batch_size_pergpu / 16,
    )
)

# train, val, test setting
train_cfg = dict(
    val_interval=1
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1)
)
