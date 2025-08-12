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
    batch_size=batch_size_pergpu, 
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
    backbone=dict(
        _delete_=True,
        type='SVT',
        in_channels=3,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        patch_sizes=[4, 2, 2, 2],
        strides=[4, 2, 2, 2],
        mlp_ratios=[4, 4, 4, 4],
        windiow_sizes=[7, 7, 7, 7],
        out_indices=(1, 2, 3),
        qkv_bias=True,
        norm_cfg=dict(type='LN'),
        depths=[2, 2, 10, 4],
        sr_ratios=[8, 4, 2, 1],
        norm_after_stage=True,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        with_cp=False,
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
