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
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='pvt_tiny',
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],
        out_indices=(0, 1, 2, 3),
        qk_scale=None,
        with_cp=False,
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)

# optimization setting
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, 
        type='AdamW', 
        lr=0.0001 * num_gpus * batch_size_pergpu / 16,
        weight_decay=0.0001
    )
)

# train, val, test setting
train_cfg = dict(
    val_interval=1
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1)
)
