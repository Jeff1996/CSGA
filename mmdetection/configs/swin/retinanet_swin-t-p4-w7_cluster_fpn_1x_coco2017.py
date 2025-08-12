_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# data setting
data_root = 'path/to/coco/'
num_gpus = 2
batch_size_pergpu = 2

train_dataloader = dict(
    batch_size=batch_size_pergpu,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
    )
)
val_dataloader = dict(
    batch_size=batch_size_pergpu, 
    num_workers=8, 
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
        type='SwinTransformerCluster',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.2,
        patch_norm=True, 
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        # init_cfg=dict(
        #     type='Pretrained', 
        #     checkpoint=pretrained
        # )
    ),
    neck=dict(
        in_channels=[192, 384, 768], 
        start_level=0, 
        num_outs=5
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)

# optimization setting
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00003,
        betas=(0.9, 0.999), 
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }
    ),
)

# train, val, test setting
train_cfg = dict(
    val_interval=1
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1)
)
