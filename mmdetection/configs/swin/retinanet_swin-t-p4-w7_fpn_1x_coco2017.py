_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/coco/2017/'
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

# 模型配置
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
checkpoint = '/home/hjf/workspace/mmdetection/work_dirs/pretrained_in1k/04swin/epoch_50.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
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

# 学习策略配置
# optimizer
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
    # 每隔interval个epoch保存一次权重
    checkpoint=dict(type='CheckpointHook', interval=1
    )
)
