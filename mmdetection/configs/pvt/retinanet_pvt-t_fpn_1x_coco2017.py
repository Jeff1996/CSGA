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
checkpoint = '/home/hjf/workspace/mmdetection/work_dirs/pretrained_in1k/05pvt/epoch_50.pth'
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,                          # 将 _base_ 中关于 backbone 的字段删除
        type='pvt_tiny',
        img_size=224,
        patch_size=4, 
        embed_dims=[64, 128, 256, 512], 
        num_heads=[1, 2, 4, 8],
        out_indices=(0, 1, 2, 3),
        qk_scale=None,
        with_cp=True,
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)
# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, 
        type='AdamW', 
        lr=0.0001 * num_gpus * batch_size_pergpu / 16,
                                                # 对应batch size = 16
        weight_decay=0.0001
    )
)

# train, val, test setting
train_cfg = dict(
    val_interval=1
)

default_hooks = dict(
    # 每隔interval个epoch保存一次权重
    checkpoint=dict(type='CheckpointHook', interval=1)
)

custom_hooks = [
    dict(type='EmptyCacheHook', priority='HIGHEST'),  # 优先级设为最高确保执行
]
