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


# 模型配置
checkpoint = '/home/hjf/workspace/mmdetection/work_dirs/pretrained_in1k/03segformer/epoch_50_heads_1248.pth'
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,                          # 将 _base_ 中关于 backbone 的字段删除
        type='MixVisionTransformer',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cp=False,                  # 开启梯度检查点以节省内存，仅降低训练速度
    ),
    neck=dict(
        in_channels=[32, 64, 128, 256]
    ),
    init_cfg=[
        dict(type='Pretrained', checkpoint=checkpoint)
    ],
)

# 学习策略配置
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
    checkpoint=dict(type='CheckpointHook', interval=2
    )
)
