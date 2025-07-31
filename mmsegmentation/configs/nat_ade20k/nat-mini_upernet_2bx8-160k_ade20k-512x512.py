_base_ = [
    '../_base_/models/upernet_nat.py', 
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/ade20k'
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


# 模型配置
data_preprocessor = dict(
    size=crop_size
)

checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/06nat/epoch_50.pth'

model = dict(
    type='EncoderDecoder',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='NAT',
        # pretrained='https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
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
        lr=0.00006,                     # batch size = 16
        # lr=0.00003, 
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
        power=1.0,          # 默认值
        begin=1500,
        end=160000,         # 默认值
        by_epoch=False,
    )
]

# # Mixed precision, 这个配置的功能未知，先注释了训练看看
# fp16 = None
# optimizer_config = dict(
#     type="Fp16OptimizerHook",
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
# )

train_cfg = dict(
    max_iters=160000,       # 默认值
    # max_iters=80000,
    val_interval=16000
)

# 开启推理过程可视化
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)