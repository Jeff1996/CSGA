_base_ = [
    '../_base_/models/fpn_nat.py', 
    '../_base_/datasets/pascal_5i.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/PASCAL-5i-5953-Filtered'
crop_size = (512, 512)
train_dataloader = dict(
    batch_size=8, 
    dataset=dict(
        data_root=data_root,
        folds=[1,2,3],                          # 需要使用的fold, 例如[1, 2, 3]
        shot=1,                                 # k-shot
    )
)

val_dataloader = dict(
    batch_size=1, 
    dataset=dict(
        data_root=data_root,
        folds=[0],                          # 需要使用的fold, 例如[0]
        shot=1,                                 # k-shot
    )
)

test_dataloader = val_dataloader

# 模型配置
data_preprocessor = dict(
    size=crop_size
)
checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/06nat_debug/epoch_50.pth'
model = dict(
    type='EncoderDecoderFSS',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='NATFSS',
        # pretrained='https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
        qk_scale=15.0,
        frozen_stages=4,                        # 固定权重
        with_cp=False,
    ),
    neck=dict(
        in_channels=[64, 128, 256, 512]
    ),
    decode_head=dict(
        num_classes=2,
        # loss_decode=[
        #     dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        # ],
    ),
)

# 学习策略配置
# # AdamW optimizer
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', 
#         lr=0.00006,                             # batch size = 16
#         betas=(0.9, 0.999), 
#         weight_decay=0.01
#     ),
#     paramwise_cfg=dict(
#         custom_keys={
#         #     'rpb': dict(decay_mult=0.), 
#         #     'norm': dict(decay_mult=0.),
#             'backbone': dict(lr_mult=0.), 
#         }
#     ),
# )

# param_scheduler = [
#     dict(
#         type='LinearLR', 
#         start_factor=1e-6, 
#         by_epoch=False, 
#         begin=0, 
#         end=1500
#     ),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,          # 默认值
#         begin=1500,
#         end=40000,         # 默认值
#         by_epoch=False,
#     )
# ]

# train_cfg = dict(
#     max_iters=40000,
#     val_interval=4000
# )


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer = dict(
        type='AdamW', 
        lr=0.0001,              # batch size = 16
        weight_decay=0.0001
    )
)

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=10000,
        by_epoch=False
    )
]

train_cfg = dict(
    max_iters=10000,
    val_interval=1000
)

# 开启推理过程可视化
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)
