_base_ = [
    '../_base_/models/segformer_mit-b0.py', 
    '../_base_/datasets/ceramics.py',                       # 修改数据集配置文件
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_20k.py'                   # 修改迭代次数
]

# 数据集配置（单卡）
data_root = '../00_datasets/ceramics/mmlab_20250311_172615' # 修改数据集路径
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
checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/03segformer/epoch_50_heads_1248.pth'
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
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
        with_cp=False,
    ),
    decode_head=dict(
        in_channels=[32, 64, 128, 256],
        num_classes=6,                                      # 修改分割类别
    )
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.00006, 
        betas=(0.9, 0.999), 
        weight_decay=0.01,
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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
        power=1.0,
        begin=1500,
        end=20000,                                          # 修改总训练次数
        by_epoch=False,
    )
]
