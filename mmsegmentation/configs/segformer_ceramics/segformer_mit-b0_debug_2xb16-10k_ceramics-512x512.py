_base_ = [
    '../_base_/models/segformer_mit-b0.py', 
    '../_base_/datasets/ceramics.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_20k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/ceramics/mmlab_20250311_172615'
crop_size = (512, 512)

train_dataloader = dict(
    batch_size=16, 
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
checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/03segformer_debug/epoch_50.pth'
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformerMod',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],         # 第三个block的heads从5->4，为了匹配聚类稀疏注意力中的聚类集合变化
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=15.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cp=False,                  # 开启梯度检查点以节省内存，仅降低训练速度
    ),
    decode_head=dict(
        in_channels=[32, 64, 128, 256],
        num_classes=6,
    )
)

# 学习策略配置
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        # lr=0.00006,
        lr=1e-5, 
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
        end=500
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=5000,
        by_epoch=False,
    )
]

train_cfg = dict(
    max_iters=5000,       # 默认值
    val_interval=1000
)
