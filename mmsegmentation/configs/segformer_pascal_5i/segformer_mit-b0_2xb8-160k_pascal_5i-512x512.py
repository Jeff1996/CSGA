_base_ = [
    '../_base_/models/segformer_mit-b0.py', 
    '../_base_/datasets/pascal_5i.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/PASCAL-5i-5953-Filtered'
crop_size = (512, 512)
train_dataloader = dict(
    batch_size=2, 
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
        folds=[0],                              # 需要使用的fold, 例如[0]
        shot=5,                                 # k-shot
    )
)

test_dataloader = val_dataloader

# 模型配置
data_preprocessor = dict(
    size=crop_size
)
checkpoint = '/home/hjf/workspace/mmsegmentation/work_dirs/pretrained_in1k/03segformer/epoch_50_heads_1248.pth'
model = dict(
    type='EncoderDecoderMod',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MixVisionTransformer',
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=32,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],         # 第三个block的heads从5->4，为了匹配聚类稀疏注意力中的聚类集合变化
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
    decode_head=dict(
        in_channels=[32, 64, 128, 256],
        num_classes=2,
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
        end=160000,
        by_epoch=False,
    )
]

train_cfg = dict(
    max_iters=160000,       # 默认值
    # max_iters=80000,
    val_interval=16000
)

# 开启推理过程可视化
default_hooks = dict(
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1)
)