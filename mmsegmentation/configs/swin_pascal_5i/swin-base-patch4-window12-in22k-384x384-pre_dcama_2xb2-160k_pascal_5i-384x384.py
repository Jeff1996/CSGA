_base_ = [
    '../_base_/models/dcama_swin.py', 
    '../_base_/datasets/pascal_5i_384x384.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]

# 数据集配置（单卡）
data_root = '../00_datasets/PASCAL-5i-5953-Filtered'
# data_root = '../00_datasets/PASCAL-5i-10582-Filtered'
crop_size = (384, 384)
num_gpus = 2                                    # GPU数量，用于确定学习率
train_dataloader = dict(
    batch_size=20, 
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
checkpoint_file = '/home/hjf/workspace/mmsegmentation/work_dirs/dcama_pretrained/pascal-5i/swin_fold0_mmlab_backbone.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file, prefix='backbone.'),
        img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        num_classes=0,
        window_size=12,
        frozen_stages=4,                        # 固定权重
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        nlayers=[2, 2, 18, 2],
        use_original_imgsize=False,

        num_classes=2,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
        ],
    )
    # decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=150),
    # auxiliary_head=dict(in_channels=512, num_classes=150)
)

# SGD optimizer, no weight decay for position embedding & layer norm
# in backbone
lr = 1e-3 / 48 * train_dataloader['batch_size'] * num_gpus  # total batch size = 48
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', 
        lr=lr,                                
        momentum=0.9,
        weight_decay=lr/10,
        nesterov=True
    ),
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'absolute_pos_embed': dict(decay_mult=0.),
    #         'relative_position_bias_table': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.)
    #     }
    # )
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
        type='ConstantLR',  # 固定学习率阶段
        factor=1.0,         # 保持学习率为初始值（base_lr）
        by_epoch=False,
        begin=1500,
        end=100000,
    )
]

train_cfg = dict(
    max_iters=100000,
    val_interval=10000
)
