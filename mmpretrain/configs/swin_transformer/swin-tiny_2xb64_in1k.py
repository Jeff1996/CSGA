_base_ = [
    '../_base_/models/swin_transformer/tiny_224.py',
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# 数据集配置（单卡）
data_root = '/mnt/ssd/hjf/ImageNet'
num_gpus = 1
batch_size_pergpu = 64

train_dataloader = dict(
    batch_size=batch_size_pergpu, 
    dataset=dict(
        data_root=data_root,
        ann_file='train.txt',
        split='',
    )
)

val_dataloader = dict(
    batch_size=batch_size_pergpu, 
    dataset=dict(
        data_root=data_root,
        ann_file='test.txt',
        split='',
    )
)

test_dataloader = val_dataloader

# 学习策略配置
# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    # optimizer=dict(
    #     type='AdamW',
    #     lr=5e-4 * num_gpus * batch_size_pergpu / 512,
    #     weight_decay=0.05,
    #     eps=1e-8,
    #     betas=(0.9, 0.999)
    # ),
    # paramwise_cfg=dict(
    #     norm_decay_mult=0.0,
    #     bias_decay_mult=0.0,
    #     flat_decay_mult=0.0,
    #     custom_keys={
    #         '.absolute_pos_embed': dict(decay_mult=0.0),
    #         '.relative_position_bias_table': dict(decay_mult=0.0)
    #     }
    # ),
    clip_grad=dict(max_norm=5.0)
)


# schedule settings
optim_wrapper = dict(clip_grad=dict(max_norm=5.0))

# train, val, test setting
train_cfg = dict(
    by_epoch=True, 
    max_epochs=300, 
    val_interval=10
)

default_hooks = dict(
    # 每 100 次迭代打印一次日志。
    logger=dict(type='LoggerHook', interval=100),
    # 每隔interval个epoch保存一次权重
    checkpoint=dict(type='CheckpointHook', interval=10)
)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=num_gpus * batch_size_pergpu)
