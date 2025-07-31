# dataset settings
dataset_type = 'PASCAL5iDataset'
data_root = '/home/hjf/workspace/00_datasets/PASCAL-5i-5953-Filtered'
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='CustomLabelMap', label_map={255: 1}),
                                                # 此方法为自定义方法, 在mmseg/datasets/transforms/transforms.py中实现
                                                # 功能为按照label_map字典中的规则对标签值进行映射
    dict(
        type='RandomResize',
        scale=(1536, 384),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),               # 标签文件也一并进行缩放处理
    dict(type='CustomLabelMap', label_map={255: 1}),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
                                                # 保持图片横纵比, 长边缩放至384
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        folds=[1,2,3],                          # 需要使用的fold, 例如[1, 2, 3]
        mode='train',                           # 使用fold*_train.txt还是fold*_val.txt
        shot=1,                                 # k-shot
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    # sampler=dict(type='DefaultSampler', shuffle=False),
    sampler=dict(
        type='RandomSampler',                   # 这个采样器在以下文件中定义: /home/hjf/.conda/envs/hjf_mmlab/lib/python3.9/site-packages/mmengine/dataset/sampler.py
        num_samples=1000,                       # 采样 1000 个 task
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        folds=[0],                              # 需要使用的fold, 例如[0]
        mode='val',                             # 使用fold*_train.txt还是fold*_val.txt
        shot=1,                                 # k-shot
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    # type='IoUMetric', iou_metrics=['mIoU']
    type='IoUMetricFSS', iou_metrics=['mIoU', 'FBIoU']
)
test_evaluator = val_evaluator
