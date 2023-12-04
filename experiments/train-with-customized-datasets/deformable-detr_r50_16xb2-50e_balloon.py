# The new config inherits a base config to highlight the necessary modification
_base_ = "mmdet::deformable_detr/deformable-detr_r50_16xb2-50e_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=1))

# Modify dataset related settings
data_root = "data/balloon/"
metainfo = {
    "classes": ("balloon",),
    "palette": [
        (220, 20, 60),
    ],
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/annotation_coco.json",
        data_prefix=dict(img="train/"),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val/annotation_coco.json",
        data_prefix=dict(img="val/"),
    )
)
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + "val/annotation_coco.json")
test_evaluator = val_evaluator

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = "https://download.openmmlab.com/mmdetection/v3.0/deformable_detr/deformable-detr_r50_16xb2-50e_coco/deformable-detr_r50_16xb2-50e_coco_20221029_210934-6bc7d21b.pth"  # noqa

# Learning rateの調整
"""
optim_wrapper = dict(
    optimizer=dict(
        lr=0.02 / 10
    )
)
"""

# Gradient Accumulation
"""
optimizer = dict(
    accumulative_counts=4
)
"""

# Gradient Checkpoint
# gredientで別にスペルミスではない
"""
gredient_checkpoint = [
    "backbone",
]
"""
