_base_ = "../../projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_1x_coco.py"

num_dec_layer = 6
loss_lambda = 2.0
num_classes = 1


data_root = "data/balloon/"  # defined in common/ssj_scp_270k_coco-instance.py
metainfo = {
    "classes": ("balloon",),
    "palette": [
        (220, 20, 60),
    ],
}
train_dataloader = dict(
    batch_size=1,  # defined in co_dino_5scale_swin_l_16xb1_1x_coco.py
    num_workers=1,  # defined in co_dino_5scale_swin_l_16xb1_1x_coco.py
    dataset=dict(
        data_root=data_root,  # defined in co_dino_5scale_r50_8xb2_1x_coco.py
        metainfo=metainfo,
        # defined in co_dino_5scale_r50_8xb2_1x.py
        ann_file="train/annotation_coco.json",
        # defined in common/ssj_scp_270k_coco-instance.py
        data_prefix=dict(img="train/")
    ),
)
val_dataloader = dict(
    batch_size=1,  # defined in common/ssj_270k_coco-instance.py
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val/annotation_coco.json",
        data_prefix=dict(img="val/"),
    ),
)
test_dataloader = val_dataloader

model = dict(
    backbone=dict(
        # with_cp: defined in co_dino_5scale_swin_l_16xb1_1x_coco.py
        with_cp=True  # whether checkpoints are used or not.
    ),
    query_head=dict(
        num_classes=num_classes,
        transformer=dict(
            encoder=dict(
                # num_layers=6 defined in co_dino_5scale_swin_l_16xb1_1x_coco.py
                with_cp=6  # number of layers that use checkpoint.
            )
        )
    ),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ],
)

# defined in common/ssj_270k_coco-instance.py
val_evaluator = dict(
    # defined in common/ssj_270k_coco-instance.py
    ann_file=data_root + "val/annotation_coco.json",
    metric="bbox",  # defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
)
test_evaluator = val_evaluator

# use the pre-trained CO-DETR model
# defined in _base_/default_runtime.py
load_from = "https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_1x_coco-27c13da4.pth"  # noqa

# Hook config
default_hooks = dict(
    # defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
    checkpoint=dict(
        _delete_=True,
        _scope_="mmdet",
        by_epoch=True,
        interval=1,
        save_best="coco/bbox_mAP",  # the best bbox_mAP of each val steps
        type="CheckpointHook"
    ),
)
env_cfg = dict(
    cudnn_benchmark=True,
)
# defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
log_processor = dict(
    by_epoch=True
)

# auto_scale_lr is probably enable by default.
# defined in co_dino_5scale_r50_lsj_8xb2_1x_coco.py
auto_scale_lr = dict(enable=False, base_batch_size=16)
