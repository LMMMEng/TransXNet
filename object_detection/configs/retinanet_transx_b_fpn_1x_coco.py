_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]


model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        pretrained=True,
        type='transxnet_b',
        drop_path_rate=0.3,
        start_level=1,
    ),
    neck=dict(
        type='FPN',
        in_channels=[76, 152, 336, 672],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512)) ## AMP Training
evaluation = dict(save_best='auto')
