_base_ = [
    "../_base_/datasets/nus-3d.py",
    "../_base_/default_runtime.py",
]

plugin = True
plugin_dir =  "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
unified_voxel_size = [0.8, 0.8, 1.6]
frustum_range = [-1, -1, 0.0, -1, -1, 64.0]
frustum_size = [-1, -1, 1.0]
cam_sweep_num = 1
fp16_enabled = True
unified_voxel_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=cam_sweep_num,
)

model = dict(
    type="UVTRGP",
    img_backbone=dict(
        type="MaskConvNeXt",
        arch="small",
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        norm_out=True,
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="convnextS_1kpretrained_official_style.pth",
        ),
        mae_cfg=dict(
            downsample_scale=32, downsample_dim=768, mask_ratio=0.3, learnable=False
        ),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[96, 192, 384, 768],
        out_channels=128,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    depth_head=dict(type="SimpleDepth"),
    pts_bbox_head=dict(
        type="GaussianHead",
        fp16_enabled=False,
        in_channels=128,
        unified_voxel_size=unified_voxel_size,
        unified_voxel_shape=unified_voxel_shape,
        pc_range=point_cloud_range,
        cam_nums=6,
        ray_sampler_cfg=dict(
            close_radius=3.0,
            far_radius=50.0,
            only_img_mask=False,
            only_point_mask=False,
            replace_sample=False,
            point_nsample=1024,
            point_ratio=0.99,
            pixel_interval=4,
            sky_region=0.4,
            merged_nsample=1024,
        ),
        view_cfg=dict(
            type="Uni3DViewTrans",
            frustum_range=frustum_range,
            frustum_size=frustum_size,
            num_convs=0,
            keep_sweep_dim=False,
            fp16_enabled=fp16_enabled,
        ),
         gs_cfg=dict(
            type="GSRegresser_Sample",
            voxel_size=unified_voxel_size,
            pc_range=point_cloud_range,
            voxel_shape=unified_voxel_shape,
            max_scale=0.01,
            split_dimensions=[4, 3, 1, 3],
            interpolate_cfg=dict(type="SmoothSampler", padding_mode="zeros"),
            param_decoder_cfg=dict(
                    in_dim=32, out_dim=4+3+1+3, hidden_size=32, n_blocks=5
                ),
        ),
        render_conv_cfg=dict(out_channels=32, kernel_size=3, padding=1),
        loss_cfg=dict(
            sensor_depth_truncation=0.1,
            sparse_points_sdf_supervised=False,
            weights=dict(
                depth_loss=1.0,
                rgb_loss=10.0,
                opacity_loss=0.0,
                opacity_focal_loss=10.0,
                lpips_loss=0.0,
                ssim_loss=0.0,
                occ_loss=0.0,
            ),
        ),
    ),
)

dataset_type = "NuScenesSweepDataset"
# data_root = "data/nuscenes/"
data_root = "path_to_datasets"

file_client_args = dict(backend="disk")

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="LoadCameraParam", cam_nums=6, znear=1.0, zfar=100.0,
    ),
    
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img"], meta_keys=('filename', 'ori_shape', 'img_shape', 
                            'lidar2img', 'lidar2cam',
                             'cam2img', 'pad_shape', 
                            'scale_factor',  'box_mode_3d', 
                             'sample_idx', 'pts_filename', 'cam_params',
                                'pad_before_shape','img_norm_cfg',
                             )),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type="LoadMultiViewMultiSweepImageFromFiles",
        sweep_num=cam_sweep_num,
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="LoadCameraParam", cam_nums=6, znear=1.0, zfar=100.0,
    ),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(type="CollectUnified3D", keys=["points", "img"], meta_keys=('filename', 'ori_shape', 'img_shape', 
                            'lidar2img', 'lidar2cam',
                             'cam2img', 'pad_shape', 
                            'scale_factor',  'box_mode_3d', 
                             'sample_idx', 'pts_filename', 'cam_params',
                                'pad_before_shape','img_norm_cfg',
                             )),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "nuscenes_unified_infos_train.pkl",  # please change to your own info file
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d="LiDAR",
        load_interval=1,
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
        load_interval=1,
    ),  # please change to your own info file
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_unified_infos_val.pkl",
        load_interval=1,
    ),
)  # please change to your own info file

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
total_epochs = 12
evaluation = dict(interval=total_epochs, pipeline=test_pipeline)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)


find_unused_parameters = True
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)
load_from = None
resume_from = None
# fp16 setting
fp16 = dict(loss_scale=32.0)
