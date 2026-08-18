"""전처리(undistort + crop + scale 0.4) 데이터용 config.

VAD_etri_tiny.py 를 상속하되 파이프라인에서 다음을 제거한다.
  UndistortMultiViewImage        - 전처리에서 이미 적용
  CropMultiViewImage             - 전처리에서 이미 적용
  RandomScaleImageMultiViewImage - 전처리에서 이미 적용

카메라 내부 파라미터는 convert_pre.py 가 pkl 에 보정해 넣는다.
"""
_base_ = ['./VAD_etri_tiny.py']

# 상속되는 최종 data dict 를 덮어써야 하므로 필요한 값을 다시 정의한다
point_cloud_range = [-30.0, -15.0, -2.0, 30.0, 15.0, 2.0]
class_names = ['Car', 'Pedestrian', 'Cyclist']
map_classes = ['lane']
map_fixed_ptsnum_per_gt_line = 20
map_eval_use_same_gt_sample_num_flag = True
bev_h_ = 100
bev_w_ = 100
queue_length = 3

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False,
    use_map=False, use_external=True)

dataset_type = 'VADCustomETRIDataset'
data_root = 'data/etri/'

# 전처리 결과 해상도 (1920x1080 crop 후 0.4배)
PRE_W, PRE_H = 768, 432

_collect_keys = [
    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'ego_his_trajs', 'ego_fut_trajs',
    'ego_fut_masks', 'ego_fut_cmd', 'ego_lcf_feat', 'gt_attr_labels'
]

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # Undistort / Crop 제거
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # RandomScale 제거
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
         with_ego=True),
    dict(type='CustomCollect3D', keys=_collect_keys),
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=True),
    dict(type='CustomObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='CustomObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(PRE_W, PRE_H),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='CustomDefaultFormatBundle3D', class_names=class_names,
                 with_label=False, with_ego=True),
            dict(type='CustomCollect3D',
                 keys=_collect_keys + ['fut_valid_flag']),
        ]),
]

# crop_size 를 실제 이미지 크기로 두어 crop_box 가 (0, 0, W, H) 가 되게 한다.
# (파이프라인에서 Crop 을 뺐으므로 쓰이지 않지만 안전장치)
_common = dict(
    type=dataset_type,
    data_root=data_root,
    classes=class_names,
    modality=input_modality,
    bev_size=(bev_h_, bev_w_),
    pc_range=point_cloud_range,
    map_classes=map_classes,
    map_fixed_ptsnum_per_line=map_fixed_ptsnum_per_gt_line,
    map_eval_use_same_gt_sample_num_flag=map_eval_use_same_gt_sample_num_flag,
    crop_size=(PRE_W, PRE_H),
    custom_eval_version='vad_nusc_detection_cvpr_2019',
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        ann_file=data_root + 'vad_etri_pre_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        use_valid_flag=True,
        queue_length=queue_length,
        box_type_3d='LiDAR',
        **_common),
    val=dict(
        ann_file=data_root + 'vad_etri_pre_infos_temporal_train.pkl',
        pipeline=test_pipeline,
        samples_per_gpu=1,
        use_pkl_result=True,
        **_common),
    test=dict(
        ann_file=data_root + 'vad_etri_pre_infos_temporal_train.pkl',
        pipeline=test_pipeline,
        samples_per_gpu=1,
        use_pkl_result=True,
        **_common),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)
