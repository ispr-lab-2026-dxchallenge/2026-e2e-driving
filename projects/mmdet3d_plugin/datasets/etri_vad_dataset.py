import os
import json
import copy
import random

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from shapely.geometry import LineString, box

from .nuscenes_vad_dataset import (
    VADCustomNuScenesDataset,
    LiDARInstanceLines,
    v1CustomDetectionConfig,
)


@DATASETS.register_module()
class VADCustomETRIDataset(VADCustomNuScenesDataset):
    """ETRI challenge dataset for VAD.
    """
    MAPCLASSES = ('lane',)

    def __init__(
        self,
        queue_length=4,
        bev_size=(200, 200),
        overlap_test=False,
        with_attr=True,
        fut_ts=6,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        map_classes=None,
        map_ann_file=None,
        map_fixed_ptsnum_per_line=-1,
        map_eval_use_same_gt_sample_num_flag=False,
        padding_value=-10000,
        use_pkl_result=False,
        custom_eval_version='vad_nusc_detection_cvpr_2019',
        crop_size=(1920, 1080),
        crop_keep_top=('camera_front_left', 'camera_front_right',
                       'camera_rear_left', 'camera_rear_right'),
        sample_interval=5,
        map_pts_dim=2,
        *args,
        **kwargs
    ):
        NuScenesDataset.__init__(self, *args, **kwargs)
        self.crop_size = crop_size
        self.crop_keep_top = crop_keep_top
        self.sample_interval = sample_interval
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.with_attr = with_attr
        self.fut_ts = fut_ts
        self.use_pkl_result = use_pkl_result

        self.custom_eval_version = custom_eval_version
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(this_dir, '%s.json' % self.custom_eval_version)
        assert os.path.exists(cfg_path), \
            'Requested unknown configuration {}'.format(self.custom_eval_version)
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        self.custom_eval_detection_configs = v1CustomDetectionConfig.deserialize(data)

        # Map lanes are loaded from the info PKL metadata.  The separate file
        # used by the upstream evaluator is only a generated GT cache.
        if map_ann_file is None:
            ann_file = kwargs.get('ann_file')
            if ann_file is not None:
                stem, _ = os.path.splitext(ann_file)
                map_ann_file = stem + '_map_gt.json'
        self.map_ann_file = map_ann_file
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = map_fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = map_eval_use_same_gt_sample_num_flag
        self.map_lanes = self.metadata.get('map_lanes', {})
        self.map_pts_dim = map_pts_dim
        self.is_vis_on_test = True

    def prepare_train_data(self, index):
        data_queue = []
        prev_indexs_list = list(
            range(index - self.queue_length * self.sample_interval, index,
                  self.sample_interval))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict['frame_idx']
        scene_token = input_dict['scene_token']
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and \
                ((example is None or ~(example['gt_labels_3d']._data != -1).any()) or
                 (example is None or ~(example['map_gt_labels_3d']._data != -1).any())):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict['frame_idx'] < frame_idx and input_dict['scene_token'] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = self.vectormap_pipeline(example, input_dict)
                if self.filter_empty_gt and \
                        (example is None or ~(example['gt_labels_3d']._data != -1).any()) and \
                        (example is None or ~(example['map_gt_labels_3d']._data != -1).any()):
                    return None
                frame_idx = input_dict['frame_idx']
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        if input_dict is None:
            return None
        info = self.data_infos[index]
        input_dict['timestamp'] = info['timestamp'] / 1e3
        if self.modality['use_camera']:
            cams = list(info['cams'].items())
            input_dict['cam_intrinsic_raw'] = [c['cam_intrinsic_raw'] for _, c in cams]
            input_dict['cam_intrinsic_undist'] = [c['cam_intrinsic'] for _, c in cams]
            input_dict['distortion'] = [c['distortion'] for _, c in cams]
            input_dict['is_fisheye'] = [c['is_fisheye'] for _, c in cams]
            crop_box = []
            for name, c in cams:
                ox = (c['image_width'] - self.crop_size[0]) // 2
                if name in self.crop_keep_top:
                    oy = 0
                else:
                    oy = c['image_height'] - self.crop_size[1]
                crop_box.append((ox, oy, self.crop_size[0], self.crop_size[1]))
            input_dict['crop_box'] = crop_box
        return input_dict

    def vectormap_pipeline(self, example, input_dict):
        scene_token = input_dict['scene_token']
        lanes = self.map_lanes.get(scene_token, [])

        rotation = Quaternion(input_dict['ego2global_rotation']).rotation_matrix
        translation = np.array(input_dict['ego2global_translation'])
        patch = box(self.pc_range[0], self.pc_range[1],
                    self.pc_range[3], self.pc_range[4])

        instances = []
        for lane in lanes:
            pts_ego = (lane[:, :3] - translation) @ rotation
            line = LineString(pts_ego[:, :self.map_pts_dim])
            cropped = line.intersection(patch)
            if cropped.is_empty:
                continue
            if cropped.geom_type == 'MultiLineString':
                cropped = max(cropped.geoms, key=lambda g: g.length)
            if cropped.geom_type != 'LineString' or cropped.length == 0:
                continue
            instances.append(cropped)

        gt_labels = [0] * len(instances)
        gt_instance = LiDARInstanceLines(
            instances, sample_dist=1, num_samples=250, padding=False,
            fixed_num=self.fixed_num, padding_value=self.padding_value,
            patch_size=self.patch_size)

        example['map_gt_labels_3d'] = DC(to_tensor(gt_labels), cpu_only=False)
        example['map_gt_bboxes_3d'] = DC(gt_instance, cpu_only=True)
        return example
