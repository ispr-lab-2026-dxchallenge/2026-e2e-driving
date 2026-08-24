import argparse
import importlib
import json
import os
from collections import OrderedDict

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def parse_args():
    parser = argparse.ArgumentParser(description='ETRI challenge submission')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--frames', type=int, default=3,
                        help='clip 당 사용할 프레임 수 (1~7). 배포 기본값은 7 이지만 '
                             '실측상 3 이 L2 도 좋고 2.4배 빠르다. tools/eval_protocol.py 참조')
    return parser.parse_args()


def pick_frames(sample_ids, n):
    """clip 의 7 프레임 중 n 개를 균등하게 고른다. 현재 프레임은 반드시 포함한다.

    테스트 clip 은 STREAM_FRAMES = range(-30, 1, 5) 로 만들어져
    [-30, -25, -20, -15, -10, -5, 0] 의 7 개가 시간순으로 들어 있다
    (tools/data_converter/etri_test_converter.py).

    n=3 이면 [0, 3, 6] -> 오프셋 [-30, -15, 0] 이 된다.

    운영국 답변(2026-08-23): "과거 3s 범위 안에서 프레임 수 및 샘플링 간격은
    자유". 다만 과거를 쓰는 프레임에는 영상이 반드시 함께 들어가야 하는데,
    여기서는 프레임 전체를 넣거나 빼므로 그 조건을 만족한다.
    """
    total = len(sample_ids)
    if n >= total or n <= 0:
        return sample_ids
    if n == 1:
        return [sample_ids[-1]]
    step = (total - 1) // (n - 1)
    idx = [total - 1 - (n - 1 - k) * step for k in range(n)]
    return [sample_ids[i] for i in idx]


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.ann_file = args.ann_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.pop('map_ann_file', None)

    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    clips = OrderedDict()
    for gi, info in enumerate(dataset.data_infos):
        clips.setdefault(info['scene_token'], []).append(gi)

    submission = {}
    n_used = None
    for clip_token, sample_ids in mmcv.track_iter_progress(list(clips.items())):
        used = pick_frames(sample_ids, args.frames)
        if n_used is None:
            n_used = len(used)
            print('clip 당 %d 프레임 중 %d 개 사용' % (len(sample_ids), n_used))
        reset_stream(model.module)
        result = None
        collated = None
        for gi in used:
            collated = collate([dataset[gi]], samples_per_gpu=1)
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **collated)
        ego_fut_preds = result[0]['pts_bbox']['ego_fut_preds']
        cmd = np.array(collated['ego_fut_cmd'][0].data[0]).reshape(
            -1, ego_fut_preds.shape[0])[0]
        traj = ego_fut_preds[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
        submission[clip_token] = traj.tolist()

    mmcv.mkdir_or_exist(os.path.dirname(os.path.abspath(args.out)))
    out_dict = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        if '__flops__' in existing:
            out_dict['__flops__'] = existing['__flops__']
    out_dict.update(submission)
    with open(args.out, 'w') as f:
        json.dump(out_dict, f)
    print(f'wrote {args.out} ({len(submission)} clips, {n_used} frames/clip)')


if __name__ == '__main__':
    main()
