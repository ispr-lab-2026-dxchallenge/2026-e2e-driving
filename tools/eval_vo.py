"""VO 헤드 정확도 평가 (관문 1). 제출 프로토콜 창(-10,-5,0)에서 s_hat 을 뽑아 라벨과 비교한다.
  python tools/eval_vo.py <config> <ckpt> [--scenarios 5]
라벨: 변위 = ego_his_trajs 마지막 원소(-5 -> 0, m), yaw 변화 = 창 안 -5 -> 0 의 can_bus[-1] 차(deg).
출력: 속도 |v| RMSE(m/s), 방향별 RMSE, yaw RMSE, 평균 예측기 대비 비율, 영상 0 일 때의 RMSE(붕괴 확인).
"""
import argparse
import copy
import importlib

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30
SCALE = np.array([5.0, 1.0, 0.2])


def reset(model):
    model.module.prev_frame_info = {'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}
    model.module.pts_bbox_head._vo_stash = None


def hist_step(model, collated, zero_img):
    m = model.module
    data = scatter(collated, [torch.cuda.current_device()])[0]
    meta = data['img_metas'][0][0]; img = data['img'][0]
    if zero_img: img.zero_()
    if meta['scene_token'] != m.prev_frame_info['scene_token']:
        m.prev_frame_info['prev_bev'] = None
    m.prev_frame_info['scene_token'] = meta['scene_token']
    tmp_pos = copy.deepcopy(meta['can_bus'][:3]); tmp_angle = copy.deepcopy(meta['can_bus'][-1])
    if m.prev_frame_info['prev_bev'] is not None:
        meta['can_bus'][:3] -= m.prev_frame_info['prev_pos']
        meta['can_bus'][-1] = (meta['can_bus'][-1] - m.prev_frame_info['prev_angle'] + 180) % 360 - 180
    else:
        meta['can_bus'][:3] = 0; meta['can_bus'][-1] = 0
    img_feats = m.extract_feat(img=img, img_metas=[meta])
    m.prev_frame_info['prev_bev'] = m.pts_bbox_head(img_feats, [meta], m.prev_frame_info['prev_bev'], only_bev=True)
    m.prev_frame_info['prev_pos'] = tmp_pos; m.prev_frame_info['prev_angle'] = tmp_angle
    return tmp_angle


def unwrap(c, k):
    v = c[k]; v = v[0] if isinstance(v, list) else v
    return v.data[0]


def main():
    p = argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('checkpoint')
    p.add_argument('--scenarios', type=int, default=5); p.add_argument('--eval-stride', type=int, default=10)
    a = p.parse_args()
    cfg = Config.fromfile(a.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None); cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0]); model.eval()
    head = model.module.pts_bbox_head
    box = {}
    orig = head._vo_forward
    def hooked(*args, **kw):
        s, v = orig(*args, **kw); box['s'] = s.detach().cpu().numpy()[0]; return s, v
    head._vo_forward = hooked

    grouped = {}
    for gi, info in enumerate(dataset.data_infos):
        grouped.setdefault(info['scene_token'], []).append(gi)
    groups = list(grouped.values())
    if a.scenarios > 0: groups = groups[:a.scenarios]

    preds, targets = {False: [], True: []}, []
    for zero in (False, True):
        for ids in groups:
            for cur in range(HIS_FRAMES, len(ids), a.eval_stride):
                win = [ids[cur - 10], ids[cur - 5], ids[cur]]
                batches = [collate([dataset[i]], samples_per_gpu=1) for i in win]
                reset(model)
                with torch.no_grad():
                    angs = [hist_step(model, b, zero) for b in batches[:-1]]
                    c = copy.deepcopy(batches[-1])
                    if zero:
                        (c['img'][0] if isinstance(c['img'], list) else c['img']).data[0].zero_()
                    model(return_loss=False, rescale=True, **c)
                preds[zero].append(box['s'] * SCALE)
                if not zero:
                    his = np.asarray(unwrap(batches[-1], 'ego_his_trajs')).reshape(-1, 2)[-1]
                    meta = (batches[-1]['img_metas'][0] if isinstance(batches[-1]['img_metas'], list)
                            else batches[-1]['img_metas']).data[0][0]
                    dyaw = (float(meta['can_bus'][-1]) - angs[-1] + 180) % 360 - 180
                    targets.append(np.array([his[0], his[1], np.deg2rad(dyaw)]))
    P0, P1, T = np.array(preds[False]), np.array(preds[True]), np.array(targets)
    n = min(len(P0), len(T)); P0, P1, T = P0[:n], P1[:n], T[:n]
    def rmse(P): return np.sqrt(((P - T) ** 2).mean(0))
    r0, r1, rm = rmse(P0), rmse(P1), rmse(np.tile(T.mean(0), (n, 1)))
    v0 = np.sqrt(((np.linalg.norm(P0[:, :2], axis=1) - np.linalg.norm(T[:, :2], axis=1)) / 0.5) ** 2).mean() ** 0.5
    print('평가점 %d | 라벨 |v| 평균 %.2f m/s' % (n, (np.linalg.norm(T[:, :2], axis=1) / 0.5).mean()))
    print('RMSE [dx m, dy m, dyaw rad] : 모델 %s | 영상0 %s | 평균예측기 %s' % (np.round(r0, 3), np.round(r1, 3), np.round(rm, 3)))
    print('속력 |v| RMSE %.3f m/s (관문: < 0.6) | 모델/평균예측기 dx 비율 %.2f (관문: <= 0.5) | 영상0/평균예측기 %.2f (붕괴 확인: ~1)'
          % (v0, r0[0] / max(rm[0], 1e-6), r1[0] / max(rm[0], 1e-6)))
    print('yaw RMSE %.2f deg/0.5s (관문: < 1.0)' % np.rad2deg(r0[2]))


if __name__ == '__main__':
    main()
