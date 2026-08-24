"""채점과 동일한 프로토콜로 val L2 와 T_infer 를 동시에 잰다.

왜 필요한가
-----------
1) 프로토콜 불일치
   기존 ablate_image.py 는 val 시나리오의 300 프레임을 연속으로 흘려보낸다.
   40초 내내 BEV 가 누적되므로 실제 채점보다 유리하다.
   실제 채점은 clip 당 **7 프레임(과거 3초, 0.5초 간격)** 만 준다.
   (etri_test_converter.py: STREAM_FRAMES = range(-30, 1, 5))

2) T_infer 정의 (2026-08-23 운영국 답변)
   "마지막 프레임이 아니라 초기화 이후 과거 시점 처리부터 마지막 궤적 출력까지
    수행되는 모든 model forward 의 누적 시간"
   즉 T_infer = 단일 프레임 시간 x 사용한 프레임 수.
   기준선 100ms 를 넘으면 스코어에 페널티가 붙는다.

3) 프레임 수는 우리가 고를 수 있다 (같은 답변 3번)
   "과거 3s 범위 안에서 프레임 수 및 샘플링 간격은 자유"
   재학습 없이 추론 시점에만 바꾸는 값이므로, 학습된 체크포인트로 바로 훑을 수 있다.

사용법
------
  python tools/eval_protocol.py <config> <checkpoint> --frames 7,4,3,1

주의: GPU 를 쓴다. 학습이 도는 머신에서 돌리지 말 것.
"""
import argparse
import importlib
import time

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30          # 과거 3초 = 30 프레임 (10Hz)


def parse_args():
    p = argparse.ArgumentParser(description='채점 프로토콜 기준 val 평가 + T_infer')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-file', default=None, help='기본값은 config 의 val split')
    p.add_argument('--frames', default='7,4,3,1',
                   help='시험할 프레임 수 목록. 과거 3초를 균등 분할한다. 7 이 배포 기본값')
    p.add_argument('--offsets', default=None,
                   help='오프셋을 직접 지정한다. 세미콜론으로 여러 조합. '
                        '예: "-10,-5,0;-6,-3,0". 지정하면 --frames 는 무시된다. '
                        '3초를 다 쓰지 않고 최근 프레임만 쓰는 조합을 시험할 때 쓴다')
    p.add_argument('--scenarios', type=int, default=10, help='0 이면 전부')
    p.add_argument('--eval-stride', type=int, default=20,
                   help='시나리오 안에서 몇 프레임마다 평가할지')
    return p.parse_args()


def frame_offsets(n):
    """과거 3초를 n 개로 균등 분할. 항상 현재 프레임(0)을 포함한다."""
    if n <= 1:
        return [0]
    step = HIS_FRAMES // (n - 1)
    return [-(n - 1 - k) * step for k in range(n)]


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def unwrap(collated, key):
    v = collated[key]
    if isinstance(v, list):
        v = v[0]
    return v.data[0]


def official_l2(errs):
    e = np.asarray(errs)
    a1, a2, a3 = e[:, :2].mean(), e[:, :4].mean(), e[:, :6].mean()
    return a1, a2, a3, (a1 + a2 + a3) / 3


def run(model, dataset, groups, offs, eval_stride):
    label = len(offs)
    errs, times = [], []
    for si, (_tok, ids) in enumerate(groups):
        # 과거 3초가 확보되는 지점부터 평가한다
        points = range(HIS_FRAMES, len(ids), eval_stride)
        for cur in points:
            # 데이터 로딩은 타이머 밖에서 먼저 끝낸다.
            # 운영국 정의는 "모든 model forward 의 누적 시간"이라
            # JPEG 디코딩·정규화는 T_infer 에 포함되지 않는다.
            batches = [collate([dataset[ids[cur + o]]], samples_per_gpu=1) for o in offs]

            reset_stream(model.module)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = None
            for collated in batches:
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **collated)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)   # ms
            collated = batches[-1]

            if not bool(np.asarray(unwrap(collated, 'fut_valid_flag')).reshape(-1)[0]):
                times.pop()
                continue
            pred = result[0]['pts_bbox']['ego_fut_preds']
            cmd = np.asarray(unwrap(collated, 'ego_fut_cmd')).reshape(-1, pred.shape[0])[0]
            pred = pred[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
            gt = np.asarray(unwrap(collated, 'ego_fut_trajs')).reshape(-1, 2).cumsum(0)
            errs.append(np.linalg.norm(pred - gt, axis=1))
        print('    [%d프레임] 시나리오 %d/%d  평가점 %d'
              % (label, si + 1, len(groups), len(errs)), flush=True)
    return np.array(errs), np.array(times)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    if args.ann_file:
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

    grouped = {}
    for gi, info in enumerate(dataset.data_infos):
        grouped.setdefault(info['scene_token'], []).append(gi)
    groups = list(grouped.items())
    if args.scenarios > 0:
        groups = groups[:args.scenarios]
    print('시나리오 %d개 | 평가 간격 %d프레임 | %s'
          % (len(groups), args.eval_stride, args.checkpoint))

    # GPU 클럭이 올라오기 전 첫 몇 회는 느리다. 버린다.
    print('워밍업...', flush=True)
    reset_stream(model.module)
    for k in range(5):
        with torch.no_grad():
            model(return_loss=False, rescale=True,
                  **collate([dataset[groups[0][1][k]]], samples_per_gpu=1))
    torch.cuda.synchronize()

    if args.offsets:
        combos = [[int(x) for x in grp.split(',')] for grp in args.offsets.split(';')]
    else:
        combos = [frame_offsets(int(x)) for x in args.frames.split(',')]

    rows = []
    for offs in combos:
        n = len(offs)
        print('--- %d 프레임  offsets=%s ---' % (n, offs), flush=True)
        errs, times = run(model, dataset, groups, offs, args.eval_stride)
        if len(errs) == 0:
            print('  유효 평가점 없음'); continue
        a1, a2, a3, avg = official_l2(errs)
        rows.append((offs, a1, a2, a3, avg, times.mean(), np.percentile(times, 95),
                     times.mean() / n))

    print()
    print('=' * 92)
    print('%-26s %7s %7s %7s %9s | %9s %9s'
          % ('offsets', '1s', '2s', '3s', 'L2_avg', 'T_infer', '프레임당'))
    print('-' * 92)
    for offs, a1, a2, a3, avg, tm, p95, per in rows:
        flag = '  <-100ms초과' if tm > 100 else ''
        print('%-26s %7.3f %7.3f %7.3f %9.4f | %7.1fms %7.1fms%s'
              % (','.join(str(o) for o in offs), a1, a2, a3, avg, tm, per, flag))
    print('=' * 92)
    print()
    print('T_infer 는 초기화 이후 모든 forward 의 누적 시간이다 (2026-08-23 운영국 답변).')
    print('100ms 를 넘으면 스코어에 페널티가 붙는다.')
    print('단, 채점 하드웨어는 이 머신과 다르다. 상대적인 경향만 신뢰할 것.')


if __name__ == '__main__':
    main()
