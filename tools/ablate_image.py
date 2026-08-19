"""영상이 궤적 예측에 실제로 기여하는지 측정한다.

왜 필요한가
-----------
챌린지 추론 규정 §3 공통원칙은 "성능의 실질적 출처"를 기준으로 판단하며,
영상이 결과에 실질 기여하지 않으면 신경망 형태여도 금지한다고 명시한다.

그런데 자차 상태(ego status)를 플래너에 넣으면 모델이 영상을 무시하고
운동학적 외삽만 학습하는 경향이 알려져 있다
("Is Ego Status All You Need for Open-Loop End-to-End Autonomous Driving?", CVPR 2024).

이 스크립트는 같은 체크포인트에 영상만 망가뜨려 넣고 L2 가 얼마나 나빠지는지 잰다.
성능이 거의 안 떨어지면 그 모델은 영상을 안 보고 있다는 뜻이므로 제출하면 안 된다.

측정 방식
---------
공식 지표와 동일하다 (VAD.py:641 과 같은 정의).
  1초 L2 = 앞 2개 waypoint 의 평균 변위(ADE)
  2초 L2 = 앞 4개
  3초 L2 = 6개 전부
  최종   = 세 값의 평균

모드
----
  normal   원본 영상
  zero     영상을 0 으로 채움        (영상 신호 완전 제거)
  shuffle  다른 시나리오의 영상으로 교체 (영상은 유효하나 현재 장면과 무관)

zero 는 정규화 통계까지 깨뜨리므로 과하게 나빠질 수 있다.
shuffle 이 "영상을 보고 있는가"에 더 공정한 질문이다. 둘 다 본다.

사용법
------
  python tools/ablate_image.py <config> <checkpoint> [--scenarios 10]

주의: GPU 를 쓴다. 같은 GPU 에서 학습이 돌고 있으면 VRAM 이 모자라 죽거나
      학습을 느리게 만든다. 학습이 없는 머신에서 돌릴 것.
"""
import argparse
import importlib

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def parse_args():
    p = argparse.ArgumentParser(description='영상 기여도 ablation')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-file', default=None,
                   help='기본값은 config 의 data.test.ann_file (= val split)')
    p.add_argument('--scenarios', type=int, default=10,
                   help='평가할 시나리오 수. 0 이면 전부. 시나리오당 300 프레임이라 '
                        '전부 돌리면 모드당 1시간이 넘는다')
    p.add_argument('--modes', default='normal,zero,shuffle')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def reset_stream(model):
    model.prev_frame_info = {
        'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def get_img_tensor(collated):
    """collate 결과에서 이미지 텐서를 꺼낸다.

    test 파이프라인은 값을 리스트로 한 번 더 감싸므로 두 형태를 모두 받는다.
    (etri_test_submit.py 가 ego_fut_cmd 를 collated[k][0].data[0] 로 읽는 것과 같은 구조)
    """
    v = collated['img']
    if isinstance(v, list):
        v = v[0]
    return v.data[0]


def unwrap(collated, key):
    v = collated[key]
    if isinstance(v, list):
        v = v[0]
    return v.data[0]


def official_l2(errs):
    """errs: (N, 6) 각 waypoint 별 유클리드 거리 -> 공식 지표."""
    e = np.asarray(errs)
    a1 = e[:, :2].mean()
    a2 = e[:, :4].mean()
    a3 = e[:, :6].mean()
    return a1, a2, a3, (a1 + a2 + a3) / 3


def run_mode(model, dataset, clips, mode, donor_imgs, rng):
    errs = []
    skipped = 0
    # 교란이 실제로 모델 입력에 반영됐는지 확인하기 위한 통계.
    # zero 인데 0 이 아니거나 normal 과 같으면 교란이 안 먹은 것이다.
    img_stats = []
    for ci, (_token, sample_ids) in enumerate(clips):
        reset_stream(model.module)
        for gi in sample_ids:
            collated = collate([dataset[gi]], samples_per_gpu=1)

            img = get_img_tensor(collated)
            if mode == 'zero':
                img.zero_()
            elif mode == 'shuffle':
                donor = donor_imgs[rng.integers(len(donor_imgs))]
                if donor.shape == img.shape:
                    img.copy_(donor)

            if len(img_stats) < 20:
                img_stats.append(float(img.abs().mean()))

            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **collated)

            valid = unwrap(collated, 'fut_valid_flag')
            if not bool(np.asarray(valid).reshape(-1)[0]):
                skipped += 1
                continue

            pred = result[0]['pts_bbox']['ego_fut_preds']          # [3, 6, 2] 변위
            cmd = np.asarray(unwrap(collated, 'ego_fut_cmd')).reshape(-1, pred.shape[0])[0]
            pred = pred[int(cmd.argmax())].cpu().double().cumsum(0).numpy()

            gt = np.asarray(unwrap(collated, 'ego_fut_trajs')).reshape(-1, 2)
            gt = gt.cumsum(0)

            errs.append(np.linalg.norm(pred - gt, axis=1))
        print('    [%s] 시나리오 %d/%d  누적 샘플 %d'
              % (mode, ci + 1, len(clips), len(errs)), flush=True)
    return np.array(errs), skipped, float(np.mean(img_stats)) if img_stats else float('nan')


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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
    clips = list(grouped.items())
    if args.scenarios > 0:
        clips = clips[:args.scenarios]
    n_frames = sum(len(s) for _, s in clips)
    print('시나리오 %d개 / 프레임 %d개 / 체크포인트 %s'
          % (len(clips), n_frames, args.checkpoint))

    # shuffle 용 기증 영상: 평가 대상과 겹치지 않는 뒤쪽 시나리오에서 뽑는다
    donor_imgs = []
    if 'shuffle' in args.modes:
        pool = list(grouped.items())[len(clips):] or list(grouped.items())
        for _token, sids in pool[:5]:
            for gi in sids[::60]:
                donor_imgs.append(get_img_tensor(
                    collate([dataset[gi]], samples_per_gpu=1)).clone())
        print('shuffle 용 기증 영상 %d장 확보' % len(donor_imgs))

    results = {}
    img_means = {}
    for mode in args.modes.split(','):
        mode = mode.strip()
        if mode == 'shuffle' and not donor_imgs:
            continue
        print('--- %s ---' % mode, flush=True)
        errs, skipped, imean = run_mode(model, dataset, clips, mode, donor_imgs, rng)
        if len(errs) == 0:
            print('  유효 샘플이 없다 (fut_valid_flag 전부 False)')
            continue
        results[mode] = official_l2(errs)
        img_means[mode] = imean
        print('  유효 %d / 제외 %d / 입력 영상 평균 |x| = %.4f' % (len(errs), skipped, imean))

    print()
    print('=' * 68)
    print('%-10s %8s %8s %8s %10s %10s %10s'
          % ('mode', '1s', '2s', '3s', 'L2_avg', '악화', '입력|x|'))
    print('-' * 80)
    base = results.get('normal', (None,) * 4)[3]
    for mode, (a1, a2, a3, avg) in results.items():
        ratio = ('%9.2f배' % (avg / base)) if base else '        -'
        print('%-10s %8.3f %8.3f %8.3f %10.4f %s %10.4f'
              % (mode, a1, a2, a3, avg, ratio, img_means.get(mode, float('nan'))))
    print('=' * 80)

    # 교란이 진짜 먹었는지부터 확인한다. 이게 깨지면 아래 결론은 전부 무의미하다.
    ok = True
    if 'zero' in img_means and img_means['zero'] > 1e-6:
        print('!! zero 모드인데 입력 영상이 0 이 아니다. 교란이 반영되지 않았다.')
        ok = False
    if ('shuffle' in img_means and 'normal' in img_means
            and abs(img_means['shuffle'] - img_means['normal']) < 1e-9):
        print('!! shuffle 모드의 입력이 normal 과 동일하다. 교란이 반영되지 않았다.')
        ok = False
    if not ok:
        print('!! 스크립트 문제다. 아래 결론을 믿지 말 것.')
        return

    if base and len(results) > 1:
        worst = max(v[3] for k, v in results.items() if k != 'normal')
        print()
        print('영상을 망가뜨렸을 때 L2 가 %.2f배로 악화됐다.' % (worst / base))
        if base > 5.0:
            # 학습된 모델이면 L2 는 보통 1~2 m 대다. 그보다 훨씬 크면 미학습이고,
            # 이때는 출력이 입력과 무관한 상수에 가까워 배율 자체가 의미를 잃는다.
            print('L2 가 %.1f m 로 지나치게 크다. 학습이 안 된 모델이다.' % base)
            print('미학습 모델은 입력과 무관하게 비슷한 값을 내므로 악화 배율은 의미가 없다.')
            print('판정을 보류한다. 학습된 체크포인트로 다시 돌릴 것.')
        elif worst / base < 1.2:
            print('악화가 20% 미만이다. 모델이 영상을 사실상 안 보고 있다는 뜻이며,')
            print('챌린지 규정 §3 공통원칙 위반 소지가 크다. 제출하면 안 된다.')
        else:
            print('영상이 결과에 실질적으로 기여하고 있다.')


if __name__ == '__main__':
    main()
