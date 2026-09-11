"""제출 프로토콜 그대로(3프레임 창, history-BEV, fp16)에서 영상·목표점 기여도를 잰다.

ablate_image.py 는 시나리오 300 프레임을 연속 full forward 해서 이력 BEV 가 길게 누적되는
조건이라 제출(3프레임 창)과 다르다. 이 스크립트는 eval_protocol.py 의 평가점·창·이력
처리(history_bev_step)를 그대로 쓰고, 창 안의 모든 프레임에 교란을 건다.

모드
----
  normal    원본
  zero      창 안 3프레임 영상 전부 0                      (영상 제거)
  shuffle   창 안 3프레임 영상을 다른 시나리오 영상으로 교체  (영상 유효, 장면 무관)
  goaloff   ego_lcf_feat 를 11차원으로 잘라 목표점 조건을 끔  (학습 시 condition dropout 과 동일 경로)
  goalshuf  목표점([11:13])을 다른 샘플의 목표점으로 교체

판정: 영상 교란(zero/shuffle 중 최악) 이 normal 대비 1.2배 이상이어야 한다(팀 내규).
goaloff/goalshuf 는 "목표점이 보조 입력" 임을 보여주는 참고 수치(악화폭이 영상보다 작아야 자연스럽다).

  python tools/ablate_window.py <config> <checkpoint> [--scenarios 0] [--offsets=-10,-5,0] [--fp16]
"""
import argparse
import contextlib
import importlib

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

HIS_FRAMES = 30   # 과거 3초 (10Hz)


def parse_args():
    p = argparse.ArgumentParser(description='제출 프로토콜(3프레임 창) 영상·목표점 ablation')
    p.add_argument('config')
    p.add_argument('checkpoint')
    p.add_argument('--ann-file', default=None, help='기본값은 config 의 val split')
    p.add_argument('--scenarios', type=int, default=0, help='0 이면 전부 (39)')
    p.add_argument('--eval-stride', type=int, default=10)
    p.add_argument('--offsets', default='-10,-5,0', help='제출과 동일: -10,-5,0')
    p.add_argument('--modes', default='normal,zero,shuffle,goaloff,goalshuf')
    p.add_argument('--fp16', action='store_true', default=True)
    p.add_argument('--fp32', dest='fp16', action='store_false')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def history_bev_step(model, collated):
    """eval_protocol.py / etri_test_submit.py 와 동일한 이력 프레임 처리."""
    import copy as _copy
    from mmcv.parallel import scatter
    m = model.module
    dev = torch.cuda.current_device()
    data = scatter(collated, [dev])[0]
    meta = data['img_metas'][0][0]
    img = data['img'][0]
    if meta['scene_token'] != m.prev_frame_info['scene_token']:
        m.prev_frame_info['prev_bev'] = None
    m.prev_frame_info['scene_token'] = meta['scene_token']
    tmp_pos = _copy.deepcopy(meta['can_bus'][:3])
    tmp_angle = _copy.deepcopy(meta['can_bus'][-1])
    if m.prev_frame_info['prev_bev'] is not None:
        meta['can_bus'][:3] -= m.prev_frame_info['prev_pos']
        meta['can_bus'][-1] = (meta['can_bus'][-1] -
                               m.prev_frame_info['prev_angle'] + 180) % 360 - 180
    else:
        meta['can_bus'][:3] = 0
        meta['can_bus'][-1] = 0
    img_feats = m.extract_feat(img=img, img_metas=[meta])
    prev_bev = m.pts_bbox_head(img_feats, [meta], m.prev_frame_info['prev_bev'], only_bev=True)
    m.prev_frame_info['prev_bev'] = prev_bev
    m.prev_frame_info['prev_pos'] = tmp_pos
    m.prev_frame_info['prev_angle'] = tmp_angle


def reset_stream(model):
    model.prev_frame_info = {'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}


def dc(collated, key):
    """collate 결과의 DataContainer (test 파이프라인은 리스트로 한 번 더 감싼다)."""
    v = collated[key]
    return v[0] if isinstance(v, list) else v


def unwrap(collated, key):
    return dc(collated, key).data[0]


def official_l2(errs):
    e = np.asarray(errs)
    a1, a2, a3 = e[:, :2].mean(), e[:, :4].mean(), e[:, :6].mean()
    return a1, a2, a3, (a1 + a2 + a3) / 3


def perturb(collated, mode, donors, rng, stats):
    """창 안 프레임 1개에 교란을 건다 (in-place). stats 에 입력 통계를 남긴다."""
    img = unwrap(collated, 'img')
    if mode == 'zero':
        img.zero_()
    elif mode == 'shuffle':
        d = donors['img'][rng.integers(len(donors['img']))]
        if d.shape == img.shape:
            img.copy_(d)
    elif mode == 'goaloff':
        c = dc(collated, 'ego_lcf_feat')
        c.data[0] = c.data[0][..., :11].contiguous()      # 13 -> 11 차원: head 가드가 조건을 끈다
    elif mode == 'goalshuf':
        lcf = unwrap(collated, 'ego_lcf_feat')
        d = donors['goal'][rng.integers(len(donors['goal']))]
        lcf[..., 11:13] = d
    if len(stats['img']) < 30:
        stats['img'].append(float(img.abs().mean()))
    lcf = unwrap(collated, 'ego_lcf_feat')
    stats['lcf_dim'] = int(lcf.shape[-1])
    if lcf.shape[-1] >= 13 and len(stats['goal']) < 30:
        stats['goal'].append(float(lcf[..., 11:13].abs().mean()))


def run_mode(model, dataset, groups, offs, stride, mode, donors, rng, fp16):
    amp = (lambda: torch.autocast('cuda', dtype=torch.float16)) if fp16 else contextlib.nullcontext
    errs, skipped = [], 0
    stats = {'img': [], 'goal': [], 'lcf_dim': -1}
    for si, (_tok, ids) in enumerate(groups):
        for cur in range(HIS_FRAMES, len(ids), stride):
            batches = [collate([dataset[ids[cur + o]]], samples_per_gpu=1) for o in offs]
            for b in batches:
                perturb(b, mode, donors, rng, stats)
            reset_stream(model.module)
            result = None
            for bi, collated in enumerate(batches):
                with torch.no_grad(), amp():
                    if bi < len(batches) - 1:
                        history_bev_step(model, collated)
                    else:
                        result = model(return_loss=False, rescale=True, **collated)
            collated = batches[-1]
            if not bool(np.asarray(unwrap(collated, 'fut_valid_flag')).reshape(-1)[0]):
                skipped += 1
                continue
            pred = result[0]['pts_bbox']['ego_fut_preds']
            cmd = np.asarray(unwrap(collated, 'ego_fut_cmd')).reshape(-1, pred.shape[0])[0]
            pred = pred[int(cmd.argmax())].cpu().double().cumsum(0).numpy()
            gt = np.asarray(unwrap(collated, 'ego_fut_trajs')).reshape(-1, 2).cumsum(0)
            errs.append(np.linalg.norm(pred - gt, axis=1))
        print('    [%s] 시나리오 %d/%d  평가점 %d' % (mode, si + 1, len(groups), len(errs)), flush=True)
    return np.array(errs), skipped, stats


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    offs = [int(x) for x in args.offsets.split(',')]
    assert offs[-1] == 0 and all(o <= 0 for o in offs), 'offsets 는 현재 프레임 0 으로 끝나야 한다'

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
    n_pts = sum(len(range(HIS_FRAMES, len(ids), args.eval_stride)) for _, ids in groups)
    print('시나리오 %d개 | 평가점 %d | 창 offsets=%s | fp16=%s | %s'
          % (len(groups), n_pts, offs, args.fp16, args.checkpoint), flush=True)

    # 목표점 조건 모델인데 val pkl 이 13차원이 아니면 측정 자체가 무의미하다
    lcf0 = unwrap(collate([dataset[groups[0][1][0]]], samples_per_gpu=1), 'ego_lcf_feat')
    use_tc = cfg.model.get('pts_bbox_head', {}).get('transformer', {}).get('use_target_cond', False)
    print('ego_lcf_feat 차원 %d, use_target_cond=%s' % (lcf0.shape[-1], use_tc))
    if use_tc and lcf0.shape[-1] < 13:
        raise SystemExit('[중단] 목표점 조건 모델인데 ann-file 이 13차원이 아니다. tgt/pre 13차원 pkl 을 쓸 것.')

    # 기증 영상·목표점: 평가 대상 뒤쪽 시나리오(없으면 전체)에서
    donors = {'img': [], 'goal': []}
    pool = list(grouped.items())[len(groups):] or list(grouped.items())
    for _tok, sids in pool[:5]:
        for gi in sids[::60]:
            c = collate([dataset[gi]], samples_per_gpu=1)
            donors['img'].append(unwrap(c, 'img').clone())
            lcf = unwrap(c, 'ego_lcf_feat')
            if lcf.shape[-1] >= 13:
                donors['goal'].append(lcf[..., 11:13].clone())
    print('기증 영상 %d장, 기증 목표점 %d개' % (len(donors['img']), len(donors['goal'])), flush=True)

    results, stat_all = {}, {}
    for mode in [m.strip() for m in args.modes.split(',')]:
        if mode == 'goalshuf' and not donors['goal']:
            continue
        print('--- %s ---' % mode, flush=True)
        errs, skipped, stats = run_mode(model, dataset, groups, offs, args.eval_stride, mode, donors, rng, args.fp16)
        if len(errs) == 0:
            print('  유효 샘플 없음'); continue
        results[mode] = official_l2(errs)
        stat_all[mode] = stats
        print('  유효 %d / 제외 %d / 입력 영상 |x| %.4f / lcf 차원 %d / 목표점 |x| %s'
              % (len(errs), skipped, np.mean(stats['img']), stats['lcf_dim'],
                 ('%.4f' % np.mean(stats['goal'])) if stats['goal'] else '-'), flush=True)

    print()
    print('=' * 84)
    print('%-10s %8s %8s %8s %10s %10s %10s %8s' % ('mode', '1s', '2s', '3s', 'L2_avg', '악화', '입력|x|', 'lcf'))
    print('-' * 84)
    base = results.get('normal', (None,) * 4)[3]
    for mode, (a1, a2, a3, avg) in results.items():
        ratio = ('%9.2f배' % (avg / base)) if base else '        -'
        st = stat_all[mode]
        print('%-10s %8.3f %8.3f %8.3f %10.4f %s %10.4f %8d'
              % (mode, a1, a2, a3, avg, ratio, np.mean(st['img']), st['lcf_dim']))
    print('=' * 84)

    ok = True
    if 'zero' in stat_all and np.mean(stat_all['zero']['img']) > 1e-6:
        print('!! zero 모드인데 입력 영상이 0 이 아니다.'); ok = False
    if 'shuffle' in stat_all and 'normal' in stat_all and \
            abs(np.mean(stat_all['shuffle']['img']) - np.mean(stat_all['normal']['img'])) < 1e-9:
        print('!! shuffle 입력이 normal 과 동일하다.'); ok = False
    if 'goaloff' in stat_all and stat_all['goaloff']['lcf_dim'] >= 13:
        print('!! goaloff 인데 lcf 차원이 안 줄었다.'); ok = False
    if not ok:
        print('!! 스크립트 문제다. 아래 결론을 믿지 말 것.'); return

    if base:
        img_modes = [m for m in ('zero', 'shuffle') if m in results]
        if img_modes:
            worst = max(results[m][3] for m in img_modes)
            print()
            print('영상을 망가뜨렸을 때 L2 가 %.2f배로 악화됐다 (제출 프로토콜 기준).' % (worst / base))
            if worst / base < 1.2:
                print('악화가 20% 미만이다. 모델이 영상을 사실상 안 보고 있다는 뜻이며 제출하면 안 된다.')
            else:
                print('영상이 결과에 실질적으로 기여하고 있다.')
        for m in ('goaloff', 'goalshuf'):
            if m in results:
                print('%s: %.2f배 (목표점 교란)' % (m, results[m][3] / base))


if __name__ == '__main__':
    main()
