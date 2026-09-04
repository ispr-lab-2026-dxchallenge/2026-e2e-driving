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
    parser.add_argument('--history-bev-only', action='store_true', default=True,
                        help='이력 프레임은 BEV 생성까지만 수행 (2026-08-25 운영국 허용). '
                             'L2 동일, T_infer -21%% 실측. 기본 켜짐')
    parser.add_argument('--full-forward', dest='history_bev_only', action='store_false',
                        help='이력 프레임도 전체 forward (배포 원래 방식)')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='forward 를 fp16 autocast 로 (2026-08-25 운영국 허용). 기본 켜짐')
    parser.add_argument('--fp32', dest='fp16', action='store_false')
    parser.add_argument('--offsets', default='-10,-5,0',
                        help='사용할 프레임 오프셋 (5의 배수, 0.5초 간격). '
                             '전체 val 실측 최적은 -10,-5,0 (eval_protocol.py 참조). '
                             '빈 문자열이면 --frames 의 균등 분할을 쓴다')
    parser.add_argument('--frames', type=int, default=3,
                        help='clip 당 사용할 프레임 수 (1~7). 배포 기본값은 7 이지만 '
                             '실측상 3 이 L2 도 좋고 2.4배 빠르다. tools/eval_protocol.py 참조')
    return parser.parse_args()


def history_bev_step(model, collated):
    """이력 프레임을 BEV 생성까지만 처리한다 (tools/eval_protocol.py 와 동일 구현)."""
    import copy as _copy
    from mmcv.parallel import scatter
    m = model.module
    data = scatter(collated, [torch.cuda.current_device()])[0]
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
    prev_bev = m.pts_bbox_head(img_feats, [meta],
                               m.prev_frame_info['prev_bev'], only_bev=True)
    m.prev_frame_info['prev_bev'] = prev_bev
    m.prev_frame_info['prev_pos'] = tmp_pos
    m.prev_frame_info['prev_angle'] = tmp_angle


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
    # 목표점 조건 모델은 13차원 ego_lcf_feat(목표점 포함) pkl 이어야 한다.
    # pre_infos_temporal_test.pkl(9차원)로 돌리면 헤드 가드가 조건을 조용히 꺼서
    # 목표점 없는 모델로 제출되는 사고가 난다 (2026-09-04 사전 발견). tgt pkl 을 쓸 것.
    _tr = cfg.model.get('pts_bbox_head', {}).get('transformer', {})
    if _tr.get('use_target_cond', False):
        import numpy as _np
        _n = _np.asarray(dataset.data_infos[0].get('gt_ego_lcf_feat', [])).reshape(-1).shape[0]
        if _n < 13:
            raise SystemExit('[중단] 목표점 조건 모델인데 ann-file 의 gt_ego_lcf_feat 가 %d차원이다 (13 필요). '
                             'data/etri/vad_etri_tgt_infos_temporal_test.pkl 을 쓸 것.' % _n)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()

    clips = OrderedDict()
    for gi, info in enumerate(dataset.data_infos):
        clips.setdefault(info['scene_token'], []).append(gi)

    submission = {}
    # 테스트 clip 의 7개 샘플은 오프셋 [-30,-25,...,0] 순서다.
    # 오프셋 o 의 인덱스는 6 + o/5 (o 는 5의 배수, -30 <= o <= 0).
    sel_idx = None
    if args.offsets:
        offs = [int(x) for x in args.offsets.split(',')]
        assert all(o % 5 == 0 and -30 <= o <= 0 for o in offs),             '오프셋은 -30~0 사이 5의 배수여야 한다 (테스트 데이터가 0.5초 간격)'
        assert offs[-1] == 0, '마지막은 반드시 현재 프레임(0)이어야 한다'
        sel_idx = [6 + o // 5 for o in offs]

    n_used = None
    for clip_token, sample_ids in mmcv.track_iter_progress(list(clips.items())):
        if sel_idx is not None:
            used = [sample_ids[i] for i in sel_idx]
        else:
            used = pick_frames(sample_ids, args.frames)
        if n_used is None:
            n_used = len(used)
            print('clip 당 %d 프레임 중 %d 개 사용' % (len(sample_ids), n_used))
        reset_stream(model.module)
        result = None
        collated = None
        import contextlib
        amp = (lambda: torch.autocast('cuda', dtype=torch.float16)) if args.fp16             else contextlib.nullcontext
        for fi, gi in enumerate(used):
            collated = collate([dataset[gi]], samples_per_gpu=1)
            with torch.no_grad(), amp():
                if args.history_bev_only and fi < len(used) - 1:
                    history_bev_step(model, collated)
                else:
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
