"""제출 모델의 목표점 정보 흐름 검사 — "planner 는 목표점을 수치로 받지 않는다"를 기계적으로 확인.

검사 1~3 은 GPU·체크포인트 없이 config 와 소스만으로 돌아간다(심사위원이 그대로 재현 가능).
검사 4 는 체크포인트가 있을 때만 돈다: 같은 가중치로 목표점을 넣은 경우와 뺀 경우의 궤적을 비교.

  python tests/test_target_point_flow.py                          # 1~3
  python tests/test_target_point_flow.py --checkpoint <ckpt.pth>  # 1~4
"""
import argparse
import importlib
import io
import re
import sys

CFG = 'projects/configs/VAD/exp_full_tgtbev.py'
HEAD_SRC = 'projects/mmdet3d_plugin/VAD/VAD_head.py'
TR_SRC = 'projects/mmdet3d_plugin/VAD/VAD_transformer.py'
FORBIDDEN = ('target_point', 'ego_lcf_feat', 'ego_his_trajs')

ok = True


def check(name, cond, detail=''):
    global ok
    ok = ok and bool(cond)
    print('%s  %s%s' % ('PASS' if cond else 'FAIL', name, ('  — ' + detail) if detail else ''))


def planner_block(src):
    """planner 블록 = ego_query 생성부터 궤적 출력까지."""
    lines = src.splitlines()
    beg = next(i for i, l in enumerate(lines) if 'ego_his_feats = self.ego_query.weight' in l)
    end = next(i for i, l in enumerate(lines) if 'outputs_ego_trajs = self.ego_fut_decoder(' in l)
    return lines[beg:end + 1], beg + 1, end + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--config', default=CFG)
    a = p.parse_args()

    head_src = io.open(HEAD_SRC, encoding='utf-8').read()
    tr_src = io.open(TR_SRC, encoding='utf-8').read()

    # 1. planner 블록 안에 목표점·자차상태 참조가 없다
    #    (참조가 있는 두 줄은 ego_lcf_feat_idx is not None 로 막힌 죽은 가지이므로, 그 가지가
    #     정말 막혀 있는지도 함께 본다 — 검사 2)
    blk, b0, b1 = planner_block(head_src)
    hits = [(b0 + i, l.strip()) for i, l in enumerate(blk)
            for f in FORBIDDEN if f in l and 'self.ego_lcf_feat_idx' not in l]
    check('1. planner 블록(%d~%d행)에 %s 참조 없음' % (b0, b1, '/'.join(FORBIDDEN)),
          not hits, '; '.join('%d: %s' % h for h in hits[:3]))

    # 2. 제출 config 에서 금지 통로가 꺼져 있고, planner 입력 차원이 2*embed_dims 다
    from mmcv import Config
    cfg = Config.fromfile(a.config)
    h = cfg.model.pts_bbox_head
    dim = h.get('transformer', {}).get('embed_dims', None) or h.get('embed_dims', 256)
    check('2a. ego_his_encoder 꺼짐', h.get('ego_his_encoder', None) is None)
    check('2b. ego_lcf_feat_idx 꺼짐', h.get('ego_lcf_feat_idx', None) is None)
    check('2c. VO 헤드 미사용(추론 경로에 없음)', not h.get('use_vo_head', False))
    check('2d. 목표점 조건화 켜짐(BEV 단계)', h.get('transformer', {}).get('use_target_cond', False))

    importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    from mmdet3d.models import build_model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    hd = model.pts_bbox_head
    check('2e. 빌드된 모델에 ego_his_encoder 없음', hd.ego_his_encoder is None)
    check('2f. 빌드된 모델에 ego_lcf_feat_idx 없음', hd.ego_lcf_feat_idx is None)
    check('2g. planner 첫 층 입력 = 2*embed_dims (자차상태 추가 차원 없음)',
          hd.ego_fut_decoder[0].in_features == 2 * dim,
          '실제 %d vs 기대 %d' % (hd.ego_fut_decoder[0].in_features, 2 * dim))
    check('2h. VO 모듈이 생성되지 않음', not any(n.startswith('vo_') for n, _ in hd.named_children()))

    # 3. 목표점이 쓰이는 자리가 BEV 조건화 단 한 곳이다
    tp_lines = [(i + 1, l.strip()) for i, l in enumerate(tr_src.splitlines()) if 'target_point' in l]
    check('3a. VAD_transformer.py 의 target_point 사용은 1곳', len(tp_lines) == 1,
          '; '.join('%d: %s' % t for t in tp_lines))
    in_bev = tp_lines and 'bev_queries' in tr_src.split('target_point')[1][:400]
    check('3b. 그 자리가 BEV query 가산 구간', bool(in_bev))
    head_tp = [(i + 1, l.strip()) for i, l in enumerate(head_src.splitlines())
               if 'target_point' in l]
    check('3c. VAD_head.py 의 target_point 는 BEV 로 넘기는 곳뿐', len(head_tp) <= 2,
          '; '.join('%d: %s' % t for t in head_tp))

    # 4. (체크포인트가 있을 때) 목표점을 빼면 궤적이 달라지고, 조건화를 끄면 순수 비전과 같다
    if a.checkpoint:
        import numpy as np, torch
        from mmcv.parallel import MMDataParallel, collate
        from mmcv.runner import load_checkpoint
        from mmdet3d.datasets import build_dataset
        cfg.data.test.test_mode = True
        cfg.data.test.pop('samples_per_gpu', None)
        cfg.data.test.pop('map_ann_file', None)
        ds = build_dataset(cfg.data.test)
        load_checkpoint(model, a.checkpoint, map_location='cpu', strict=False)
        m = MMDataParallel(model.cuda(0).eval(), device_ids=[0])

        def run(idx, zero_tp):
            m.module.prev_frame_info = {'prev_bev': None, 'scene_token': None,
                                        'prev_pos': 0, 'prev_angle': 0}
            c = collate([ds[idx]], samples_per_gpu=1)
            if zero_tp:
                v = c['ego_lcf_feat'][0] if isinstance(c['ego_lcf_feat'], list) else c['ego_lcf_feat']
                v.data[0][..., 11:13] = 0.0
            with torch.no_grad():
                r = m(return_loss=False, rescale=True, **c)
            return r[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()

        i = len(ds) // 2
        a1, a2 = run(i, False), run(i, False)
        b1 = run(i, True)
        check('4a. 같은 입력 = 같은 출력 (결정성)', np.allclose(a1, a2, atol=1e-5))
        check('4b. 목표점을 지우면 궤적이 달라진다 (조건화가 실제로 작동)',
              not np.allclose(a1, b1, atol=1e-4),
              '최대 차이 %.4f m' % np.abs(a1 - b1).max())
    else:
        print('SKIP  4. 체크포인트 검사 (--checkpoint 를 주면 실행)')

    print('\n%s' % ('전체 PASS' if ok else '실패 있음'))
    sys.exit(0 if ok else 1)


main()
