"""VO 헤드 출처 검증 (2차 심사 방어용). GPU 에서 실행:
  python tests/test_vo_provenance.py projects/configs/VAD/exp_full_tgtbev_vo.py work_dirs/exp_full_tgtbev/soup_e5e6.pth

검사:
  (a) can_bus 를 무작위로 바꿔도 ego_motion_pred 가 비트 단위로 같다
  (b) ego_lcf_feat(목표점 포함)·ego_his_trajs 를 바꿔도 같다
  (c) 영상을 0 으로 채우면 ego_motion_pred 가 바뀌고, 서로 다른 두 프레임에서 같은 값(입력 무관 상수)이 된다
  (d) vo_to_planner 가 0 이면(로드 직후) ego_fut_preds 가 use_vo_head=False 모델과 동일하다 (하방 0)
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


def build(cfg, ckpt, use_vo):
    cfg = copy.deepcopy(cfg)
    cfg.model.pts_bbox_head.use_vo_head = use_vo
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt, map_location='cpu')
    model.compute_planner_metric_stp3 = lambda *a, **k: {}
    model = MMDataParallel(model.cuda(0), device_ids=[0])
    model.eval()
    return model


def reset(model):
    model.module.prev_frame_info = {'prev_bev': None, 'scene_token': None, 'prev_pos': 0, 'prev_angle': 0}
    model.module.pts_bbox_head._vo_stash = None


def hist_step(model, collated, perturb=None):
    m = model.module
    dev = torch.cuda.current_device()
    data = scatter(collated, [dev])[0]
    meta = data['img_metas'][0][0]
    img = data['img'][0]
    if perturb: perturb(meta, img, data)
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
    prev_bev = m.pts_bbox_head(img_feats, [meta], m.prev_frame_info['prev_bev'], only_bev=True)
    m.prev_frame_info['prev_bev'] = prev_bev
    m.prev_frame_info['prev_pos'] = tmp_pos; m.prev_frame_info['prev_angle'] = tmp_angle


def full_step(model, collated, perturb=None):
    c = copy.deepcopy(collated)
    if perturb:
        v = c['img_metas'][0] if isinstance(c['img_metas'], list) else c['img_metas']
        meta = v.data[0][0]
        img = (c['img'][0] if isinstance(c['img'], list) else c['img']).data[0]
        perturb(meta, img, c)
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **c)
    head = model.module.pts_bbox_head
    return out[0]['pts_bbox']['ego_fut_preds'].clone(), head._last_vo.clone()


def run_window(model, dataset, ids, perturb=None):
    """3프레임 창(-10,-5,0) 을 제출 프로토콜대로 돌린다. 교란은 창의 모든 프레임에 건다."""
    reset(model)
    batches = [collate([dataset[i]], samples_per_gpu=1) for i in ids]
    with torch.no_grad():
        for b in batches[:-1]:
            hist_step(model, b, perturb)
    return full_step(model, batches[-1], perturb)


def main():
    p = argparse.ArgumentParser(); p.add_argument('config'); p.add_argument('checkpoint')
    a = p.parse_args()
    cfg = Config.fromfile(a.config)
    if hasattr(cfg, 'plugin_dir'):
        importlib.import_module(cfg.plugin_dir.replace('/', '.').rstrip('.'))
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None); cfg.data.test.pop('map_ann_file', None)
    dataset = build_dataset(cfg.data.test)
    grouped = {}
    for gi, info in enumerate(dataset.data_infos):
        grouped.setdefault(info['scene_token'], []).append(gi)
    scen = list(grouped.values())[0]
    win1 = [scen[30 - 10], scen[30 - 5], scen[30]]
    win2 = [scen[60 - 10], scen[60 - 5], scen[60]]

    model = build(cfg, a.checkpoint, True)
    head = model.module.pts_bbox_head
    # ego_motion_pred 를 잡아두기 위한 훅
    orig = head._vo_forward
    def hooked(*args, **kw):
        s, v = orig(*args, **kw); head._last_vo = s.detach(); return s, v
    head._vo_forward = hooked

    rng = np.random.default_rng(0)
    def perturb_canbus(meta, img, c):
        meta['can_bus'] = np.asarray(meta['can_bus'], dtype=np.float64) + rng.normal(0, 3.0, size=18)
    def perturb_lcf(meta, img, c):
        for k in ('ego_lcf_feat', 'ego_his_trajs'):
            v = c[k][0] if isinstance(c[k], list) else c[k]
            v.data[0].normal_()
    def perturb_zero(meta, img, c):
        img.zero_()

    traj0, vo0 = run_window(model, dataset, win1)
    _, vo_cb = run_window(model, dataset, win1, perturb_canbus)
    _, vo_lcf = run_window(model, dataset, win1, perturb_lcf)
    _, vo_z1 = run_window(model, dataset, win1, perturb_zero)
    _, vo_z2 = run_window(model, dataset, win2, perturb_zero)
    _, vo_w2 = run_window(model, dataset, win2)

    ok = True
    def check(name, cond, detail):
        nonlocal ok
        print('%s %-48s %s' % ('PASS' if cond else 'FAIL', name, detail))
        ok = ok and cond
    check('(a) can_bus 무작위 -> ego_motion_pred 불변', torch.equal(vo0, vo_cb), str(vo0.flatten().tolist()))
    check('(b) ego_lcf_feat/ego_his_trajs 무작위 -> 불변', torch.equal(vo0, vo_lcf), '')
    check('(c1) 영상 0 -> ego_motion_pred 변함', not torch.equal(vo0, vo_z1), str(vo_z1.flatten().tolist()))
    check('(c2) 영상 0 이면 다른 프레임에서도 같은 값(상수)', torch.allclose(vo_z1, vo_z2, atol=1e-5), '')
    check('(c3) 정상 영상 두 창의 예측은 서로 다름', not torch.equal(vo0, vo_w2), str(vo_w2.flatten().tolist()))

    plain = build(cfg, a.checkpoint, False)
    reset(plain)
    batches = [collate([dataset[i]], samples_per_gpu=1) for i in win1]
    with torch.no_grad():
        for b in batches[:-1]:
            hist_step(plain, b)
        outp = plain(return_loss=False, rescale=True, **batches[-1])
    trajp = outp[0]['pts_bbox']['ego_fut_preds']
    w = head.vo_to_planner.weight.abs().max().item()
    check('(d) vo_to_planner==0 -> 궤적이 VO 없는 모델과 동일', w == 0.0 and torch.allclose(traj0, trajp, atol=1e-6),
          'max|W|=%.3g, max|dtraj|=%.3g' % (w, (traj0 - trajp).abs().max().item()))
    print('\n결과:', '전부 통과' if ok else '실패 있음')


if __name__ == '__main__':
    main()
