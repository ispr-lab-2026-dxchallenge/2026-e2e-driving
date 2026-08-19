# -*- coding: utf-8 -*-
"""자명한 베이스라인들의 L2 를 val split 에서 측정.

리더보드 상위 점수(0.077)가 학습 없이도 나오는 수준인지 확인한다.
"""
import numpy as np, mmcv

PKL = "/workspace/VAD/data/etri/vad_etri_pre_infos_temporal_valsplit.pkl"
d = mmcv.load(PKL)
infos = d["infos"]
print("val 샘플:", len(infos))

fut, his, mask = [], [], []
for i in infos:
    fut.append(np.asarray(i["gt_ego_fut_trajs"], np.float64).reshape(-1, 2))
    his.append(np.asarray(i["gt_ego_his_trajs"], np.float64).reshape(-1, 2))
    mask.append(np.asarray(i["gt_ego_fut_masks"], np.float64).reshape(-1))
fut = np.stack(fut); his = np.stack(his); mask = np.stack(mask)
print("gt_ego_fut_trajs :", fut.shape, " (delta)")
print("gt_ego_his_trajs :", his.shape)
valid = mask.min(1) > 0
print("전 구간 유효 샘플 : %d / %d" % (valid.sum(), len(infos)))

fut, his = fut[valid], his[valid]
gt = np.cumsum(fut, axis=1)                    # (N,6,2) 누적 위치

def l2_report(name, pred):
    err = np.linalg.norm(pred - gt, axis=2)    # (N,6)
    # 가이드라인: 각 시점 L2 = 해당 구간 waypoint 의 평균 변위(ADE)
    a1, a2, a3 = err[:, :2].mean(), err[:, :4].mean(), err[:, :6].mean()
    avg = (a1 + a2 + a3) / 3
    # 참고용: 그 시점의 단일 오차
    p1, p2, p3 = err[:, 1].mean(), err[:, 3].mean(), err[:, 5].mean()
    print("  %-26s ADE  1s %.3f  2s %.3f  3s %.3f  -> avg %.4f   |  단일 %.3f %.3f %.3f"
          % (name, a1, a2, a3, avg, p1, p2, p3))
    return avg

print()
print("=" * 96)
print("자명한 예측들의 L2 (val 39 시나리오)")
print("=" * 96)

l2_report("① 정지 (전부 0)", np.zeros_like(gt))

v = his[:, -1, :]                              # 직전 0.5초 변위
pred = np.cumsum(np.repeat(v[:, None, :], 6, axis=1), axis=1)
l2_report("② 등속 (과거 0.5초 유지)", pred)

v2 = his.mean(1)                               # 과거 1초 평균 속도
pred = np.cumsum(np.repeat(v2[:, None, :], 6, axis=1), axis=1)
l2_report("③ 등속 (과거 1초 평균)", pred)

# 등속 + 등각속도 (과거 두 구간의 방향 변화를 이어감)
a0 = np.arctan2(his[:, 0, 1], his[:, 0, 0])
a1_ = np.arctan2(his[:, 1, 1], his[:, 1, 0])
dth = np.arctan2(np.sin(a1_ - a0), np.cos(a1_ - a0))
sp = np.linalg.norm(his[:, -1, :], axis=1)
steps = []
th = a1_.copy()
for k in range(6):
    th = th + dth
    steps.append(np.stack([sp * np.cos(th), sp * np.sin(th)], 1))
l2_report("④ 등속+등각속도", np.cumsum(np.stack(steps, 1), axis=1))

print()
print("전체 평균 주행 거리 (3초) : %.1f m" % np.linalg.norm(gt[:, -1, :], axis=1).mean())
print("=" * 96)
