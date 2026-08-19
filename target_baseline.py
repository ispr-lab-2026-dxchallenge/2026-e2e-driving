# -*- coding: utf-8 -*-
"""5초 목표점을 쓰는 자명한 예측들의 L2.

테스트셋 ego_pose.parquet 에는 frame=+50 (5초 후 위치) 이 주어진다.
이걸 쓰면 학습 없이도 어디까지 가는지 확인한다.
"""
import numpy as np, pandas as pd, mmcv, os
from scipy.spatial.transform import Rotation

ROOT = "/workspace/VAD/data/etri_pre/train"
PKL = "/workspace/VAD/data/etri/vad_etri_pre_infos_temporal_valsplit.pkl"
FRAME_OFFSET, MAIN = 50, 300

d = mmcv.load(PKL)
infos = d["infos"]
scen = sorted(set(i["scene_token"] for i in infos))
print("val 시나리오 %d개, 샘플 %d개" % (len(scen), len(infos)))

# 시나리오별로 5초 목표점을 미리 계산 (converter 의 ego_positions_in_frame 과 동일)
tgt_of = {}
for s in scen:
    p = os.path.join(ROOT, s)
    ts = pd.read_parquet(p + "/meta/timestamps.parquet")[["timestamp", "frame_id"]]
    ep = pd.read_parquet(p + "/annotation/ego_pose.parquet").merge(ts, on="timestamp")
    n = FRAME_OFFSET * 2 + MAIN
    xyz = np.full((n, 3), np.nan); rpy = np.full((n, 3), np.nan)
    idx = ep["frame_id"].to_numpy() + FRAME_OFFSET
    xyz[idx] = ep[["x", "y", "z"]].to_numpy()
    rpy[idx] = ep[["roll", "pitch", "yaw"]].to_numpy()
    for f in range(MAIN):
        i0, i5 = f + FRAME_OFFSET, min(f + 50 + FRAME_OFFSET, n - 1)
        R = Rotation.from_euler("xyz", rpy[i0]).as_matrix()
        tgt_of[(s, f)] = ((xyz[i5] - xyz[i0]) @ R)[:2]

fut, his, tgt = [], [], []
for i in infos:
    if float(np.asarray(i["gt_ego_fut_masks"]).min()) <= 0:
        continue
    fut.append(np.asarray(i["gt_ego_fut_trajs"], np.float64).reshape(-1, 2))
    his.append(np.asarray(i["gt_ego_his_trajs"], np.float64).reshape(-1, 2))
    tgt.append(tgt_of[(i["scene_token"], int(i["frame_idx"]))])
fut = np.stack(fut); his = np.stack(his); tgt = np.stack(tgt)
gt = np.cumsum(fut, axis=1)
print("유효 샘플 %d,  5초 목표점 평균 거리 %.1f m" % (len(gt), np.linalg.norm(tgt, axis=1).mean()))

T = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

def rep(name, pred):
    e = np.linalg.norm(pred - gt, axis=2)
    a1, a2, a3 = e[:, :2].mean(), e[:, :4].mean(), e[:, :6].mean()
    print("  %-34s ADE 1s %.3f  2s %.3f  3s %.3f  ->  avg %.4f"
          % (name, a1, a2, a3, (a1 + a2 + a3) / 3))

print()
print("=" * 92)
print("5초 목표점을 이용한 예측")
print("=" * 92)

# ⑤ 목표점까지 직선 등속
rep("⑤ 목표점까지 직선 등속", tgt[:, None, :] * (T / 5.0)[None, :, None])

# ⑥ 과거 속도로 진행거리 결정 + 방향은 목표점
v = np.linalg.norm(his[:, -1, :], axis=1) / 0.5              # m/s
u = tgt / np.maximum(np.linalg.norm(tgt, axis=1, keepdims=True), 1e-6)
rep("⑥ 과거 속력 x 목표점 방향", (v[:, None] * T[None, :])[:, :, None] * u[:, None, :])

# ⑦ 3차 Hermite: p(0)=0, p'(0)=과거속도, p(5)=목표점, p'(5)=목표점방향*과거속력
p0 = np.zeros_like(tgt); m0 = his[:, -1, :] / 0.5
m1 = u * v[:, None]
s = (T / 5.0)[None, :, None]
h00 = 2*s**3 - 3*s**2 + 1; h10 = s**3 - 2*s**2 + s
h01 = -2*s**3 + 3*s**2;    h11 = s**3 - s**2
rep("⑦ 3차 Hermite (양끝 속도 반영)",
    h00*p0[:, None, :] + h10*5.0*m0[:, None, :] + h01*tgt[:, None, :] + h11*5.0*m1[:, None, :])

# ⑧ 등속+등각속도(과거만)와 ⑦의 평균
a0 = np.arctan2(his[:, 0, 1], his[:, 0, 0]); a1_ = np.arctan2(his[:, 1, 1], his[:, 1, 0])
dth = np.arctan2(np.sin(a1_-a0), np.cos(a1_-a0)); sp = np.linalg.norm(his[:, -1, :], axis=1)
th = a1_.copy(); st = []
for k in range(6):
    th = th + dth; st.append(np.stack([sp*np.cos(th), sp*np.sin(th)], 1))
kin = np.cumsum(np.stack(st, 1), axis=1)
her = (h00*p0[:, None, :] + h10*5.0*m0[:, None, :] + h01*tgt[:, None, :] + h11*5.0*m1[:, None, :])
rep("⑧ ⑦과 등속+등각속도의 평균", 0.5*(kin+her))
print("=" * 92)
