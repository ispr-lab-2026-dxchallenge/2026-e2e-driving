# -*- coding: utf-8 -*-
"""전처리 정합 검증.

(1) 수치 검증 - 3D 점을 두 경로로 투영해 픽셀 오차를 비교
    A: 원본 K 로 투영한 뒤 2D 에서 crop/scale        (파이프라인이 실제로 하는 일)
    B: crop/scale 을 반영한 K 로 바로 투영           (전처리 후 쓸 방식)
    두 결과가 일치해야 한다.

(2) 시각 검증 - 전처리된 이미지 위에 3D 박스와 차선을 그린다.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

RAW = "/workspace/VAD/data/etri/train"
PRE = "/workspace/VAD/data/etri_pre/train"
SCEN = "20260112-105434"
FRAME = 150
OUT = "/workspace/VAD/work_dirs/verify_pre"
CROP_W, CROP_H, SCALE = 1920, 1080, 0.4
CROP_KEEP_TOP = ("camera_front_left", "camera_front_right",
                 "camera_rear_left", "camera_rear_right", "camera_rear_wide")
CAM_NAMES = ["camera_front", "camera_front_right", "camera_front_left",
             "camera_rear_wide", "camera_rear_left", "camera_rear_right"]
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
os.makedirs(OUT, exist_ok=True)


def euler_deg_R(e):
    return Rotation.from_euler("xyz", np.asarray(e, float), degrees=True).as_matrix()


def euler_rad_R(e):
    return Rotation.from_euler("xyz", np.asarray(e, float), degrees=False).as_matrix()


def undist_K(K, d, size, fish):
    if fish:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, d[:4], size, np.eye(3), balance=0.0)
    return cv2.getOptimalNewCameraMatrix(K, d, size, alpha=0)[0]


def lidar2img(R, t, K):
    r = np.linalg.inv(R)
    tt = np.asarray(t, float) @ r.T
    rt = np.eye(4); rt[:3, :3] = r.T; rt[3, :3] = -tt
    vp = np.eye(4); vp[:3, :3] = K
    return vp @ rt.T


def project(pts, P):
    h = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    uvw = h @ P.T
    z = uvw[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return uvw[:, :2] / z[:, None], z


def g2e_yaw(p, o, yaw):
    c, s = np.cos(-yaw), np.sin(-yaw)
    return (np.asarray(p, float) - o) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T


def g2e_full(p, o, rpy):
    return (np.asarray(p, float) - o) @ euler_rad_R(rpy)


def corners(r):
    lon, lat, h = r["width[m]"], r["length[m]"], r["height[m]"]
    dx, dy = lon / 2, lat / 2
    c = np.array([[dx, dy, 0], [dx, -dy, 0], [-dx, -dy, 0], [-dx, dy, 0],
                  [dx, dy, h], [dx, -dy, h], [-dx, -dy, h], [-dx, dy, h]], float)
    y = r["heading[rad]"]; cs, sn = np.cos(y), np.sin(y)
    return c @ np.array([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]]).T + \
        np.array([r["x[m]"], r["y[m]"], r["z[m]"]], float)


# ---------- 데이터 로드 ----------
S = os.path.join(RAW, SCEN)
calib = pd.read_parquet(S + "/calibration/calibration.parquet").set_index("camera_name")
ts = pd.read_parquet(S + "/meta/timestamps.parquet")
obj = pd.read_parquet(S + "/annotation/object.parquet").merge(
    ts[["timestamp", "frame_id"]], on="timestamp")
pose = pd.read_parquet(S + "/annotation/ego_pose.parquet").merge(
    ts[["timestamp", "frame_id"]], on="timestamp")
lanes = [np.stack([np.asarray(q, float) for q in pts])
         for pts in pd.read_parquet(S + "/annotation/map.parquet")["points"]]
ego_rows, dyn = obj[obj["class"] == "ego"], obj[obj["class"] != "ego"]

pp = pose[pose.frame_id == FRAME].iloc[0]
ee = ego_rows[ego_rows.frame_id == FRAME].iloc[0]
exyz = np.array([pp.x, pp.y, pp.z]); rpy = [pp.roll, pp.pitch, pp.yaw]
bxyz = np.array([ee["x[m]"], ee["y[m]"], ee["z[m]"]]); byaw = ee["heading[rad]"]


def cam_params(cam):
    r = calib.loc[cam]
    K = np.array(r["K"], float).reshape(3, 3)
    d = np.array(r["distortion"], float)
    fish = bool(r["is_fisheye"])
    W, H = int(r["image_width"]), int(r["image_height"])
    nK = undist_K(K, d, (W, H), fish)
    ox = (W - CROP_W) // 2
    oy = 0 if cam in CROP_KEEP_TOP else H - CROP_H
    aK = nK.copy(); aK[0, 2] -= ox; aK[1, 2] -= oy; aK[:2, :] *= SCALE
    R = euler_deg_R(r["euler"]); t = np.array(r["translation"], float)
    return nK, aK, ox, oy, R, t


# ---------- (1) 수치 검증 ----------
print("=" * 70)
print("(1) 수치 검증 : 원본K + 2D crop/scale   vs   보정K 직접 투영")
print("=" * 70)
rng = np.random.default_rng(0)
worst = 0.0
for cam in CAM_NAMES:
    nK, aK, ox, oy, R, t = cam_params(cam)
    P_orig, P_adj = lidar2img(R, t, nK), lidar2img(R, t, aK)
    pts = rng.uniform([-30, -20, -3], [60, 20, 3], size=(4000, 3))
    uv0, z0 = project(pts, P_orig)
    uv1, z1 = project(pts, P_adj)
    m = z0 > 0.5
    expect = (uv0[m] - [ox, oy]) * SCALE          # 파이프라인이 하는 2D 변환
    err = np.abs(expect - uv1[m]).max()
    worst = max(worst, err)
    print("  %-20s 최대 오차 %.3e px   (표본 %d)" % (cam, err, m.sum()))
print("-" * 70)
print("  전체 최대 오차: %.3e px  ->  %s" %
      (worst, "일치" if worst < 1e-6 else "불일치! 수식 확인 필요"))

# ---------- (2) 시각 검증 ----------
print()
print("=" * 70)
print("(2) 시각 검증 : 전처리 이미지 위에 3D 박스 + 차선 투영")
print("=" * 70)


def render_pre(cam, frame, max_dist=60):
    nK, aK, ox, oy, R, t = cam_params(cam)
    P = lidar2img(R, t, aK)
    img = cv2.imread("%s/%s/%s/%08d.jpg" % (PRE, SCEN, cam, frame))
    if img is None:
        raise FileNotFoundError("%s/%s/%s/%08d.jpg" % (PRE, SCEN, cam, frame))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for l in lanes:
        q = g2e_full(l[:, :3], exyz, rpy)
        q = q[(q[:, 0] > 0) & (np.linalg.norm(q[:, :2], axis=1) < max_dist)]
        if len(q) < 2:
            continue
        uv, z = project(q, P)
        ok = (z > 0.5) & (np.abs(uv).max(1) < 1e4)
        u2 = uv[ok].astype(int)
        for a, b in zip(u2[:-1], u2[1:]):
            cv2.line(img, tuple(a), tuple(b), (255, 220, 0), 2)
    col = {"Car": (60, 120, 255), "Pedestrian": (255, 60, 60), "Cyclist": (60, 220, 60)}
    for _, o in dyn[dyn.frame_id == frame].iterrows():
        c = g2e_yaw(corners(o), bxyz, byaw)
        if np.linalg.norm(c[:, :2].mean(0)) > max_dist:
            continue
        uv, z = project(c, P)
        if (z <= 0.5).any() or np.abs(uv).max() > 1e4:
            continue
        uv = uv.astype(int)
        for a, b in EDGES:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), col.get(o["class"], (255,) * 3), 2)
    return img


im = render_pre("camera_front", FRAME)
print("  camera_front 크기:", im.shape)
plt.figure(figsize=(13, 8)); plt.imshow(im); plt.axis("off")
plt.title("PREPROCESSED  camera_front  frame %d  (768x432 + adjusted K)" % FRAME)
plt.savefig(OUT + "/pre_front.png", bbox_inches="tight", dpi=110); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(18, 8))
for a, c in zip(ax.ravel(), CAM_NAMES):
    a.imshow(render_pre(c, FRAME)); a.set_title(c, fontsize=10); a.axis("off")
plt.tight_layout(); plt.savefig(OUT + "/pre_all6.png", bbox_inches="tight", dpi=100)
plt.close()
print("  저장: %s/pre_front.png , pre_all6.png" % OUT)
print("=" * 70)
