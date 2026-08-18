# -*- coding: utf-8 -*-
"""ETRI E2E Driving - 이미지 사전 전처리.

학습 파이프라인이 매 스텝 반복하던 undistort / crop / scale 을 미리 한 번만 수행해
작은 JPEG 로 저장한다. tar 를 디스크에 펼치지 않고 스트리밍으로 읽는다.

원본 파이프라인과 동일한 연산을 사용한다:
  undistort : tools/data_converter/etri_vad_converter.py  undistorted_intrinsic()
              projects/.../pipelines/transform_3d.py       UndistortMultiViewImage
  crop      : projects/.../datasets/etri_vad_dataset.py    get_data_info()  crop_box
              projects/.../pipelines/transform_3d.py       CropMultiViewImage
  scale     : projects/.../pipelines/transform_3d.py       RandomScaleImageMultiViewImage
"""
import argparse
import io
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd

cv2.setNumThreads(1)  # 프로세스 병렬이므로 OpenCV 내부 스레딩은 끈다

CAM_NAMES = [
    "camera_front", "camera_front_right", "camera_front_left",
    "camera_rear_wide", "camera_rear_left", "camera_rear_right",
]

# VAD_etri_tiny.py 의 crop_keep_top 과 동일 (camera_front 만 아래쪽을 남긴다)
CROP_KEEP_TOP = (
    "camera_front_left", "camera_front_right",
    "camera_rear_left", "camera_rear_right", "camera_rear_wide",
)


def undistorted_intrinsic(K, dist, size, is_fisheye):
    """converter 의 undistorted_intrinsic() 와 동일."""
    if is_fisheye:
        return cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, dist[:4], size, np.eye(3), balance=0.0)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha=0)
    return new_K


def build_maps(K, dist, new_K, size, is_fisheye):
    """UndistortMultiViewImage._get_maps() 와 동일."""
    if is_fisheye:
        return cv2.fisheye.initUndistortRectifyMap(
            K, dist[:4], np.eye(3), new_K, size, cv2.CV_32FC1)
    return cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_32FC1)


def crop_offset(cam_name, W, H, crop_w, crop_h):
    """etri_vad_dataset.get_data_info() 의 crop_box 계산과 동일."""
    ox = (W - crop_w) // 2
    oy = 0 if cam_name in CROP_KEEP_TOP else H - crop_h
    return ox, oy


def adjusted_intrinsic(new_K, ox, oy, scale):
    """crop(shift) + scale 을 K 에 반영. CropMultiViewImage / RandomScale 과 동일."""
    K = np.array(new_K, dtype=np.float64).copy()
    K[0, 2] -= ox
    K[1, 2] -= oy
    K[:2, :] *= scale
    return K


def read_calibration(tf):
    """tar 안 아무 위치의 calibration.parquet 를 찾아 읽는다."""
    for m in tf.getmembers():
        if m.isfile() and m.name.endswith("calibration.parquet"):
            return pd.read_parquet(io.BytesIO(tf.extractfile(m).read()))
    raise FileNotFoundError("calibration.parquet not found in tar")


def process_one(tar_path, out_root, crop_w, crop_h, scale, quality):
    """tar 하나를 처리. (이름, 이미지수, 입력바이트, 출력바이트, 초) 반환."""
    t0 = time.time()
    n_img = 0
    in_bytes = 0
    out_bytes = 0

    with tarfile.open(tar_path, mode="r:") as tf:
        calib = read_calibration(tf).set_index("camera_name")

        maps = {}
        for cam in CAM_NAMES:
            if cam not in calib.index:
                continue
            r = calib.loc[cam]
            K = np.array(r["K"], dtype=np.float64).reshape(3, 3)
            dist = np.array(r["distortion"], dtype=np.float64)
            fisheye = bool(r["is_fisheye"])
            W, H = int(r["image_width"]), int(r["image_height"])
            new_K = undistorted_intrinsic(K, dist, (W, H), fisheye)
            m1, m2 = build_maps(K, dist, new_K, (W, H), fisheye)
            ox, oy = crop_offset(cam, W, H, crop_w, crop_h)
            maps[cam] = (m1, m2, ox, oy)

        dst_w, dst_h = int(crop_w * scale), int(crop_h * scale)
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        for m in tf:
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()
            in_bytes += len(data)
            dst = os.path.join(out_root, m.name)
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            if m.name.lower().endswith(".jpg"):
                cam = os.path.basename(os.path.dirname(m.name))
                if cam not in maps:                       # 미지의 카메라는 원본 복사
                    with open(dst, "wb") as f:
                        f.write(data)
                    out_bytes += len(data)
                    continue
                m1, m2, ox, oy = maps[cam]
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)          # undistort
                img = img[oy:oy + crop_h, ox:ox + crop_w]               # crop
                img = cv2.resize(img, (dst_w, dst_h),
                                 interpolation=cv2.INTER_LINEAR)        # scale
                ok, buf = cv2.imencode(".jpg", img, enc)
                if not ok:
                    raise RuntimeError("jpeg encode failed: %s" % m.name)
                with open(dst, "wb") as f:
                    f.write(buf.tobytes())
                out_bytes += buf.size
                n_img += 1
            else:                                          # parquet 등은 그대로
                with open(dst, "wb") as f:
                    f.write(data)
                out_bytes += len(data)

    return (os.path.basename(tar_path), n_img, in_bytes, out_bytes,
            time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description="ETRI 이미지 사전 전처리")
    ap.add_argument("--src", required=True, help="tar 들이 있는 디렉토리")
    ap.add_argument("--out", required=True, help="출력 루트")
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--crop-w", type=int, default=1920)
    ap.add_argument("--crop-h", type=int, default=1080)
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--names", nargs="*", default=None,
                    help="처리할 tar 이름 (확장자 제외). 미지정 시 전부")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tars = sorted(f for f in os.listdir(args.src) if f.endswith(".tar"))
    if args.names:
        want = set(args.names)
        tars = [t for t in tars if os.path.splitext(t)[0] in want]
    if args.limit:
        tars = tars[:args.limit]
    if not tars:
        print("처리할 tar 가 없습니다"); sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    print("tar %d개 | scale %.2f | crop %dx%d -> %dx%d | Q%d | workers %d"
          % (len(tars), args.scale, args.crop_w, args.crop_h,
             int(args.crop_w * args.scale), int(args.crop_h * args.scale),
             args.quality, args.workers))
    print("-" * 68)

    t0 = time.time()
    tot_img = tot_in = tot_out = 0
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, os.path.join(args.src, t), args.out,
                          args.crop_w, args.crop_h, args.scale,
                          args.quality): t for t in tars}
        for fu in as_completed(futs):
            name, n, bi, bo, dt = fu.result()
            done += 1
            tot_img += n; tot_in += bi; tot_out += bo
            el = time.time() - t0
            eta = el / done * (len(tars) - done)
            print("[%4d/%4d] %-24s %5d장 %6.1fs  %5.1f -> %5.1f MB   ETA %5.1f분"
                  % (done, len(tars), name, n, dt, bi / 1e6, bo / 1e6, eta / 60))

    el = time.time() - t0
    print("-" * 68)
    print("완료: %d개 tar, 이미지 %d장, %.1f분" % (len(tars), tot_img, el / 60))
    print("용량: %.2f GB -> %.2f GB  (%.1f%%)"
          % (tot_in / 1e9, tot_out / 1e9, 100 * tot_out / max(tot_in, 1)))
    if tot_img:
        print("장당 평균 %.1f KB" % (tot_out / tot_img / 1e3))


if __name__ == "__main__":
    main()
