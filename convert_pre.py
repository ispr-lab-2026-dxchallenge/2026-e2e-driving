# -*- coding: utf-8 -*-
"""전처리된 데이터용 컨버터 래퍼.

원본 컨버터(tools/data_converter/etri_*.py)를 그대로 쓰되, 카메라 내부 파라미터만
전처리(crop + scale)에 맞춰 보정한다. 원본 파일은 수정하지 않는다.

  cx -= ox ,  cy -= oy        (CropMultiViewImage 와 동일)
  fx, fy, cx, cy *= scale     (RandomScaleImageMultiViewImage 와 동일)
  image_width/height = crop * scale
"""
import argparse
import numpy as np

CROP_KEEP_TOP = (
    "camera_front_left", "camera_front_right",
    "camera_rear_left", "camera_rear_right", "camera_rear_wide",
)


def patch(module, crop_w, crop_h, scale):
    """module.load_camera_infos 를 감싸서 내부 파라미터를 보정한다."""
    orig = module.load_camera_infos

    def wrapped(path):
        infos = orig(path)
        for cam, info in infos.items():
            W, H = int(info["image_width"]), int(info["image_height"])
            ox = (W - crop_w) // 2
            oy = 0 if cam in CROP_KEEP_TOP else H - crop_h
            K = np.array(info["cam_intrinsic"], dtype=np.float64).copy()
            K[0, 2] -= ox
            K[1, 2] -= oy
            K[:2, :] *= scale
            info["cam_intrinsic"] = K
            info["image_width"] = int(crop_w * scale)
            info["image_height"] = int(crop_h * scale)
            info["pre_crop"] = (ox, oy, crop_w, crop_h)
            info["pre_scale"] = scale
        return infos

    module.load_camera_infos = wrapped
    return module


def main():
    ap = argparse.ArgumentParser(description="전처리 데이터용 info pkl 생성")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--root-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-tag", default="vad_etri_pre")
    ap.add_argument("--crop-w", type=int, default=1920)
    ap.add_argument("--crop-h", type=int, default=1080)
    ap.add_argument("--scale", type=float, default=0.4)
    ap.add_argument("--ego-length", type=float, default=4.635)
    ap.add_argument("--ego-width", type=float, default=1.890)
    args = ap.parse_args()

    print("보정: crop %dx%d, scale %.2f -> %dx%d"
          % (args.crop_w, args.crop_h, args.scale,
             int(args.crop_w * args.scale), int(args.crop_h * args.scale)))

    if args.split == "train":
        import tools.data_converter.etri_vad_converter as C
        patch(C, args.crop_w, args.crop_h, args.scale)
        C.create_etri_infos(args.root_path, args.out_dir, args.extra_tag,
                            args.ego_length, args.ego_width)
    else:
        import tools.data_converter.etri_test_converter as C
        patch(C, args.crop_w, args.crop_h, args.scale)
        C.create_test_infos(args.root_path, args.out_dir, args.extra_tag)


if __name__ == "__main__":
    main()
