# -*- coding: utf-8 -*-
"""train / val 분할 생성.

- 시나리오 단위로 나눈다 (프레임 단위로 섞으면 같은 장면이 양쪽에 들어가 성능이 부풀려진다)
- 수집 날짜별로 비례 배분한다 (특정 날짜를 통째로 빼면 그 구간을 학습에서 못 본다)
- 시드를 고정해 누가 돌려도 같은 결과가 나오게 한다
"""
import argparse
import os
import random
from collections import defaultdict

import mmcv

SEED = 42
RATIO = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/workspace/VAD/data")
    ap.add_argument("--pkl", default="etri/vad_etri_pre_infos_temporal_train.pkl")
    ap.add_argument("--out-list", default="/workspace/VAD/docs/val_split.txt")
    args = ap.parse_args()

    src = os.path.join(args.data_root, args.pkl)
    scen_dir = os.path.join(args.data_root, "etri_pre/train")
    scenarios = sorted(os.listdir(scen_dir))

    by_date = defaultdict(list)
    for s in scenarios:
        by_date[s.split("-")[0]].append(s)

    rng = random.Random(SEED)
    val = []
    print("날짜별 배분")
    print("-" * 46)
    for date in sorted(by_date):
        group = sorted(by_date[date])
        k = int(len(group) * RATIO + 0.5)          # 표준 반올림
        if k == 0 and len(group) >= 5:
            k = 1
        picked = sorted(rng.sample(group, k)) if k else []
        val += picked
        print("  %s   전체 %3d → val %2d" % (date, len(group), k))
    val = set(val)
    print("-" * 46)
    print("  전체 %d 시나리오  →  train %d / val %d (%.1f%%)"
          % (len(scenarios), len(scenarios) - len(val), len(val),
             100 * len(val) / len(scenarios)))

    os.makedirs(os.path.dirname(args.out_list), exist_ok=True)
    with open(args.out_list, "w") as f:
        f.write("# train/val 분할 - 시나리오 단위, 날짜별 비례 %d%%, seed=%d\n"
                % (int(RATIO * 100), SEED))
        f.write("# 이 목록을 바꾸면 이전 실험과 비교할 수 없게 된다\n")
        for s in sorted(val):
            f.write(s + "\n")
    print("\n목록 저장: %s" % args.out_list)

    print("\npkl 분할 중 (1.4GB 로딩)...")
    d = mmcv.load(src)
    infos, meta = d["infos"], d["metadata"]
    tr = [i for i in infos if i["scene_token"] not in val]
    va = [i for i in infos if i["scene_token"] in val]
    lanes = meta.get("map_lanes", {})

    base = os.path.join(args.data_root, "etri", "vad_etri_pre_infos_temporal")
    mmcv.dump(dict(infos=tr, metadata=dict(
        version=meta.get("version", "etri-v1.0"),
        map_lanes={k: v for k, v in lanes.items() if k not in val})),
        base + "_trainsplit.pkl")
    mmcv.dump(dict(infos=va, metadata=dict(
        version=meta.get("version", "etri-v1.0"),
        map_lanes={k: v for k, v in lanes.items() if k in val})),
        base + "_valsplit.pkl")

    print("  train : %6d 샘플 / %3d 시나리오" % (len(tr), len(set(i["scene_token"] for i in tr))))
    print("  val   : %6d 샘플 / %3d 시나리오" % (len(va), len(set(i["scene_token"] for i in va))))
    assert not (set(i["scene_token"] for i in tr) & set(i["scene_token"] for i in va)), "겹침!"
    print("  겹치는 시나리오 없음 확인")


if __name__ == "__main__":
    main()
