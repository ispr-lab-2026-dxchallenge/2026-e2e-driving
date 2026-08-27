"""스크리닝용 1/3 학습 데이터셋을 만든다.

전체 학습(337 시나리오 x 10 epoch = 4~5일)은 실험 선별용으로 너무 비싸다.
1/3 시나리오 x 4 epoch (= 약 7~14시간) 으로 먼저 거른다 (docs/RESULTS.md 3관문).

- trainsplit(337개)에서 날짜별 층화 추출로 1/3 을 뽑는다 (SEED 고정)
- val 은 손대지 않는다 - 스크리닝끼리도 같은 39개 시나리오로 비교한다
- 모든 스크리닝 실험은 이 pkl 을 쓴다. 각자 다르게 뽑으면 비교 불가
"""
import argparse
import os
import random
from collections import defaultdict

import mmcv

SEED = 42
FRAC = 1.0 / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/workspace/VAD/data')
    ap.add_argument('--src', default='etri/vad_etri_pre_infos_temporal_trainsplit.pkl')
    args = ap.parse_args()

    src = os.path.join(args.data_root, args.src)
    d = mmcv.load(src)
    infos, meta = d['infos'], d['metadata']

    scenarios = sorted(set(i['scene_token'] for i in infos))
    by_date = defaultdict(list)
    for s in scenarios:
        by_date[s.split('-')[0]].append(s)

    rng = random.Random(SEED)
    keep = []
    for date in sorted(by_date):
        group = sorted(by_date[date])
        k = max(1, int(len(group) * FRAC + 0.5))
        keep += rng.sample(group, k)
    keep = set(keep)

    sel = [i for i in infos if i['scene_token'] in keep]
    lanes = meta.get('map_lanes', {})
    out = src.replace('_trainsplit.pkl', '_screen.pkl')
    mmcv.dump(dict(infos=sel, metadata=dict(
        version=meta.get('version', 'etri-v1.0'),
        map_lanes={k: v for k, v in lanes.items() if k in keep})), out)
    print('스크리닝셋: %d 시나리오 / %d 샘플  -> %s'
          % (len(keep), len(sel), out))
    import hashlib
    joined = chr(10).join(sorted(keep))
    print('시나리오 목록 md5(팀 대조용):',
          hashlib.md5(joined.encode()).hexdigest()[:10])


if __name__ == '__main__':
    main()
