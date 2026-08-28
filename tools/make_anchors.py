# -*- coding: utf-8 -*-
"""트랙 V: 학습 데이터 궤적으로 앵커(궤적 어휘) 생성.

[규정 근거 - 2026-08-28 운영국 답변]
  "학습 데이터로부터 생성한 고정 앵커 + 비전 기반 선택/보정은 허용"
앵커 후보는 오직 '학습 split'의 ego 미래 궤적에서만 만든다.
테스트/평가 clip 은 어떤 형태로도 사용하지 않는다.

입력 : data/etri/vad_etri_pre_infos_temporal_trainsplit.pkl (학습 split 전용)
출력 : data/etri/anchors_k4096.npy  (4096, 6, 2) float32
       ego 좌표계 누적 waypoint (0.5s 간격 6개)

검증 : valsplit 에 대해 oracle L2 (가장 가까운 앵커를 골랐을 때의 상계)를
       대회 지표(L2_avg = 앞2/앞4/전체6 ADE 평균)로 출력한다.

사용법 (컨테이너 안):
  python tools/make_anchors.py
"""
import hashlib
import os
import pickle

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '4')

K = 4096
TRAIN_PKL = 'data/etri/vad_etri_pre_infos_temporal_trainsplit.pkl'
VAL_PKL = 'data/etri/vad_etri_pre_infos_temporal_valsplit.pkl'
OUT = 'data/etri/anchors_k%d.npy' % K


def load_full_futures(path):
    """마스크 6개가 전부 1인 프레임만 골라 누적 waypoint (N,6,2)로 반환."""
    d = pickle.load(open(path, 'rb'))
    infos = d['infos'] if isinstance(d, dict) else d
    trajs = []
    for info in infos:
        m = np.asarray(info['gt_ego_fut_masks'])
        if m.shape != (6,) or m.min() < 1:
            continue
        t = np.asarray(info['gt_ego_fut_trajs'], dtype=np.float64)  # (6,2) 스텝 오프셋
        trajs.append(np.cumsum(t, axis=0))
    return np.asarray(trajs), len(infos)


def l2_avg_to_anchors(gt, anchors):
    """샘플별 oracle L2_avg (가장 좋은 앵커 기준). gt (N,6,2), anchors (K,6,2)."""
    best = np.empty(len(gt))
    a = anchors[None]  # (1,K,6,2)
    for i in range(0, len(gt), 256):
        g = gt[i:i + 256, None]              # (b,1,6,2)
        d = np.linalg.norm(g - a, axis=-1)   # (b,K,6)
        ade2 = d[..., :2].mean(-1)
        ade4 = d[..., :4].mean(-1)
        ade6 = d.mean(-1)
        l2 = (ade2 + ade4 + ade6) / 3.0      # (b,K)
        best[i:i + len(g)] = l2.min(-1)
    return best


def main():
    tr, n_all = load_full_futures(TRAIN_PKL)
    print('학습 프레임 %d개 중 완전한 6스텝 미래 보유: %d개' % (n_all, len(tr)))

    from sklearn.cluster import MiniBatchKMeans
    x = tr.reshape(len(tr), -1).astype(np.float32)  # (N,12)
    km = MiniBatchKMeans(n_clusters=K, batch_size=10000, n_init=3,
                         max_iter=300, random_state=0)
    km.fit(x)
    anchors = km.cluster_centers_.reshape(K, 6, 2).astype(np.float32)

    np.save(OUT, anchors)
    md5 = hashlib.md5(anchors.tobytes()).hexdigest()[:10]
    print('저장: %s  shape=%s  md5=%s' % (OUT, anchors.shape, md5))

    # 검증 1: 학습셋 자체 양자화 오차
    tr_or = l2_avg_to_anchors(tr[:20000], anchors)
    print('train(2만 표본) oracle L2_avg: %.4f' % tr_or.mean())
    # 검증 2: valsplit oracle (스크리닝 기준과 같은 지표)
    va, _ = load_full_futures(VAL_PKL)
    va_or = l2_avg_to_anchors(va, anchors)
    print('val(%d) oracle L2_avg: %.4f  (리더보드 1위 0.1305 대비)' % (len(va), va_or.mean()))


if __name__ == '__main__':
    main()
