"""스크리닝 트랙 V: 학습 데이터 고정 앵커(K=4096) + 비전 선택/보정.

2026-08-28 운영국 확정: "학습 데이터로부터 생성한 고정 앵커 + 비전 기반
선택 및 보정은 허용". 앵커는 tools/make_anchors.py 로 생성한다
(학습 split 전용, md5=7d055ecaf8, val oracle L2_avg 0.0765).

기존 회귀 브랜치와 손실은 그대로 두고(표현 학습 유지), 앵커 분류 CE +
할당 앵커 잔차 L1 을 추가한다. 평가/제출 시에는 앵커 기반 궤적이
ego_fut_preds 자리를 대체한다.

실행 전 준비: 컨테이너에서 python tools/make_anchors.py (약 5분)
비교 대상: exp_screen_base (기준값 0.8369)
"""
_base_ = ['./exp_screen_base.py']

model = dict(
    pts_bbox_head=dict(
        use_anchor_head=True,
        anchor_file='data/etri/anchors_k4096.npy',
    ))
