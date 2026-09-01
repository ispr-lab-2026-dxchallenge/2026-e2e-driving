"""스크리닝 결합 2호: 목표점 힌트 + 경로 사전(앵커).

1호 결합(tb150 = 목표점 x 해상도150)은 기각됐다 (0.7228 - 양쪽 단독보다
나쁨, 학습 내내 +-0.3 요동. RESULTS.md 8장). 이번 조합이 다른 이유:
해상도는 목표점과 같은 곳(지도 형성)을 건드렸지만, 앵커는 출력단(경로
선택/보정)에 살아서 겹침이 없다. 앵커 브랜치의 영상 의존(1.55배)이
목표점 모델의 얇은 ablation 여유(1.22배)를 보강하는 효과도 기대.

실행 전제: data/etri/anchors_k4096.npy (tools/make_anchors.py, md5 7d055ecaf8)
ABLATION=1 필수. 비교 대상: tgtbev 단독 0.5306.
"""
_base_ = ['./exp_screen_tgtbev.py']

model = dict(
    pts_bbox_head=dict(
        use_anchor_head=True,
        anchor_file='data/etri/anchors_k4096.npy',
    ))
