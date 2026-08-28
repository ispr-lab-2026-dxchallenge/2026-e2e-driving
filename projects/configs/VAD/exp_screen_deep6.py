"""스크리닝: 트랜스포머 깊이 3층 -> 6층 (base 구조의 "깊이 축"만 분리).

full base(BEV 200 + 층 6 + queue 4)는 A100 40GB 에서 OOM 이고, 설령
스크리닝을 이겨도 40 epoch 장기 학습을 돌릴 하드웨어가 없다.
그래서 base 를 변수별로 쪼갠다:
  격자 축(BEV 150)  -> 서브컴에서 시험 중
  깊이 축(층 6)     -> 이 실험. BEV 100 유지라 메모리 부담이 작다
둘 다 이기면 "BEV150 + 층6" 조합이 장기 학습 후보가 된다 (실측상 탑재 가능).

인코더/디코더/맵 디코더 모두 3 -> 6 (base 와 동일한 축).
비교 대상: exp_screen_base (1/3 데이터, 10 epoch, 기준 0.8369)
"""
_base_ = ['./exp_screen_base.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            encoder=dict(num_layers=6),
            decoder=dict(num_layers=6),
            map_decoder=dict(num_layers=6),
        ),
    ))
