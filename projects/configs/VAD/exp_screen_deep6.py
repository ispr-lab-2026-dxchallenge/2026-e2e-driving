"""스크리닝: 트랜스포머 깊이 3층 -> 6층 (base 의 "깊이 축"만 분리).

배경: base 구조 통째(exp_screen_basearch: BEV 200 + 6층 + queue 4)는
A100 40GB 에서 OOM 이 났다. 들어가게 배치를 낮추면 3~4일이 걸려
스크리닝으로 부적합하다. 그래서 base 를 변수별로 쪼갠다:

  격자 축 (BEV 100->150)  : exp_screen_bev150  (서브컴에서 진행 중)
  깊이 축 (층수 3->6)      : 이 실험            (BEV 100 유지, 메모리 문제 없음)

주최측 base config 가 인코더/디코더/맵디코더를 모두 6층으로 두므로
세 곳을 함께 6으로 올린다. 두 축이 각각 이기면 장기 학습에서 합친다.

비교 대상: exp_screen_base (1/3 데이터, 10 epoch, 기준값 0.8369)
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
