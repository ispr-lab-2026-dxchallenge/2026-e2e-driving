"""스크리닝: 차선 이탈 제약(loss_plan_bound) 강화.

VAD 원논문의 핵심 기여인 벡터화 계획 제약 3종 중 ETRI 배포판이 0.0 으로 꺼둔
것을 켜고, 발동 문턱도 넓힌다.

  원논문        loss_weight 1.0, dis_thresh 1.0
  ETRI 배포판   loss_weight 0.0  (꺼짐)
  이 실험       loss_weight 1.0, dis_thresh 1.5

dis_thresh 를 1.5 로 키운 이유: 자차~경계선 거리 중앙값이 2.09m 라 1.0m 기준으론
발동이 드물다 (실측 손실이 전체의 0.06%). 1.5m 면 견제가 실제로 작동하기 시작한다.

목적 두 가지:
  1. L2 개선 - 차선을 벗어나는 예측 감소
  2. planner 를 "영상에서 인식한" 차선에 결박 - 비전 결합 강화.
     (planner 지름길 학습을 견제하는, 규정상 완전 합법인 유일한 구조적 장치)

비교 대상: exp_screen_base (같은 1/3 데이터, 같은 4 epoch)
"""
_base_ = ['./exp_screen_base.py']

model = dict(
    pts_bbox_head=dict(
        loss_plan_bound=dict(loss_weight=1.0, dis_thresh=1.5),
    ))
