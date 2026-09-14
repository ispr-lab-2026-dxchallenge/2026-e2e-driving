"""본 학습 + VO 헤드 (A-2, 9/14): exp_full_tgtbev 레시피 그대로 + 영상 유래 자차 운동 추정을 planner 에 가산.

근거 (WORKFLOW.md 0장): 2026-09-08 운영국 "네트워크가 영상으로부터 직접 추론한 ego status 를
planner 에 사용하는 것은 영상에서 파생된 정보이므로 허용". 채점의 61% 가 0.5s·1.0s 점이라
현재 속도·회전 정보의 지렛대가 크다 (Hermite+운동학 수식 0.166, 우리 BEV 조건 모델 0.326).

구조 (VAD_head.py VO 블록): 현재/직전(-5 프레임) FPN 영상 특징만 입력받는 소형 CNN(0.56M) ->
s_hat=[dx/5, dy/1, dyaw/0.2] -> zero-init Linear(3->512) 를 ego_feats 에 가산(detach, 학습 중 p=0.3 드롭).
can_bus 가 주입된 BEV 를 거치지 않는다. 라벨(직전->현재 변위, yaw 변화)은 loss_vo 에서만 쓴다.
목표점 조건은 기존과 동일(BEV 형성 단계).

- 학습 큐: 직전 프레임(-5)을 항상 포함 (fix_prev_frame) -> 추론 오프셋 -10,-5,0 과 정합
- 정점 e5~e6 + 인접 평균, 재개 직후 epoch 제외 (기존 규칙)
- 완주 후 ablate_window (zero >= 1.2배, goaloff < zero) + vo 기여 확인
"""
_base_ = ['./exp_full_tgtbev.py']

model = dict(
    pts_bbox_head=dict(
        use_vo_head=True,
        vo_drop=0.3,
        vo_loss_weight=1.0,
        vo_detach=True,
    ))

data = dict(train=dict(fix_prev_frame=True))
