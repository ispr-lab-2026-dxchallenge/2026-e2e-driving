"""S1b (9/21 메인): 5초 목표점 앵커 꼬리 + 목표점 condition dropout 0.5 유지.

S0(dropout 0)의 조기 평가가 iter_10000 0.3474 / iter_15000 0.3556 으로 A-1(iter_10000 0.3351)보다
나빠지는 추세였다 → dropout 제거 자체가 꼬리에 해로울 수 있다(꼬리도 목표점 지름길에 기댄다).
그래서 dropout 은 원래대로 두고, 목표점이 지워진 배치에서는 앵커 손실을 건너뛴다
(VAD_transformer.last_goal_dropped 플래그). 앵커가 절반의 배치에만 걸리므로 가중치를 0.3 으로.
"""
_base_ = ['./exp_stage2_vo_goal10.py']

model = dict(pts_bbox_head=dict(
    goal_anchor_weight=0.3,
    transformer=dict(target_cond_drop=0.5),
))
