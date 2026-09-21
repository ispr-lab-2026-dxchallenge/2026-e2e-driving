"""S1 변형 (A100): 5초 목표점 앵커 가중치를 0.2 -> 0.5 로 올린다. 앵커 강도에 따른 반응을 본다."""
_base_ = ['./exp_stage2_vo_goal10.py']

model = dict(pts_bbox_head=dict(goal_anchor_weight=0.5))
