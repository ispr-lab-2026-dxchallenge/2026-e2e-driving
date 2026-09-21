"""exp_stage2_vo_goal10 체크포인트의 평가·제출용 config (VO 없음, 목표점 -> BEV 조건화 동일).

exp_full_tgtbev 와 차이는 꼬리 마지막 Linear 가 10스텝(60출력)이라는 것뿐이다. 추론에서는
ego_fut_preds 로 앞 6스텝만 나가고 뒤 4스텝(ego_fut_preds_ext)은 학습 라벨용이라 쓰이지 않는다.
"""
_base_ = ['./exp_full_tgtbev.py']

model = dict(pts_bbox_head=dict(goal_anchor_ts=10))
