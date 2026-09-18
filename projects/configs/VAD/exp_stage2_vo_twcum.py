"""T3 (9/18, A100): A-1 레시피 + 시간 가중(변위) + 누적 좌표 손실 동시 적용.

T1(exp_stage2_vo_tw)과 T2(exp_stage2_vo_cum)의 결합판. 둘 다 "지표와 학습 손실의 불일치"를
고치지만 거는 자리가 다르다 — T1 은 스텝 변위에 suffix-sum 가중치, T2 는 누적 좌표에 직접.
결합이 단독보다 나은지는 실측으로만 판정한다(RESULTS.md 8장: 결합 2건 기각 선례).
평가는 VO 끔(exp_full_tgtbev). 관문: val ≤ 0.288.
"""
_base_ = ['./exp_stage2_vo.py']

model = dict(pts_bbox_head=dict(
    plan_reg_time_weight=[36, 25, 14, 9, 4, 2],
    plan_reg_cum_weight=[11, 11, 5, 5, 2, 2],
))
