"""B-3 (9/16): A-1 레시피에서 VO 주입 드롭을 0.3 → 0.5 로 올리고 4 epoch.

근거: A-1 의 이득은 "VO 추정치를 확률적으로 넣다 빼며 학습한 꼬리를 주입 없이 추론" 조합에서
나왔다(RESULTS.md 10장). 추론이 '주입 없음' 이므로 학습에서 그 상태를 더 자주(50%) 보게 하고,
추세(0.310→0.298)가 2 epoch 에서 아직 내려가던 중이라 4 epoch 으로 늘린다.
평가는 VO 를 끄고(exp_full_tgtbev config). 관문: 전체 val ≤ 0.288 이면 채택.
"""
_base_ = ['./exp_stage2_vo.py']

model = dict(pts_bbox_head=dict(vo_drop=0.5))

total_epochs = 4
runner = dict(type='EpochBasedRunner', max_epochs=4)
