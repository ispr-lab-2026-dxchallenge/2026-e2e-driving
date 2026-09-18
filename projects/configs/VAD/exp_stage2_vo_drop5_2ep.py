"""B-3 의 2 epoch 판 (9/18 메인컴, seed 0): A-1 과 VO 주입 드롭 확률(0.3 → 0.5)만 다르다.

근거: A-1 의 이득은 "VO 추정치를 확률적으로 넣다 빼며 학습한 꼬리를 주입 없이 추론" 조합에서 나왔고
(RESULTS.md 10장), 추론이 '주입 없음' 이므로 학습에서 그 상태를 더 자주(50%) 보게 하면 나아질 수 있다.
4080 의 B-3(4ep) 와 달리 2ep 로 두어 A-1(seed 0, 2ep) 과 1:1 비교가 되게 한다. 결과 꼬리는 다중 실행
soup 의 멤버로도 쓴다. 평가는 VO 끔(exp_full_tgtbev config).
"""
_base_ = ['./exp_stage2_vo_drop5.py']

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
