"""본 학습 2차: 목표점->BEV 조건, 7 epoch 완전 어닐링 (9/8 결정).

근거 (RESULTS.md 10장): 10 epoch 코사인 스케줄에서 epoch 6(lr 아직 1e-4)이 정점이고
7부터 과적합. 스크리닝에서도 4ep 완주가 10ep 완주보다 좋았다. 코사인이 7 epoch 에서
끝나도록 당겨 정점 근처에서 lr 이 충분히 식게 한다. 구조·입력은 exp_full_tgtbev 와
완전히 같다(규정 영향 없음).

- iter 체크포인트를 40개까지 보존 -> epoch 5~7 구간의 중간 후보도 검증 점수로 선택
  (1000 iter 간격, 258MB x 40 = 약 10GB)
- 완주 후 ablation 필수 (ABLATION=1)
"""
_base_ = ['./exp_full_tgtbev.py']

total_epochs = 7
runner = dict(type='EpochBasedRunner', max_epochs=7)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=40)
