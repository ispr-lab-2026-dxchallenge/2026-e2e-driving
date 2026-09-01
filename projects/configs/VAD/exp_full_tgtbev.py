"""본 학습 1호: 목표점->BEV 조건 (전체 데이터, 10 epoch). A100 담당.

스크리닝 채택 확정 1위(0.5306, RESULTS.md 3장)의 전체 데이터 버전.
1차 제출 후보를 가장 먼저 확보하는 것이 목적. 결합(tb150) 본 학습은
스크리닝 완주 확인 후 별도 진행.

- 전체 학습 데이터 (trainsplit, 337 시나리오)
- 10 epoch + epoch 별 체크포인트 저장 -> 검증 점수로 최선 선택 (6장 전략)
- fp16 학습 (스크리닝과 동일 설정. 부모 config 에는 없어서 여기 명시)
- 완주 후 ablation 필수 (보조 입력 모델)
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            use_target_cond=True,
            target_cond_drop=0.5,
        ),
    ))

total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=10)
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
