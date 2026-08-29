"""스크리닝 기준선 — 1/3 데이터 x 4 epoch 의 베이스라인.

스크리닝 실험들은 전체 학습이 아니라 이 값과 비교한다.
(1/3 x 4ep 조건에서의 0.55 가 얼마인지 모르면 판정 불가)
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

data_root = 'data/etri/'
data = dict(
    samples_per_gpu=5, workers_per_gpu=8,
    train=dict(ann_file=data_root + 'vad_etri_pre_infos_temporal_screen.pkl'))

# 스크리닝 표준은 10 epoch (docs/RESULTS.md #10 참고).
# train_watch 는 EPOCHS 미지정 시 config 값을 쓰므로 여기 값이 곧 표준이다.
# (8/28 base2 가 4ep 로 돌아간 사고의 재발 방지 - 의도적 단축은 EPOCHS=4 로)
total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=10)
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
