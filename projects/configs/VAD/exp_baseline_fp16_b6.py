"""첫 베이스라인 - 기준점 확보용.

구조 변경 없이 전처리 데이터 + fp16 으로만 돌린다.
이후 모든 실험은 이 결과와 비교한다.
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=6, workers_per_gpu=6)

# ---------------------------------------------------------------------------
# 체크포인트는 반드시 iteration 단위로 저장한다.
#
# 처음엔 interval=1 로 뒀는데, 이건 "1 epoch 마다"라는 뜻이다.
# 1 epoch 이 약 11시간이라 그 안에 죽으면 남는 게 없다.
# 실제로 08-18 밤 iteration 2000 (5시간 20분) 에서 CUDA 오류로 죽었을 때
# 체크포인트가 하나도 없어서 전부 날렸다.
#
# by_epoch=False + interval=1000 이면 약 38분마다 저장된다.
# 258MB x 3 = 774MB 로 용량 부담도 없다.
# ---------------------------------------------------------------------------
# IterSafeCheckpointHook 을 쓰는 이유는 custom_hooks.py 의 docstring 참조.
# 요약: mmcv 기본 훅은 epoch 중간에 저장해도 메타에 'epoch+1' 을 적어서,
#      재개하면 그 epoch 을 통째로 건너뛴다. 재개 1회당 epoch 을 하나 까먹는다.
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
