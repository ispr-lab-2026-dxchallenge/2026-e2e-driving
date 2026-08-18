"""첫 베이스라인 - 기준점 확보용.

구조 변경 없이 전처리 데이터 + fp16 + batch 6 으로만 돌린다.
이후 모든 실험은 이 결과와 비교한다.
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=6, workers_per_gpu=6)

# 체크포인트가 258MB 라 기본값 60 이면 15GB 가 쌓인다
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
