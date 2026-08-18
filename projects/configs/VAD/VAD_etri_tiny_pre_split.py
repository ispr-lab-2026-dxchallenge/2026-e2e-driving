"""train/val 분할을 적용한 기준 config.

VAD_etri_tiny_pre.py 와 동일하되 학습에 train split(337 시나리오)만 사용하고
val split(39 시나리오)을 검증에 쓴다. 분할 목록은 docs/val_split.txt.

실험 config 는 이 파일을 상속할 것:
    _base_ = ['./VAD_etri_tiny_pre_split.py']
"""
_base_ = ['./VAD_etri_tiny_pre.py']

data_root = 'data/etri/'
data = dict(
    train=dict(ann_file=data_root + 'vad_etri_pre_infos_temporal_trainsplit.pkl'),
    val=dict(ann_file=data_root + 'vad_etri_pre_infos_temporal_valsplit.pkl'),
    test=dict(ann_file=data_root + 'vad_etri_pre_infos_temporal_valsplit.pkl'),
)
