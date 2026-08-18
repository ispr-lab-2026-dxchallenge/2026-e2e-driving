_base_ = ['./VAD_etri_tiny.py']
fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=4)
