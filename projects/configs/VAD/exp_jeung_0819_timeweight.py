"""평가 지표에 맞춰 플래닝 손실의 시간 가중치를 조정한다.

지표는 waypoint 를 균등하게 보지 않는다 (2026-08-19 운영국 확정).

    L2_avg = (1초 L2 + 2초 L2 + 3초 L2) / 3
    1초 L2 = 앞 2개 ADE,  2초 = 앞 4개,  3초 = 6개

이걸 풀면 누적 좌표 k 의 실효 가중치는 [11,11,5,5,2,2]/36 이고,
0.5초와 1.0초 두 점이 점수의 61% 를 차지한다.

그런데 모델이 내놓는 ego_fut_preds 는 누적 좌표가 아니라 스텝 변위다.
변위 d_i 의 오차는 i 이후 모든 누적 좌표에 전파되므로, 변위에 걸어야 할
가중치는 위 값의 suffix sum 이다.

    d_1 : 11+11+5+5+2+2 = 36
    d_2 :    11+5+5+2+2 = 25
    d_3 :       5+5+2+2 = 14
    d_4 :         5+2+2 =  9
    d_5 :           2+2 =  4
    d_6 :             2 =  2

내부에서 평균 1 로 정규화되므로 loss_weight 는 건드릴 필요가 없다.

구조 변경이 전혀 없고 손실 가중치만 바뀌므로, 베이스라인과 직접 비교 가능하다.
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

model = dict(
    pts_bbox_head=dict(
        plan_reg_time_weight=[36, 25, 14, 9, 4, 2],
    ))

fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=5, workers_per_gpu=5)
# IterSafeCheckpointHook 을 쓰는 이유는 custom_hooks.py 의 docstring 참조.
# 요약: mmcv 기본 훅은 epoch 중간에 저장해도 메타에 'epoch+1' 을 적어서,
#      재개하면 그 epoch 을 통째로 건너뛴다. 재개 1회당 epoch 을 하나 까먹는다.
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
