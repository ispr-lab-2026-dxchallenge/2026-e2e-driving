"""차선 이탈 제약(loss_plan_bound)을 켠다.

주최측 ETRI config 는 이 손실의 가중치를 0.0 으로 꺼뒀다.
원본 VAD(nuScenes) 는 1.0 이다 (VAD_base_e2e.py:288).

    loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=0.0, ...)   <- ETRI
    loss_plan_bound=dict(type='PlanMapBoundLoss', loss_weight=1.0, ...)   <- 원본

무엇을 하는 손실인가
--------------------
예측한 자차 궤적이 **모델이 인식한 차선 경계**에 dis_thresh(1.0m) 이내로
붙으면 벌점을 준다. 여기서 차선은 정답이 아니라 **모델이 영상에서 뽑아낸
map 예측**이다 (map_thresh=0.5 이상 신뢰도만 사용).

그래서 이 손실은 "영상에서 차선을 제대로 읽고, 그 안에 머무는 궤적을 내라"를
직접 가르친다. 두 가지를 동시에 노린다.

  1. L2 개선 - 차선을 벗어나는 예측이 줄어든다
  2. 영상 의존도 상승 - 차선 인식이 안 되면 이 손실을 줄일 수 없다

2 번이 특히 중요하다. 자차 상태 실험(exp_jeung_0818_egostate)이 영상 기여도
1.13 배로 규정 §3 위반 소지가 생겨 폐기됐는데, 이 손실은 반대 방향으로 작용한다.

주의할 점
---------
학습 초반에는 map 예측 신뢰도가 낮아 이 손실이 0 근처에 머문다. 정상이다.
1~2 epoch 지나 값이 올라오는지 확인할 것. 끝까지 정확히 0 이면 ETRI 데이터에서
map 예측이 threshold 를 못 넘는다는 뜻이므로, 그때는 map_thresh 를 낮춰본다.

기타 설정은 전부 베이스라인과 동일하다. 손실 하나만 켠 것이므로 직접 비교된다.
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

# 가중치만 덮어쓴다. type, dis_thresh, lane_bound_cls_idx, point_cloud_range 는
# base 의 값이 그대로 유지된다 (mmcv Config 는 중첩 dict 를 병합한다).
model = dict(
    pts_bbox_head=dict(
        loss_plan_bound=dict(loss_weight=1.0),
    ))

fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=5, workers_per_gpu=8)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
