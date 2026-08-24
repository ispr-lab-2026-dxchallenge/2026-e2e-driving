"""5초 목표점을 goal conditioning 으로 넣는다.

왜 하는가
---------
가이드라인 14쪽의 입력 조건은 "카메라 6채널 + 5초 타깃 포인트" 인데,
배포된 컨버터가 목표점을 읽지 않고 버리고 있었다. 절반만 쓰고 있었던 셈이다.

  테스트 clip  ego_pose.parquet  frame = [-30 ... 0, 50]   <- 50 이 목표점
  학습 시나리오 프레임 0~299 전부 +50 확보 가능

규정(§3-1)이 "내비게이션 목적지와 동일한 성격의 조건 입력"으로 명시적으로
허용한 정보다.

왜 자차 속도와 다른가
---------------------
exp_jeung_0818_egostate (속도 vx,vy,yaw_rate 만 추가) 는 영상 기여도가
1.13 배로 떨어져 폐기했다. 속도를 주면 "하던 대로 계속"이 곧 정답이 되어
영상을 볼 이유가 사라진다.

목표점은 "어디로" 만 알려준다. "어떻게 거기까지 갈지" — 차선을 지키고
앞차를 피하는 것 — 는 여전히 영상에서 읽어야 한다.
타 팀 사례가 target+status+command 를 쓰면서 영상 기여도 4.83 배를
유지한 것도 이 차이로 설명된다.

인덱스
------
  0 vx   1 vy   2 ax   3 ay   4 yaw_rate
  5 length  6 width  7 |v|  8 (미사용)
  9 target_x   10 target_y        (m, 원시)
 11 target_x/100  12 target_y/100 (정규화)   <- 이걸 쓴다

정규화를 쓰는 이유: 목표점은 40~150m 인데 ego_fut_decoder 는 정규화 없는
Linear 스택이고 다른 입력(트랜스포머 특징)은 약 N(0,1) 이다. 원시값을 넣으면
첫 Linear 가 100 배 큰 입력을 받는다.

※ 이 config 는 목표점이 들어간 pkl 을 요구한다. 없으면 재생성할 것:
     python convert_pre.py --split train --root-path .../etri_pre/train \
         --out-dir .../data/etri --extra-tag vad_etri_pre
     python convert_pre.py --split test  ... --extra-tag vad_etri_pre
     python make_split.py
   gt_ego_lcf_feat 길이가 13 이면 정상이다.
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[11, 12],
    ))

fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=5, workers_per_gpu=8)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
