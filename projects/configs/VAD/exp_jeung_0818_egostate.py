# 자차 상태(ego status)를 플래너 입력에 추가한다.
#
# 베이스라인은 카메라 영상만으로 궤적을 예측한다. 자차가 지금 몇 m/s 로
# 움직이는지조차 모델에 알려주지 않는다. 리더보드 분석 결과 단순 등속 외삽만으로도
# 3위권 점수가 나오는데, 그 외삽에 필요한 재료를 모델이 아예 못 받고 있는 셈이다.
#
# 코드(VAD_head.py)에는 이미 구현돼 있고 config 에서 None 으로 꺼져 있을 뿐이라
# 코드 수정 없이 켤 수 있다.
_base_ = ['./VAD_etri_tiny_pre_split.py']

# ego_lcf_feat 9개 원소의 의미 (tools/data_converter/etri_vad_converter.py:271)
#   0 vx        1 vy        2 ax        3 ay        4 yaw_rate
#   5 ego_length  6 ego_width  7 |v|     8 (미사용, 항상 0)
#
# [0, 1, 4] 만 쓰는 이유:
#   - 2, 3 (가속도) : 테스트 컨버터는 현재 프레임의 다음 프레임이 없어서
#                     accel = zeros 로 채운다 (etri_test_converter.py:109).
#                     학습 때만 값이 들어오고 추론 때는 항상 0 -> 쓰면 해롭다.
#   - 5, 6         : 차량 제원 상수. 정보량 0.
#   - 8            : 항상 0.
#   - 7 (|v|)      : 0, 1 로부터 유도되는 값이라 중복. 필요하면 따로 실험.
#   VADv2 공식 config 도 [0, 1, 4] 를 쓴다 (VADv2/VADv2_config_voca4096.py:102).
model = dict(
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[0, 1, 4],
    ))

# RTX 3060 12GB 기준. 16GB 머신에서 돌린다면 6 으로 올린다.
fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=4, workers_per_gpu=4)
# iteration 단위 저장. epoch 단위(interval=1)로 두면 첫 저장까지 14시간이 걸려
# 그 전에 죽으면 전부 날아간다 (08-18 밤에 실제로 겪음).
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=3)
