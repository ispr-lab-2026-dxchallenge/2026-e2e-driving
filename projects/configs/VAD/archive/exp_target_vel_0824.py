"""5초 목표점 + 자차 속도. exp_target_0824 와 한 쌍으로 돌린다.

타 팀이 공개한 조합이 target + status + command 이고, val 0.192 에
영상 기여도 4.83 배를 유지했다. 우리도 같은 조합을 시험한다.

핵심 질문
---------
자차 속도 단독(exp_jeung_0818_egostate)은 영상 기여도 1.13 배로 실패했다.
**목표점과 함께 주면 달라지는가?**

  달라진다면  -> 이 조합을 채택. 타 팀 수준(0.19)을 노린다
  같다면      -> 목표점만 쓰는 exp_target_0824 로 간다

두 실험 다 학습 후 반드시 ablate_image.py 를 돌린다. 악화 2 배 미만이면
규정 §3 위반 소지가 있어 제출하지 않는다.

인덱스: 11,12 = 목표점(정규화) / 0,1 = vx,vy / 4 = yaw_rate
가속도(2,3)는 넣지 않는다. 테스트 컨버터가 항상 0 으로 채운다
(etri_test_converter.py:109).
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

model = dict(
    pts_bbox_head=dict(
        ego_lcf_feat_idx=[11, 12, 0, 1, 4],
    ))

fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=5, workers_per_gpu=8)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
