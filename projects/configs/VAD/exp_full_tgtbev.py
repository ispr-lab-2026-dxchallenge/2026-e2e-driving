"""본 학습 1호: 목표점->BEV 조건 (전체 데이터, 10 epoch). A100 담당.

스크리닝 채택 확정 1위(0.5306, RESULTS.md 3장)의 전체 데이터 버전.
1차 제출 후보를 가장 먼저 확보하는 것이 목적. 결합(tb150) 본 학습은
스크리닝 완주 확인 후 별도 진행.

- 전체 학습 데이터 (trainsplit, 337 시나리오)
- 10 epoch + epoch 별 체크포인트 저장 -> 검증 점수로 최선 선택 (6장 전략)
- fp16 학습 (스크리닝과 동일 설정. 부모 config 에는 없어서 여기 명시)
- 완주 후 ablation 필수 (보조 입력 모델)
"""
_base_ = ['./VAD_etri_tiny_pre_split.py']

model = dict(
    pts_bbox_head=dict(
        # 아래 둘은 부모 config 의 기본값과 같다(값 변경 없음). 운영국 2026-08-27 답변이
        # "베이스라인이 지원하지만 기본으로 쓰지 않는 구조"로 지목한 두 통로 —
        # 자차 과거 궤적/상태를 planner 에 직접 넣는 경로 — 가 꺼져 있음을 여기서 바로 보이게
        # 적어 둔다 (docs/JUDGE_NOTES.md 2장).
        ego_his_encoder=None,
        ego_lcf_feat_idx=None,
        transformer=dict(
            # 5초 목표점을 BEV query 생성 단계에만 조건 입력으로 더한다(공통 feature 단계).
            # planner 블록에는 목표점이 들어가지 않는다 — tests/test_target_point_flow.py
            use_target_cond=True,
            target_cond_drop=0.5,
        ),
    ))

total_epochs = 10
runner = dict(type='EpochBasedRunner', max_epochs=10)
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=1000, max_keep_ckpts=3)
