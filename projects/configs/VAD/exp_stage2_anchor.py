"""2단계 앵커 헤드 (9/10 결정): 최선 트렁크를 동결하고 앵커 선택·보정 브랜치만 학습.

근거 (RESULTS.md 8장 ta 기각 진단 + 9/9 코드 추적): 결합 학습에서는 앵커 분류 CE 가
회귀 손실의 수십 배라 기울기를 독점해 트렁크가 후퇴했다. 트렁크(메인컴 e5+e6 가중치
평균, 검증 0.3260)를 통째로 동결하면 회귀 출력은 정의상 보존되고(같은 ckpt 를
exp_full_tgtbev 로 평가하면 0.3260 이 그대로 나와야 한다 = 동결 검증), 앵커 출력이
그보다 좋은지만 본다. 하방 0.

규정 (WORKFLOW.md 0장): "학습 데이터로 만든 고정 앵커 + 영상 기반 선택·보정" 은
운영국 명시 허용. 앵커 헤드 입력은 ego_feats(BEV 유래)와 앵커 좌표뿐, 목표점은
BEV 조건으로만 들어간다(기존 제출 라인과 동일).

- 동결: train_only(tools/train.py 가 읽음, 그 외 파라미터 requires_grad=False) +
  paramwise lr_mult=0 이중 안전장치
- 다른 손실은 그대로 (동결 파라미터는 기울기가 없어 grad clip 에도 안 잡힌다)
- 판정: 앵커 출력 L2 가 0.3260 보다 0.03 이상 낮으면 GO, 0.015 미만이면 폐기
"""
_base_ = ['./exp_full_tgtbev.py']

model = dict(
    pts_bbox_head=dict(
        use_anchor_head=True,
        anchor_file='data/etri/anchors_k4096.npy',
        anchor_cls_weight=0.5,
        anchor_reg_weight=1.0,
        # 다른 손실은 그대로 둔다: 트렁크가 동결(requires_grad=False)이라 기울기를 못 만들고,
        # 검출/맵 손실 가중치는 매처 가중치와 같아야 한다는 단언이 있어 0 으로 못 바꾼다.
    ))

# 트렁크 = 메인컴 시드 e5+e6 가중치 평균 (tools/avg_ckpt.py, 검증 0.3260)
load_from = 'work_dirs/exp_full_tgtbev/soup_e5e6.pth'
# 이 접두를 포함한 파라미터만 학습 (anchor_cls_branch / anchor_emb / anchor_refine_branch)
train_only = ['pts_bbox_head.anchor']
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0),
            'pts_bbox_head.anchor': dict(lr_mult=1.0),
        }))

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=2500, max_keep_ckpts=20)
