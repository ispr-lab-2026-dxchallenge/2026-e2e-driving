"""planner 꼬리 미세조정 (9/15): A-1 진단에서 얻은 이득의 순수 재현. VO 없음.

A-1(exp_stage2_vo) 의 epoch_2 를 VO 를 끄고 평가하면 0.2979 (soup 0.3260 대비 −0.028), VO 를 켜면
0.4117. 즉 이득은 VO 주입이 아니라 "학습 큐를 추론 오프셋(-10,-5,0)에 맞추고(fix_prev_frame)
동결 트렁크 위에서 planner 꼬리만 낮은 lr 로 2 epoch 미세조정" 에서 나왔다. 이 config 는 그 요소만
남긴 것이다. 구조·입력은 제출 라인(exp_full_tgtbev)과 완전히 동일하므로 규정 영향 없음.

- load_from = 메인 e5+e6 평균(검증 0.3260), 트렁크·검출·맵·lane_encoder 동결(train_only)
- 학습 대상: ego_fut_decoder, ego_agent/map decoder, ego_query, pos mlp (lr 0.1배 = 2e-5)
- 2 epoch, iter ckpt 2500 간격 보존 → 전 체크포인트 검증 후 최선 선택
"""
_base_ = ['./exp_full_tgtbev.py']

data = dict(train=dict(fix_prev_frame=True))

load_from = 'work_dirs/exp_full_tgtbev/soup_e5e6.pth'
train_only = [
    'pts_bbox_head.ego_fut_decoder', 'pts_bbox_head.ego_agent_decoder', 'pts_bbox_head.ego_map_decoder',
    'pts_bbox_head.ego_query', 'pts_bbox_head.ego_agent_pos_mlp', 'pts_bbox_head.ego_map_pos_mlp',
]
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0),
            'pts_bbox_head.ego_fut_decoder': dict(lr_mult=0.1),
            'pts_bbox_head.ego_agent_decoder': dict(lr_mult=0.1),
            'pts_bbox_head.ego_map_decoder': dict(lr_mult=0.1),
            'pts_bbox_head.ego_query': dict(lr_mult=0.1),
            'pts_bbox_head.ego_agent_pos_mlp': dict(lr_mult=0.1),
            'pts_bbox_head.ego_map_pos_mlp': dict(lr_mult=0.1),
        }))

total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(
    type='IterSafeCheckpointHook', by_epoch=False, interval=2500, max_keep_ckpts=20)
