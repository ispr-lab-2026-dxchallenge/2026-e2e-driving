"""VO 헤드 2단계 (A-1, 9/14): 메인컴 e5+e6 평균 트렁크(검증 0.3260)를 동결하고 VO 헤드 + planner 꼬리만 학습.

- load_from = work_dirs/exp_full_tgtbev/soup_e5e6.pth (tools/avg_ckpt.py 로 만든 가중치 평균; 서브컴
  exp_full_tgtbev_main/epoch_22.pth 와 동일, 팀 공유 폴더의 soup_main_e5e6.pth)
- train_only (tools/train.py): vo_* (VO 헤드, 주입층) 전부 + planner 꼬리(ego_fut_decoder, ego_agent/map
  decoder, ego_query, pos mlp) 만 학습. 트렁크·검출·맵·lane_encoder 동결 -> 회귀 출력 하방 0.
- lr: vo_* 1.0배, planner 꼬리 0.1배 (이미 수렴한 꼬리를 흔들지 않기 위해).
- vo_to_planner 가 zero-init 이라 학습 시작 시점의 궤적 출력은 soup 와 동일해야 한다 (동결 검증:
  같은 ckpt 를 exp_full_tgtbev config 로 평가하면 0.3260 재현).
- 관문 1 (+2h, iter 6000 ckpt): tools/eval_vo.py 로 VO 오차 확인 (|v| RMSE < 0.6 m/s).
"""
_base_ = ['./exp_full_tgtbev_vo.py']

load_from = 'work_dirs/exp_full_tgtbev/soup_e5e6.pth'
train_only = [
    'pts_bbox_head.vo_',
    'pts_bbox_head.ego_fut_decoder', 'pts_bbox_head.ego_agent_decoder', 'pts_bbox_head.ego_map_decoder',
    'pts_bbox_head.ego_query', 'pts_bbox_head.ego_agent_pos_mlp', 'pts_bbox_head.ego_map_pos_mlp',
]
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.0),
            'img_neck': dict(lr_mult=0.0),
            'pts_bbox_head': dict(lr_mult=0.0),
            'pts_bbox_head.vo_': dict(lr_mult=1.0),
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
