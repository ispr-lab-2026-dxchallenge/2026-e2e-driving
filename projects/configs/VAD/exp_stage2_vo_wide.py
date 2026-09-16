"""B-2 (9/16): A-1 레시피(동결 트렁크 + VO 확률 주입)를 planner 꼬리 + 모션/상호작용 디코더까지 넓혀 적용.

근거 (RESULTS.md 10장 VO 실험): A-1 은 VO 추정치를 p=0.3 으로 넣다 빼며 planner 꼬리를 학습했고,
추론에서 VO 를 빼면 0.2979(soup 0.3260 대비 −0.028). VO 없이 같은 미세조정만 하면 0.413 으로 실패.
→ 이득이 "확률적 주입 = 정규화" 라면, 다듬는 범위를 넓히면 더 커질 수 있다. 검출·맵·BEV(인식)는
그대로 동결한다(제출 구조 동일, 규정 영향 없음).

- 추가 학습 대상: motion_decoder, motion_map_decoder, motion_mode_query, lane_encoder, pos_mlp(_sa),
  agent_fus_mlp (모두 lr 0.1배). traj_branches(에이전트 궤적 출력)는 동결 유지.
- 평가는 VO 를 끄고(exp_full_tgtbev config) 한다. 관문: 전체 val ≤ 0.288 (A-1 0.298 대비 −0.01) 이면 채택.
"""
_base_ = ['./exp_stage2_vo.py']

train_only = [
    'pts_bbox_head.vo_',
    'pts_bbox_head.ego_fut_decoder', 'pts_bbox_head.ego_agent_decoder', 'pts_bbox_head.ego_map_decoder',
    'pts_bbox_head.ego_query', 'pts_bbox_head.ego_agent_pos_mlp', 'pts_bbox_head.ego_map_pos_mlp',
    'pts_bbox_head.motion_decoder', 'pts_bbox_head.motion_map_decoder', 'pts_bbox_head.motion_mode_query',
    'pts_bbox_head.lane_encoder', 'pts_bbox_head.pos_mlp', 'pts_bbox_head.agent_fus_mlp',
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
            'pts_bbox_head.motion_decoder': dict(lr_mult=0.1),
            'pts_bbox_head.motion_map_decoder': dict(lr_mult=0.1),
            'pts_bbox_head.motion_mode_query': dict(lr_mult=0.1),
            'pts_bbox_head.lane_encoder': dict(lr_mult=0.1),
            'pts_bbox_head.pos_mlp': dict(lr_mult=0.1),
            'pts_bbox_head.agent_fus_mlp': dict(lr_mult=0.1),
        }))
