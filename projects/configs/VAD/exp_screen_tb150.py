"""스크리닝 결합: BEV150 + 목표점->BEV 조건 입력 (두 채택 축의 상호작용 검증).

단독 성적 (10ep, 1/3 데이터): bev150 0.6713, tgtbev 0.5306 (기준 0.8369).
둘은 직교 축(해상도 x 조건 입력)이라 파괴적 상호작용 가능성은 낮지만,
장기런(A100)에 싣기 전에 확인한다.

tgtbev 채택 근거 (8/30): 영상 ablation zero 1.22배 (순수 비전 기준 1.55배)
- 의존도는 낮아졌으나 붕괴 아님. 결합 모델도 ABLATION=1 필수.

메인컴 5060Ti 기준 b4 권장 (bev150 이 무겁다).
"""
_base_ = ['./exp_screen_bev150.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            use_target_cond=True,
            target_cond_drop=0.5,
        ),
    ))
