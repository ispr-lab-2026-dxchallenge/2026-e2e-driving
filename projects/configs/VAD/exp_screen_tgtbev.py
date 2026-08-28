"""스크리닝 트랙 T: 목표점 -> BEV 조건 입력 + condition dropout 0.5.

2026-08-28 운영국 확정: "planner 입력이 아닌 BEV/scene feature 형성 단계의
조건 입력으로 사용하는 간접적 활용은 허용" - 이 config 가 정확히 그 구조다.
can_bus 가 BEV 쿼리에 더해지는 베이스라인 기존 경로와 나란히, 목표점(정규화
좌표 2개)을 MLP 로 임베딩해 더한다. planner 는 그대로 scene 특징만 읽는다.

condition dropout 0.5: 학습 중 절반의 스텝에서 목표점을 지운다.
자차속도·목표점 직접 입력 실험(RESULTS.md #4~7)에서 실증된 지름길 학습을
차단하기 위함이다. 채택 요건: ablation 통과 (ABLATION=1 로 실행할 것).

비교 대상: exp_screen_base (기준값 0.8369)
"""
_base_ = ['./exp_screen_base.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            use_target_cond=True,
            target_cond_drop=0.5,
        ),
    ))
