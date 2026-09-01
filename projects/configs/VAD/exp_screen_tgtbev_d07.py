"""스크리닝: 목표점 힌트의 condition dropout 0.5 -> 0.7.

챔피언(tgtbev 0.5306)의 영상 가림 검사가 1.22배로 합격선(1.2) 턱걸이였다.
목표점을 더 자주 지우며 학습하면 영상 의존이 올라가 심사 방어 여유가
커진다. 성능이 0.5306 근처를 유지하면서 ablation 배율이 오르면 채택.
비교 대상: tgtbev(dropout 0.5) 0.5306. ABLATION=1 필수.
서브컴(3060 12GB)은 batch 4 로 - 배치 차이가 있으므로 해석 시 명시.
"""
_base_ = ['./exp_screen_tgtbev.py']

model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            target_cond_drop=0.7,
        ),
    ))
