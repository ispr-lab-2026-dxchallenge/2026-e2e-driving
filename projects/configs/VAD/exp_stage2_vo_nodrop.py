"""S0 (9/21, 메인): A-1 레시피에서 목표점 condition dropout 을 끈다 (0.5 -> 0.0).

문제: 트렁크 본학습은 지름길 학습을 막으려고 목표점 임베딩을 50% 확률로 0 으로 지웠다
(VAD_transformer.py target_cond_drop, self.training 일 때만). stage-2 도 train 모드라 이 dropout 이
그대로 살아 있어, 꼬리는 절반은 "목표점 없는 BEV" 로 배웠다. 추론은 100% 목표점이 있다 → 학습·추론
불일치. 트렁크가 동결된 stage-2 에서는 트렁크가 지름길을 배울 수 없으므로 dropout 의 존재 이유가 없다.
구조·입력·FLOPs·추론 경로 불변. 평가는 VO 끔(exp_full_tgtbev). 관문: val ≤ 0.288.
"""
_base_ = ['./exp_stage2_vo.py']

model = dict(pts_bbox_head=dict(transformer=dict(target_cond_drop=0.0)))
