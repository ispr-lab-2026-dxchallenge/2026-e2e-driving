"""A-1 재현 (9/17, 메인컴, seed 2). 이름만 다른 config — 같은 이름이면 train_watch 가 옛 latest.pth 를 이어받는다.

목적: (1) A-1 레시피(동결 soup 트렁크 + VO 확률 주입으로 planner 꼬리 학습, 추론은 VO 끔)의 시드 재현성,
     (2) 같은 초기값(soup_e5e6)에서 갈라진 꼬리 둘(A-1 e2 + 이 실행 e2)의 가중치 평균(tail soup) 후보.
     교차 시드 soup 은 본학습에서는 실패했지만(L2 2.1+), 여기서는 트렁크가 동일하고 꼬리만 같은 점에서
     lr 0.1배로 미세조정되므로 같은 골짜기(basin) 안에 머문다 → 평균이 성립할 가능성이 높다.
실행: SEED=2 ABLATION=1 bash tools/train_watch.sh exp_stage2_vo_s2 "data.samples_per_gpu=5 data.workers_per_gpu=3"
평가: VO 를 끄고(exp_full_tgtbev config) 서브컴 exp_stage2_vo_s2/ 에서 (chain_s2vo.sh).
"""
_base_ = ['./exp_stage2_vo.py']
