"""본 학습 시드 3 (9/11): exp_full_tgtbev 와 완전히 같은 레시피(10 epoch 코사인), seed 만 3.

근거 (RESULTS.md 10장): 세 시드 모두 lr 1e-4 근처인 epoch 5~6 에서 정점, 7 epoch 어닐링은
두 머신에서 기각. 남은 합법 지렛대는 같은 레시피의 새 시드 + 정점 인접 epoch 가중치 평균.
메인컴 11h/epoch 이라 epoch 7 까지(9/15 새벽) 쓰고 9/16 12:00 하드스톱 전에 평가·제출한다.
work_dir 를 분리하기 위한 이름뿐인 config. 실행: SEED=3 ABLATION=1 train_watch.sh exp_full_tgtbev_s3
"""
_base_ = ['./exp_full_tgtbev.py']
