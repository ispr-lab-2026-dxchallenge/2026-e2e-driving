"""본 학습 시드 2 (4080 용, 9/13): exp_full_tgtbev 와 완전히 같은 레시피(10 epoch 코사인), seed 만 2.

work_dir 분리용 이름뿐인 config. 같은 이름(exp_full_tgtbev)으로 걸면 기존 완주 실행의
latest.pth 를 이어받아 새 시드가 되지 않는다(9/11 4080 에서 실제로 발생).
실행: SEED=2 ABLATION=1 ETRI_CONTAINER=etri2 bash tools/train_watch.sh exp_full_tgtbev_s2 "data.samples_per_gpu=5 data.workers_per_gpu=8"
정점은 epoch 5~6 (lr 1e-4 근처) 이므로 하드스톱(9/16 12:00) 전에 epoch 6 까지만 있어도 된다.
"""
_base_ = ['./exp_full_tgtbev.py']
