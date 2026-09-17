"""A-1 재현 seed 3 (9/17, A100). 이름만 다른 config — 시드는 train.py --seed 3 으로 준다.

목적: A-1 레시피(동결 soup 트렁크 + VO 확률 주입으로 planner 꼬리 학습, 추론은 VO 끔)를 독립 시드로
한 번 더 돌려, 같은 초기값에서 갈라진 꼬리들(A-1 e2, seed2 e2, seed3 e2)의 가중치 평균(tail soup)을
만든다. 9/17 확인: 같은 트렁크·같은 출발점의 꼬리는 평균이 성립(A-1 e2 + 이어학습 e1 = 0.2969).
실행(A100):
  python tools/train.py projects/configs/VAD/exp_stage2_vo_s3.py --seed 3 \
      --work-dir work_dirs/exp_stage2_vo_s3 --cfg-options load_from=/content/soup_main_e5e6.pth
평가는 VO 를 끄고(exp_full_tgtbev config) 서브컴에서.
"""
_base_ = ['./exp_stage2_vo.py']
