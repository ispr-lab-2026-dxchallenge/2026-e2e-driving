"""S1 (9/21): 5초 목표점 앵커 꼬리. A-1 레시피(동결 soup 트렁크 + VO 확률 주입) + condition dropout 0
+ planner 꼬리가 10스텝(5초)을 내고 7~10스텝을 5초 목표점(학습 라벨)으로 묶는다.

근거 (9/18 오차 해부): 오차의 93%가 종방향 드리프트 = "얼마나 갈지"의 눈금이 틀린다. 영상 없는
공식(등속->목표점 전환, 0.243)이 우리(0.294)를 이기는 이유는 목표점 거리/5초 = 앞으로의 평균 속력을
궤적 눈금에 직접 쓰기 때문이다. 우리 모델은 목표점을 BEV 조건으로 받지만(허용 형태) 6스텝 라벨만으로는
목표점과 3초 궤적의 눈금이 느슨하게만 묶인다. 꼬리가 5초까지 내고 10번째 누적 위치 = 목표점 이라는
정확한 라벨을 주면, BEV 에서 읽은 목표점과 궤적 진행거리가 한 사슬로 묶인다.

규정: 목표점의 모델 입력 경로는 불변(BEV 조건화 한 곳). 5초 위치는 학습 손실의 라벨로만 쓴다
(모든 궤적 학습이 미래 GT 위치를 라벨로 쓰는 것과 같은 성격). 추론은 앞 6스텝만 제출하고 뒤 4스텝은
버린다. 구조 변화 = 꼬리 마지막 Linear 출력 36 -> 60. FLOPs 변화 무시 가능.
초기값: tools/expand_fut_ckpt.py 로 soup 의 마지막 Linear 를 6->10 스텝으로 넓힌 것(7~10 스텝은
6번째 스텝 가중치 복제 = 마지막 속도 유지).
평가는 exp_full_tgtbev_goal10 (VO 끔, 같은 꼬리 모양). 관문: val ≤ 0.288.
"""
_base_ = ['./exp_stage2_vo.py']

load_from = 'work_dirs/exp_full_tgtbev/soup_e5e6_fut10.pth'
model = dict(pts_bbox_head=dict(
    goal_anchor_ts=10,
    goal_anchor_weight=0.2,
    transformer=dict(target_cond_drop=0.0),
))
