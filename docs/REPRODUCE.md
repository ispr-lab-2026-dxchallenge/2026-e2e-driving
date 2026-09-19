# 재현 패키지 — 제출 결과를 그대로 다시 만드는 절차

2차 평가용. 환경 구축은 `docs/SETUP.md`, 모델의 구조·규정 대조는 `docs/JUDGE_NOTES.md` 를 본다.
여기서는 **무엇을 실행하면 제출한 숫자가 그대로 나오는지**만 적는다.

## 0. 패키지 구성

| 파일 | 내용 | 크기 |
|---|---|---|
| `etri-vad-cu128.tar` | Docker 이미지 (CUDA 12.8, torch 2.7.1, mmdet3d/mmcv 1.4.0 + 플러그인 의존성) | 약 10GB |
| 이 저장소 | 코드·config·문서·테스트 | 약 30MB |
| `final_model.pth` | 제출 모델 가중치 (미사용 키 제거 완료) | 60MB |
| `submission.json` | 제출한 예측 (1,125 clip × (6,2) + `__flops__`) | 12MB |
| `data/etri/` | 전처리된 주석 pkl + 이미지 (운영국 제공 데이터에서 `preprocess_etri.py` 로 생성) | — |

`final_model.pth` 는 `tools/strip_ckpt.py` 로 추론에 쓰이지 않는 `vo_*` 가중치를 제거한 것이다.
제거 전후 검증 점수가 소수점 넷째 자리까지 같음을 확인했다(JUDGE_NOTES 5장).

## 1. 환경 기동

```bash
docker load -i etri-vad-cu128.tar
docker run -d --name etri --gpus all --shm-size 16g \
  -v $PWD/ETRI_E2E_Driving_Challenge:/workspace/VAD \
  -v $PWD/data:/workspace/VAD/data \
  -v $PWD/work_dirs:/workspace/VAD/work_dirs \
  etri-vad:cu128 sleep infinity
docker exec etri python -c "import torch, mmcv, mmdet3d; print(torch.__version__, mmcv.__version__, mmdet3d.__version__, torch.cuda.is_available())"
```

## 2. 구조 검사 (GPU 불필요, 30초)

```bash
docker exec etri python tests/test_target_point_flow.py
docker exec etri python tests/test_plan_cum_loss.py
docker exec etri python tests/test_vo_provenance.py projects/configs/VAD/exp_full_tgtbev_vo.py final_model.pth   # GPU 필요
```

첫 번째가 이 제출의 핵심 주장(목표점이 planner 입력이 아니다)을 기계적으로 확인한다. 검사 14종.

## 3. 검증 점수 재현 (3060 기준 9분)

```bash
docker exec etri python tools/eval_protocol.py \
  projects/configs/VAD/exp_full_tgtbev.py final_model.pth \
  --scenarios 0 --eval-stride 10 --offsets=-10,-5,0 --history-bev-only --fp16
```

기대 출력:

```
offsets                         1s      2s      3s    L2_avg |   T_infer      프레임당
-10,-5,0                     0.144   0.278   0.461    0.2941 |   241.7ms    80.6ms
```

평가는 결정적이라 같은 하드웨어에서 소수점 넷째 자리까지 재현된다. 검증 집합은 학습에 쓰지 않은
39 시나리오 1,053 평가점이며 분할은 `docs/val_split.txt` 에 고정돼 있다.

## 4. 영상 의존성 재측정 (30분)

```bash
docker exec etri python tools/ablate_window.py \
  projects/configs/VAD/exp_full_tgtbev.py final_model.pth --scenarios 0
```

기대: normal 0.2941 / 영상 zero 0.4199(1.43배) / shuffle 0.3543 / 목표점 off 0.3740 / 목표점 교체 1.1044.

## 5. 제출 JSON 재생성 (약 45분)

```bash
bash tools/make_submission.sh <work_dir 이름> final_model.pth exp_full_tgtbev
```

ablation → FLOPs 측정 → 테스트 1,125 clip 추론 → 형식 검사를 순서대로 돌리고, 하나라도 실패하면
중단한다. 산출물을 제출한 `submission.json` 과 비교하면 된다(동일 하드웨어에서 완전 일치).

## 6. 학습 재현 (선택, GPU 시간 약 120시간)

제출 모델은 세 단계로 만들어졌다.

```bash
# 1단계: 본 학습 10 epoch (3090 기준 11h/epoch)
bash tools/train_watch.sh exp_full_tgtbev "data.samples_per_gpu=5 data.workers_per_gpu=3"
# 2단계: epoch 5·6 가중치 평균으로 트렁크 확정
python tools/avg_ckpt.py --out work_dirs/exp_full_tgtbev/soup_e5e6.pth \
    work_dirs/exp_full_tgtbev/epoch_5.pth work_dirs/exp_full_tgtbev/epoch_6.pth
# 3단계: 트렁크 동결 + planner 꼬리 2 epoch (6.7h/epoch)
bash tools/train_watch.sh exp_stage2_vo "data.samples_per_gpu=5 data.workers_per_gpu=3"
# 최종: 서로 다른 실행의 꼬리 3개를 평균
python tools/avg_ckpt.py --out final_model.pth <꼬리1> <꼬리2> <꼬리3>
```

**학습은 결정적이지 않다.** 같은 config·같은 시드를 두 대의 GPU 에서 돌린 결과 검증 L2 가
0.2989 와 0.3078 로 0.009 차이났다(cuDNN 비결정성). 재현 시 0.29~0.31 범위면 정상이며,
소수점 셋째 자리까지 같기를 기대해서는 안 된다. 제출 가중치 자체는 위 3·4·5번으로 정확히
검증되므로, 판정은 학습 재실행이 아니라 제출 가중치로 하는 것이 맞다.

## 7. 데이터 전처리

운영국 제공 원본에서 주석 pkl 을 만드는 절차는 `preprocess_etri.py` 이고, train/val 분할은
`make_split.py` 가 고정 시드로 생성해 `docs/val_split.txt` 에 기록해 두었다. 목표점(5초 후 위치)은
학습·검증 주석에 이미 포함된 값을 그대로 쓰며 별도 계산이 없다.
