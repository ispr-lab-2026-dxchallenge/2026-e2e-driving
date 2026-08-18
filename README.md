# E2E Driving Challenge — 팀 저장소

카메라 6채널 입력으로 자차의 미래 3초 궤적을 예측하는 end-to-end 모델.
베이스라인은 **VAD** (ICCV 2023) 이며 mmdetection3d 플러그인 형태로 제공된다.

---

## 시작하기 전에 — 이것만은 지켜주세요

| 규칙 | 이유 |
|---|---|
| **데이터·체크포인트를 커밋하지 않는다** | `.pth` 하나가 258MB. 한 번 들어가면 되돌리기 어렵다 |
| **공통 config 를 직접 수정하지 않는다** | 4명이 같은 파일을 고치면 매번 충돌한다. 새 파일을 만들어 상속할 것 |
| **전처리를 각자 하지 않는다** | 파라미터가 다르면 실험 결과를 서로 비교할 수 없다 |
| **제출 전에 팀에 알린다** | 제출은 **팀당 5회**다. 개인당이 아니다 |

---

## 1. 환경 구축

처음 세팅하는 경우 → **[docs/SETUP.md](docs/SETUP.md)**

이미 세팅된 경우:

```bash
git clone <저장소 주소> ~/ETRI_Challenge/ETRI_E2E_Driving_Challenge
docker start etri          # 컨테이너가 이미 있다면
```

## 2. 데이터 준비

전처리된 데이터를 담당자에게 받는다. 직접 만들지 않는다.
→ **[docs/DATA.md](docs/DATA.md)**

배치 위치:

```
~/ETRI_Challenge/
├── ETRI_E2E_Driving_Challenge/   ← 이 저장소
├── data/etri_pre/                ← 전처리 데이터 (배포받은 것)
├── data/etri/*.pkl               ← 컨버터 산출물
└── work_dirs/                    ← 학습 결과 (각자 로컬)
```

## 3. 학습 실행

```bash
# 컨테이너 안에서
python tools/train.py projects/configs/VAD/<내_config>.py \
  --work-dir work_dirs/<실험명> --no-validate
```

장시간 학습은 반드시 백그라운드로:

```bash
nohup docker exec etri python tools/train.py ... >> ~/train.log 2>&1 &
```

## 4. 실험 규칙 / 제출 절차

→ **[docs/WORKFLOW.md](docs/WORKFLOW.md)**

---

## 저장소 구조

```
projects/
├── configs/VAD/
│   ├── VAD_etri_tiny.py           원본 (수정 금지)
│   ├── VAD_etri_base.py           원본 (수정 금지)
│   ├── VAD_etri_tiny_pre.py       전처리 데이터용 기준 config
│   └── exp_*.py                   각자의 실험 config
└── mmdet3d_plugin/                모델·데이터셋 구현
    ├── VAD/                       VADHead, transformer, planner
    └── datasets/                  VADCustomETRIDataset

tools/
├── train.py  test.py              학습·평가
├── etri_test_submit.py            추론 → submission.json
├── measure_flops.py               FLOPs 측정
└── data_converter/                원본 컨버터

preprocess_etri.py                 이미지 사전 전처리 (undistort/crop/scale)
convert_pre.py                     전처리 데이터용 컨버터 래퍼
verify_pre.py                      전처리 정합 검증
explore_etri.ipynb                 데이터 탐색 + 3D 라벨 투영 검증
```

---

## 알아두면 좋은 수치

측정 환경: RTX 5060 Ti 16GB, 전처리 데이터(768×432), fp16

| 항목 | 값 |
|---|---|
| 모델 파라미터 | 39.86 M |
| **FLOPs** | **446.6 G** (cut-off 7,053 G 의 **6.3%**) |
| VRAM (batch 6) | 11.9 GB |
| 학습 속도 | 0.385 s/sample |
| **1 epoch (112,800 샘플)** | **약 12 시간** |
| 체크포인트 크기 | 258 MB |

**연산량 제약은 걸림돌이 아니다.** cut-off 까지 15.8배 여유가 있다.
진짜 제약은 **학습 시간과 VRAM** 이다.

| GPU | 권장 배치 | 근거 |
|---|---|---|
| 16GB (5060 Ti 등) | **fp16 batch 6** | 11.9GB, batch 8 은 VRAM 초과로 20배 느려짐 |
| 12GB (3060 등) | **fp16 batch 4** | 8.1GB |

> VRAM 이 넘치면 Windows/WSL 은 에러 없이 시스템 RAM 으로 흘려보내며 **조용히 20배 느려진다.**
> 학습이 갑자기 느려지면 `nvidia-smi` 로 VRAM 점유율부터 확인할 것.

---

## 대회 규정 요약

| 항목 | 내용 |
|---|---|
| 입력 | 카메라 6채널 (과거 3초) + 5초 목표점 + 주행 명령 |
| 출력 | 미래 3초 궤적 — 2Hz, 6점, 현재 프레임 기준 상대좌표 |
| 1차 평가 | Planning L2 (1·2·3초 평균). FLOPs 가 베이스라인 3배 초과 시 **탈락** |
| 2차 평가 | `Error Score = L2 × (1 + max(0, T_infer − 100) / 200)` |
| 제출 | **팀당 5회** |
| 금지 | 테스트 데이터로 학습·라벨 생성·GT 역추적 |
| 허용 | 라이선스 문제 없는 외부 공개 데이터셋 (사용 내역 제출 필요) |
