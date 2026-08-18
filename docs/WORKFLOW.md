# 실험 규칙과 제출 절차

## 왜 규칙이 필요한가

4명이 각자 돌리면 한 주만 지나도 **어떤 설정이 무슨 결과를 냈는지 아무도 모르게 된다.**
그리고 제출이 **팀당 5회**뿐이라, 무엇을 제출할지 근거를 갖고 정해야 한다.

---

## 1. config 는 새 파일을 만들어 상속한다

공통 config 를 직접 고치면 4명이 매번 충돌한다. **바꾸는 부분만 적은 새 파일**을 만든다.

```
projects/configs/VAD/
├── VAD_etri_tiny.py           원본 (수정 금지)
├── VAD_etri_base.py           원본 (수정 금지)
├── VAD_etri_tiny_pre.py       전처리 데이터용 기준 (수정 시 팀에 공유)
├── exp_jeung_0815_bev150.py   ← 각자 여기에
├── exp_kim_0815_queue5.py
└── exp_lee_0816_base.py
```

### 명명 규칙

```
exp_<이름>_<MMDD>_<변경점>.py
```

### 작성 예

```python
# exp_jeung_0815_bev150.py
_base_ = ['./VAD_etri_tiny_pre.py']

bev_h_ = 150
bev_w_ = 150
fp16 = dict(loss_scale=512.)
data = dict(samples_per_gpu=6)
```

세 줄이면 충분하다. 나머지는 상속된다.

> 공통 코드(`projects/mmdet3d_plugin/`)를 고쳐야 할 때는 **먼저 팀에 알린다.**
> 모두의 실험에 영향을 준다.

---

## 2. 배치 크기는 GPU 에 맞춘다

| VRAM | 권장 | 근거 |
|---|---|---|
| 16 GB | **fp16 batch 6** | 11.9 GB 점유 |
| 12 GB | **fp16 batch 4** | 8.1 GB 점유 |
| 24 GB 이상 | batch 8 시도 | 논문 기준 실효 배치와 일치 |

**VRAM 을 90% 이상 쓰면 안 된다.** Windows/WSL 은 넘치면 에러 대신 시스템 RAM 으로
흘려보내며 **조용히 20배 느려진다.** (batch 8 을 16GB 에서 돌렸을 때 실측: 2.3초 → 43초)

학습이 이유 없이 느리면 `nvidia-smi` 로 VRAM 점유율부터 확인할 것.

> config 의 `lr=2e-4` 는 원래 **8 GPU × batch 1 = 실효 배치 8** 기준으로 맞춰진 값이다.
> 단일 GPU batch 1 은 의도된 설정에서 8배 벗어난 상태다. 배치를 키우는 건 속도뿐 아니라
> **학습 품질** 문제이기도 하다.

---

## 3. 실험 기록

### 최소 — 공유 스프레드시트

| 실험명 | 담당 | 날짜 | 변경점 | 데이터 | epoch | val L2 | FLOPs | VRAM | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| exp_jeung_0815_bev150 | 정인호 | 8/15 | BEV 150 | pre_0.4 | 12 | 1.24 | 890G | 14GB | |

### 권장 — Weights & Biases

`mmcv 1.4.0` 에 `WandbLoggerHook` 이 있다. config 에 몇 줄만 추가하면 4명이 각자 다른
머신에서 돌려도 **한 화면에서 비교**된다.

```python
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='etri-e2e',
                              entity='<팀 계정>',
                              name='exp_jeung_0815_bev150')),
    ])
```

스프레드시트만 쓰면 누군가 갱신을 빼먹는 순간 신뢰가 깨진다. 자동 기록을 권한다.

---

## 4. 자체 검증셋 — 사실상 우리 리더보드

제출이 5회뿐이므로 **자체 검증이 실질적인 리더보드 역할**을 한다.

- 학습 시나리오 376개 중 일부를 검증용으로 뗀다
- **분할 기준을 팀이 합의하고 고정한다.** 사람마다 다르면 비교가 무의미하다
- 시나리오 단위로 나눈다 (프레임 단위로 섞으면 같은 장면이 양쪽에 들어가 성능이 부풀려진다)

분할이 정해지면 이 문서에 기록할 것.

```
train : (예) 시나리오 ___ 개
val   : (예) 시나리오 ___ 개   목록: docs/val_split.txt
```

---

## 5. 장시간 학습

1 epoch 이 약 12시간이다. 터미널에 묶어두면 안 된다.

```bash
nohup docker exec etri python tools/train.py \
  projects/configs/VAD/exp_jeung_0815_bev150.py \
  --work-dir work_dirs/exp_jeung_0815_bev150 --no-validate \
  >> ~/logs/exp_jeung_0815_bev150.log 2>&1 &
```

진행 확인:

```bash
tail -f ~/logs/<실험명>.log
nvidia-smi
```

중단된 학습 이어서:

```bash
python tools/train.py <config> --work-dir <같은 경로> --resume-from <경로>/latest.pth
```

체크포인트가 258MB 이므로 `checkpoint_config.max_keep_ckpts` 를 줄여두는 게 좋다.
기본값 60 이면 15GB 가 쌓인다.

---

## 6. 제출 절차

### 제출 전 체크리스트

- [ ] 자체 검증셋에서 기존 최고 기록보다 나은가
- [ ] FLOPs 가 cut-off(7,053 G) 이하인가 — 현재 tiny 는 446.6 G 로 여유 충분
- [ ] clip 수가 **1,125개** 전부 들어 있는가
- [ ] 각 clip 이 (6, 2) 형태인가
- [ ] `__flops__` 키가 있는가
- [ ] **팀에 알렸는가**

> 누락되거나 형식이 틀린 clip 은 `[0,0]`(제자리)로 채점된다. 제출 실수는 참가자 책임이다.

### 생성 명령

```bash
# 1) FLOPs 기록
python tools/measure_flops.py <config> \
  --ann-file data/etri/vad_etri_pre_infos_temporal_test.pkl \
  --out submission.json

# 2) 추론 (같은 파일에 병합된다. 순서 무관)
python tools/etri_test_submit.py <config> <checkpoint.pth> \
  --ann-file data/etri/vad_etri_pre_infos_temporal_test.pkl \
  --out submission.json
```

### 형식

```json
{
  "__flops__": 446605899456,
  "<clip_hash>": [[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5],[x6,y6]],
  ...
}
```

- 6점 = 미래 0.5 / 1.0 / 1.5 / 2.0 / 2.5 / 3.0 초의 **누적** waypoint
- 좌표계 = 현재 프레임 원점, x=전방 y=좌
- 다중모드 모델이라도 **궤적 1개**로 줄여서 제출

### 5회 배분 (제안)

| 회차 | 목적 |
|---|---|
| 1 | 베이스라인 그대로 — 리더보드 기준점 확보 |
| 2~3 | 유력 후보 검증 |
| 4~5 | 최종. 마감 직전 여유 확보 |

---

## 7. 재현성 — 처음부터 챙긴다

수상 후보가 되면 **재현성 평가**가 있고, 2차 제출물로 다음을 내야 한다.

- Docker 이미지
- 제출 성적에 해당하는 체크포인트 (`.pth`)
- 모델 및 추론 코드
- FLOPs 측정 코드
- 외부 데이터셋 사용 시 보고서 및 백본 변경 내역서

마감 직전에 몰아서 하면 반드시 사고가 난다. 학습할 때마다:

- `work_dirs/<실험명>/` 에 config 스냅샷이 자동 저장된다 (mmcv 기능)
- 거기에 **git commit 해시**를 함께 기록해두면 완벽하다

```bash
git rev-parse HEAD > work_dirs/<실험명>/git_commit.txt
```

---

## 8. 역할 분담 (제안)

| 역할 | 하는 일 |
|---|---|
| **데이터** | 전처리·배포, train/val 분할 확정, HD Map·command 활용 검토 |
| **모델** | 구조 실험 (BEV 해상도, 디코더 층수, 백본, VADv2 head) |
| **학습 운영** | 장기 학습 관리, 머신 스케줄링, 체크포인트 보관 |
| **평가·제출** | 자체 검증 파이프라인, FLOPs·추론시간 관리, 2차 제출물 준비 |

특히 **평가·제출 담당을 따로 두기를** 권한다.
