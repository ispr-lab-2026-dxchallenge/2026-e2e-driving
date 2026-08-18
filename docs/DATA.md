# 데이터 — 전처리 명세와 배포

## 왜 전처리를 하는가

원본 학습 파이프라인은 **매 학습 스텝마다** 다음을 반복한다.

```python
LoadMultiViewImageFromFiles      # 1920×1536 JPEG 18장 디코딩 (카메라 6 × 시계열 3)
UndistortMultiViewImage          # 렌즈 왜곡 보정 (cv2.remap, 전체 해상도)
CropMultiViewImage               # 1920×1080 으로 잘라내기
RandomScaleImageMultiViewImage   # 768×432 로 축소
```

이 네 단계에는 **랜덤 요소가 없다.** 같은 이미지에 항상 같은 결과가 나온다.
그런데 60 epoch × 112,800 샘플 = **676만 번** 반복된다.

한 번만 계산해서 저장해두면 되는 일이다. 실측 효과:

| | data_time | 샘플당 | 1 epoch |
|---|---|---|---|
| 원본 | 1.278 s | 0.666 s | 20.9 h |
| **전처리 후** | **0.127 s** | **0.385 s** | **12.1 h** |

`data_time` 이 10분의 1로 줄고 전체는 **1.73배** 빨라진다.

---

## 전처리 파라미터 (이 값을 바꾸면 실험 비교가 불가능해진다)

```
scale     0.4
crop      1920 × 1080
결과 해상도 768 × 432
JPEG 품질  85
```

| 항목 | 원본 | 전처리 후 |
|---|---|---|
| 해상도 | 1920 × 1536 | 768 × 432 |
| 장당 용량 | 약 347 KB | 약 60 KB |
| 전체 (train+test) | 265 GB | **약 40 GB** |

---

## 무슨 연산을 하는가

`preprocess_etri.py` 는 원본 파이프라인 코드와 **동일한 함수**를 사용한다.

### 1) 왜곡 보정 (undistort)

카메라 렌즈는 직선을 휘게 찍는다. 3D 라벨을 이미지에 정확히 투영하려면 이걸 펴야 한다.

```python
# 보정 후의 새 내부 파라미터를 먼저 구한다
new_K = cv2.getOptimalNewCameraMatrix(K, distortion, size, alpha=0)[0]
# fisheye(camera_rear_wide)는 별도 함수
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(...)

# 픽셀 대응표를 만들어 remap
map1, map2 = cv2.initUndistortRectifyMap(K, distortion, None, new_K, size, cv2.CV_32FC1)
img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

대응표(`map1/map2`)는 카메라별로 한 번만 만들어 시나리오 안에서 재사용한다.
원본 파이프라인이 매 스텝 새로 만들던 부분이다.

### 2) 자르기 (crop) — 카메라마다 다르다

1920×1536 에서 1920×1080 만 남긴다. **어디를 남기는지가 카메라마다 다르다.**

| 카메라 | 남기는 부분 | 잘려나가는 것 |
|---|---|---|
| `camera_front` | **아래쪽** 1080줄 (oy=456) | 하늘 |
| 나머지 5대 | **위쪽** 1080줄 (oy=0) | 차체·사이드미러 |

```python
ox = (W - 1920) // 2                      # = 0
oy = 0 if cam in CROP_KEEP_TOP else H - 1080   # front 만 456
img = img[oy:oy+1080, ox:ox+1920]
```

### 3) 축소 (scale)

```python
img = cv2.resize(img, (768, 432), interpolation=cv2.INTER_LINEAR)
```

---

## ★ 가장 중요한 부분 — 카메라 파라미터도 같이 고쳐야 한다

이미지를 자르고 줄이면 **내부 파라미터(intrinsic)도 그만큼 변한다.**
이걸 안 고치면 3D 라벨이 이미지의 엉뚱한 곳에 투영되는데, **에러가 나지 않는다.**
학습은 멀쩡히 돌고 loss 도 떨어지지만 모델이 잘못된 대응을 배운다.

```
crop  : cx -= ox ,  cy -= oy
scale : fx, fy, cx, cy 에 0.4 곱하기
```

`convert_pre.py` 가 pkl 을 만들 때 이 보정을 적용한다. 실측 확인:

```
camera_front   fx 1360.9 → 544.4      (1360.9 × 0.4 = 544.4)
               cy  791.4 → 134.2      ((791.4 − 456) × 0.4 = 134.2)
               size 1920×1536 → 768×432
```

### 검증 방법

`verify_pre.py` 가 두 가지로 확인한다.

1. **수치** — 3D 점 4,000개를 두 경로로 투영해 비교
   - A: 원본 K 로 투영 후 2D 에서 crop/scale
   - B: 보정된 K 로 바로 투영
   - 결과: 최대 오차 **1.09e-11 px** (부동소수점 잡음 수준)

2. **시각** — 전처리 이미지에 3D 박스와 차선을 그려서 눈으로 확인
   - 차선이 노면 표시 위에, 박스가 객체 위에 얹히면 정상

**전처리 파라미터를 바꾸면 반드시 이 검증을 다시 돌릴 것.**

---

## 직접 만들어야 한다면

원본 tar 가 있는 경우에만 해당한다. **보통은 담당자에게 받는다.**

```bash
# Windows (Docker Desktop) - D: 를 컨테이너에 마운트하려면 Windows CLI 를 써야 한다
docker run --rm \
  -v "D:\...\dx_challenge:/src:ro" \
  -v "\\wsl.localhost\Ubuntu-24.04\home\<user>\ETRI_Challenge\ETRI_E2E_Driving_Challenge:/workspace/VAD" \
  -v "\\wsl.localhost\Ubuntu-24.04\home\<user>\ETRI_Challenge\data:/workspace/VAD/data" \
  etri-vad:cu128 python /workspace/VAD/preprocess_etri.py \
    --src /src/train --out /workspace/VAD/data/etri_pre/train \
    --quality 85 --workers 12
```

```bash
# 리눅스
docker run --rm \
  -v /path/to/dx_challenge:/src:ro \
  -v ~/ETRI_Challenge/ETRI_E2E_Driving_Challenge:/workspace/VAD \
  -v ~/ETRI_Challenge/data:/workspace/VAD/data \
  etri-vad:cu128 python /workspace/VAD/preprocess_etri.py \
    --src /src/train --out /workspace/VAD/data/etri_pre/train \
    --quality 85 --workers 12
```

12 워커 기준 약 1시간.

> 컨테이너가 root 로 파일을 만들기 때문에 끝나고 소유권을 넘겨야 한다.
> `docker exec etri chown -R 1000:1000 /workspace/VAD/data/etri_pre`

---

## info pkl 생성

전처리 데이터에는 **반드시 `convert_pre.py`** 를 쓴다. 원본 컨버터를 쓰면 내부 파라미터가
보정되지 않아 정합이 깨진다.

```bash
docker exec etri python /workspace/VAD/convert_pre.py --split train \
  --root-path /workspace/VAD/data/etri_pre/train \
  --out-dir /workspace/VAD/data/etri --extra-tag vad_etri_pre

docker exec etri python /workspace/VAD/convert_pre.py --split test \
  --root-path /workspace/VAD/data/etri_pre/test \
  --out-dir /workspace/VAD/data/etri --extra-tag vad_etri_pre
```

생성물:

```
data/etri/vad_etri_pre_infos_temporal_train.pkl
data/etri/vad_etri_pre_infos_temporal_test.pkl
```

> pkl 에는 이미지의 **절대 경로**가 박힌다. 컨테이너 마운트 경로가 팀원마다 같아야 한다
> (`/workspace/VAD/data/...`). 다르면 pkl 을 각자 다시 만들어야 한다.

---

## 데이터 배포

**전처리는 한 사람이 한 번만 한다.** 각자 하면 시간 낭비이고, 파라미터가 어긋나면
실험 결과를 비교할 수 없게 된다.

| 방법 | 소요 |
|---|---|
| 외장 SSD 복사 | 가장 빠름 |
| 연구실 NAS·공유 스토리지 | 환경에 따라 |
| 기가비트 LAN `scp` | 40GB 기준 약 6분 |
| 100Mbps LAN `scp` | 약 1시간 40분 — 스위치 추가 권장 |

배포할 때 이 문서의 **파라미터 표를 같이 전달**할 것. 나중에 "이 데이터 뭘로 만든 거였지"가
반드시 나온다.

---

## 원본 데이터 구조 참고

### 학습 (`<시나리오>.tar`, 시나리오당 1,818 파일)

```
20260112-105434/
├── camera_front/ ... camera_rear_right/   6대 × 300장 (00000000.jpg ~ 00000299.jpg)
├── calibration/calibration.parquet         K · distortion · is_fisheye · euler · translation
├── annotation/
│   ├── object.parquet      3D 박스 (Car/Pedestrian/Cyclist + ego), obj_id 추적 ID
│   ├── ego_pose.parquet    자차 6DOF
│   ├── map.parquet         3D 차선 폴리라인
│   ├── hd_map.parquet      HD Map 2D (베이스라인 미사용)
│   └── hd_ego_pose.parquet HD Map 자차 궤적 (베이스라인 미사용)
└── meta/
    ├── timestamps.parquet
    └── command.parquet     LANE_KEEP / TURN_LEFT / ... (베이스라인 미사용)
```

- 시나리오 = **40초 400프레임**. 과거 5초(−50~−1) + 메인 30초(0~299) + 미래 5초(300~349)
- **카메라는 메인 300프레임만** 있다. 앞뒤는 라벨(궤적)만

### 테스트 (`<clip_hash>.tar`, clip 당 196 파일)

```
00017c73feca4fde/
├── camera_*/            6대 × 31장 (frame_0 = 현재 … frame_-30 = 3초 전)
├── ego_pose.parquet     과거 궤적 + frame=+50 (5초 목표점)
├── calibration.parquet
└── command.parquet
```

### 주의 — 박스 치수 규약이 일반 관례와 반대다

| 컬럼 | 의미 |
|---|---|
| `width[m]` | **진행방향(전후)** 길이 — 승용차 평균 4.69 m |
| `length[m]` | **횡방향(좌우)** 길이 — 승용차 평균 2.03 m |
| `height[m]` | 수직 높이 |
| `z[m]` | **박스 바닥면** — 중심 높이 = `z + height/2` |
