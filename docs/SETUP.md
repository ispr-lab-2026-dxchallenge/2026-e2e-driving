# 환경 구축

베이스라인은 Docker 이미지로만 배포된다. `mmcv-full 1.4.0`(2021년) 을 `PyTorch 2.7.1`(2025년)
위에 올린 조합이고, 주최측이 **직접 패치한 소스**로 빌드한 뒤 그 소스를 삭제했다.
즉 **이 이미지 없이는 같은 환경을 만들 수 없다.** 직접 설치를 시도하지 말 것.

---

## 준비물

| | 비고 |
|---|---|
| NVIDIA GPU | VRAM 12GB 이상 권장 |
| `etri-vad_cu128.tar` | 10.1 GB. 담당자에게 받는다 |
| 전처리 데이터 | 약 40 GB. 담당자에게 받는다 ([DATA.md](DATA.md)) |
| 디스크 | 최소 80 GB 여유 |

---

## A. 리눅스

### 1. Docker + NVIDIA Container Toolkit

```bash
# Docker 가 없다면
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER        # 로그아웃 후 재로그인

# GPU 를 컨테이너에서 쓰기 위해
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

확인:

```bash
docker info --format '{{range $k,$v := .Runtimes}}{{$k}} {{end}}'
# nvidia 가 목록에 있어야 한다
```

### 2. 이미지 로드

```bash
docker load -i etri-vad_cu128.tar
```

### 3. 드라이버 버전이 낮은 경우

이미지는 CUDA 12.8 을 요구한다. 드라이버가 그보다 낮으면 이런 에러가 난다.

```
Error 804: forward compatibility was attempted on non supported HW
```

**이름과 달리 GPU 아키텍처 문제가 아니다.** 이미지에 들어 있는 forward-compatibility
라이브러리가 GeForce 에서 지원되지 않아 생기는 문제다.

가장 깔끔한 해결은 **드라이버를 570 이상으로 올리는 것**이다. 올릴 수 없다면:

```bash
printf 'FROM etri-vad:cu128\nRUN rm -rf /usr/local/cuda-12.8/compat\n' > Dockerfile
docker build -t etri-vad:cu128-nocompat .
```

이 경우 실행할 때 `-e NVIDIA_DISABLE_REQUIRE=1` 을 추가한다.

---

## B. Windows

Docker 는 리눅스 커널이 필요한데 Windows 에는 없다. **WSL2** 가 그 역할을 한다.

### 1. 가상화 확인

```powershell
(Get-CimInstance Win32_Processor).VirtualizationFirmwareEnabled
```

`False` 면 BIOS 에서 켜야 한다.

### 2. WSL2 + 우분투

```powershell
wsl --install                      # 관리자 권한, 재부팅 필요
```

재부팅 후 배포판이 없을 수 있다. 최근 Windows 에서 흔하다.

```powershell
$env:WSL_UTF8=1                    # 출력 깨짐 방지
wsl -l -v                          # "설치된 배포가 없습니다" 라면
wsl --install -d Ubuntu-24.04
```

시작 메뉴에서 **Ubuntu** 를 한 번 실행해 계정을 만든다.

### 3. 메모리 상한 조정

WSL2 는 기본적으로 **호스트 RAM 의 절반만** 쓴다. 학습에는 부족할 수 있다.

`C:\Users\<사용자>\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

```powershell
wsl --shutdown                     # 적용
```

### 4. Docker Desktop

[공식 사이트](https://www.docker.com/products/docker-desktop/)에서 **AMD64** 버전.
CPU 가 Intel 이든 AMD 든 AMD64 다 (제조사가 아니라 명령어 규격 이름).

설치 후 **반드시**:

> **Settings → Resources → WSL Integration → `Ubuntu-24.04` 토글 ON → Apply & Restart**

기본값이 꺼져 있다. 이걸 놓치면 우분투 안에서 `docker` 명령이 아예 없다.
**"Docker Desktop 깔았는데 왜 안 되지"의 대부분이 이것이다.**

### 5. WSL 안에 NVIDIA 드라이버를 설치하지 말 것

Windows 드라이버가 WSL 로 그대로 노출된다. 우분투에서 `apt install nvidia-driver-*` 를
하면 **오히려 깨진다.**

아무것도 설치하지 않은 상태에서 이게 나와야 정상이다.

```bash
nvidia-smi
```

안 나오면 **Windows 쪽** 드라이버를 업데이트한다.

### 6. 이미지 로드

```bash
docker load -i /mnt/d/etri-vad_cu128.tar     # 10~15분
```

---

## C. 디렉토리 구성 (공통)

```bash
mkdir -p ~/ETRI_Challenge/{data,work_dirs}
cd ~/ETRI_Challenge
git clone <저장소 주소> ETRI_E2E_Driving_Challenge
```

전처리 데이터를 `~/ETRI_Challenge/data/etri_pre/` 에 배치한다.

**데이터는 반드시 리눅스 파일시스템 안(`~/`)에 둔다.**
Windows 드라이브(`/mnt/c`, `/mnt/d`)에 두면 작은 파일 대량 접근이 극도로 느려
학습이 데이터 로딩에 묶인다.

---

## D. 컨테이너 기동

```bash
docker run -d --name etri --restart unless-stopped \
  --gpus all --ipc=host --shm-size=8g \
  -v ~/ETRI_Challenge/ETRI_E2E_Driving_Challenge:/workspace/VAD \
  -v ~/ETRI_Challenge/data:/workspace/VAD/data \
  -v ~/ETRI_Challenge/work_dirs:/workspace/VAD/work_dirs \
  -v ~/.nv_cache:/root/.nv \
  -w /workspace/VAD \
  etri-vad:cu128 sleep infinity
```

드라이버가 낮아 `nocompat` 이미지를 만든 경우 태그를 바꾸고 `-e NVIDIA_DISABLE_REQUIRE=1` 추가.

`--ipc=host` 는 필수다. 없으면 DataLoader 워커가 공유메모리 부족으로 죽는다.
`.nv_cache` 마운트는 PTX JIT 결과를 보존해 두 번째 실행부터 빨라지게 한다.

---

## E. 검증

```bash
docker exec etri python -c "
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), 'sm_%d%d' % torch.cuda.get_device_capability(0))
for m in ['mmcv','mmdet','mmseg','mmdet3d']:
    print(m, __import__(m).__version__)
"
```

`torch.cuda.is_available()` 이 True 여도 안심할 수 없다. **mmcv 의 CUDA 확장은 따로
컴파일된 것**이라 별도로 확인해야 한다.

```bash
docker exec etri python /workspace/VAD/build_test.py
```

`BUILD TEST PASSED` 가 나오면 완료다.

> 최신 GPU(Blackwell 등)는 첫 커널 호출에서 PTX JIT 컴파일이 일어나 잠깐 멈춘 것처럼
> 보일 수 있다. 정상이며 이후 캐시된다.

---

## F. VS Code 연동 (선택)

학습만 돌린다면 필요 없다. **코드를 읽고 디버깅할 때** 유용하다.
mmdet3d 는 `type='VADHead'` 같은 문자열로 클래스를 찾는 구조라, 정의 이동 없이는
따라가기 어렵다.

1. **Dev Containers** 확장 설치
2. `F1` → `Dev Containers: Attach to Running Container` → `/etri`
3. `Ctrl+K Ctrl+O` → `/workspace/VAD`
4. 컨테이너 안에 **Python / Jupyter** 확장 설치 (`Ctrl+Shift+X` → `Install in Container`)
5. 인터프리터: `/opt/venv/bin/python`

> "No Python found. Install using uv?" 알림이 뜨면 **Cancel / Don't ask again**.
> 이미지에는 표준 경로에 파이썬이 없어 자동 탐색이 실패할 뿐, `/opt/venv/bin/python` 이
> 정상 동작한다. 저장소의 `.vscode/settings.json` 이 이 경로를 지정해둔다.

원격 리눅스 머신이라면 **Remote-SSH → Dev Containers** 2단 중첩도 지원된다.

---

## 자주 막히는 곳

| 증상 | 원인 |
|---|---|
| `docker: command not found` (WSL) | Docker Desktop → WSL Integration 토글 |
| `could not select device driver [[gpu]]` | nvidia-container-toolkit (리눅스) / Windows 드라이버 업데이트 |
| `Error 804: forward compatibility` | 드라이버가 낮음. 위 A-3 참조 |
| `no kernel image is available` | GPU 아키텍처 미지원. 담당자 문의 |
| DataLoader 가 죽음 | `--ipc=host` 또는 `--shm-size=8g` |
| 학습이 이유 없이 느림 | 데이터가 `/mnt/` 에 있거나, VRAM 초과로 시스템 RAM 스필 |
