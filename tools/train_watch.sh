#!/usr/bin/env bash
# 장시간 학습 감시 + 자동 재개.
#
# 학습이 죽으면 latest.pth 에서 알아서 이어서 다시 띄운다.
# train.py 는 --resume-from 파일이 없으면 무시하므로(tools/train.py:158)
# 최초 실행과 재개에 같은 명령을 쓸 수 있다.
#
# 컨테이너 "밖"(호스트 셸 또는 WSL)에서 실행한다.
#
# [주의] mmcv EpochBasedRunner 는 epoch 중간 체크포인트에서 재개하면
#        그 epoch 의 남은 부분을 건너뛰고 다음 epoch 으로 간다.
#        가중치는 보존되지만 재개 1회당 epoch 이 하나 줄어든다.
#        자세한 내용과 대응은 docs/WORKFLOW.md 5장 참조.
#
#   bash tools/train_watch.sh exp_baseline_fp16_b6
#   bash tools/train_watch.sh exp_baseline_fp16_b6 "data.samples_per_gpu=5"
#
# 백그라운드로 돌리려면:
#   setsid nohup bash tools/train_watch.sh <이름> > /dev/null 2>&1 < /dev/null &
set -u
# 컨테이너 이름이 다른 머신(예: 4080 은 etri2)은 이렇게 실행한다:
#   ETRI_CONTAINER=etri2 bash tools/train_watch.sh <config> "..."
CTR=${ETRI_CONTAINER:-etri}
NAME=${1:?"config 이름을 넣어라 (확장자 없이)"}
EXTRA=${2:-}
# EPOCHS 를 지정하면 config 의 total_epochs 를 덮어쓴다. 미지정이면 config 값 사용.
# (예전 기본값 10 이 스크리닝 config 의 4 를 조용히 덮어쓴 사고가 있었다)
EPOCHS=${EPOCHS:-}
MAX_RETRY=${MAX_RETRY:-20}
WD="work_dirs/$NAME"

# work_dir 은 호스트에서 만든다. 컨테이너(root)가 만들면 소유자가 root 가 되어
# 호스트 사용자가 watch.log 를 쓰지 못한다.
mkdir -p "$HOME/ETRI_Challenge/$WD"
docker exec "$CTR" mkdir -p "$WD"
WLOG="$WD/watch.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$HOME/ETRI_Challenge/$WLOG"; }

log "감시 시작  config=$NAME  epochs=${EPOCHS:-config값}  extra='$EXTRA'"

for i in $(seq 1 "$MAX_RETRY"); do
  ITER=$(docker exec "$CTR" sh -c "grep -o 'Epoch \[[0-9]*\]\[[0-9]*/' $WD/train.log 2>/dev/null | tail -1" || true)
  log "시도 $i/$MAX_RETRY  (마지막 진행: ${ITER:-없음})"

  EPOPT=""
  if [ -n "$EPOCHS" ]; then EPOPT="runner.max_epochs=${EPOCHS:-config값} total_epochs=${EPOCHS:-config값}"; fi
  docker exec "$CTR" bash -c "python tools/train.py \
      projects/configs/VAD/$NAME.py \
      --work-dir $WD --no-validate \
      --resume-from $WD/latest.pth \
      --cfg-options $EPOPT $EXTRA \
      >> $WD/train.log 2>&1"
  rc=$?

  if [ "$rc" -eq 0 ]; then
    log "정상 종료 (학습 완료)"
    # 자동 평가: 전체 val 39 시나리오, 제출과 동일한 추론 설정
    # 학습량 곡선까지 같이 뽑는다: epoch 4/7/최종 (RESULTS.md #10 과적합 판별)
    # epoch_N.pth 는 IterSafeCheckpointHook 의 epoch 경계 저장이 만든다.
    log "자동 평가 시작 (체크포인트별 10~25분)"
    for CK in epoch_4.pth epoch_7.pth latest.pth; do
      if ! docker exec "$CTR" test -f "$WD/$CK"; then log "  $CK 없음 - 건너뜀"; continue; fi
      EVCMD="python tools/eval_protocol.py projects/configs/VAD/$NAME.py $WD/$CK --scenarios 0 --eval-stride 10 --offsets=-10,-5,0 --history-bev-only --fp16"
      EVLOG="$WD/eval_${CK%.pth}.log"
      if docker exec "$CTR" sh -c "$EVCMD > $EVLOG 2>&1"; then
        RES=$(docker exec "$CTR" sh -c "grep -E '^-10' $EVLOG" | head -1)
        log "평가 결과 [$CK]: $RES"
      else
        log "!! 자동 평가 실패 [$CK] - $EVLOG 확인"
      fi
    done
    # ABLATION=1 일 때만: 영상 기여도 검사 (보조 입력 모델용)
    if [ "${ABLATION:-0}" = "1" ]; then
      log "영상 ablation 시작 (약 30분)"
      ABCMD="python tools/ablate_image.py projects/configs/VAD/$NAME.py $WD/latest.pth --scenarios 10"
      if docker exec "$CTR" sh -c "$ABCMD > $WD/ablation.log 2>&1"; then
        ABRES=$(docker exec "$CTR" sh -c "grep -E '배로 악화' $WD/ablation.log" | head -1)
        log "ablation: $ABRES"
      else
        log "!! ablation 실패 - $WD/ablation.log 확인"
      fi
    fi
    exit 0
  fi

  # 체크포인트가 없으면 재시작해도 처음부터라 무한 반복이 된다. 확인 후 중단.
  if ! docker exec "$CTR" test -f "$WD/latest.pth"; then
    log "rc=$rc 로 죽었는데 latest.pth 가 없다. 첫 체크포인트 전에 실패한 것이므로 중단한다."
    docker exec "$CTR" tail -20 "$WD/train.log"
    exit 1
  fi

  log "rc=$rc 로 죽음. 60초 후 latest.pth 에서 재개한다."
  sleep 60
done

log "재시도 $MAX_RETRY 회 소진. 중단."
exit 1
