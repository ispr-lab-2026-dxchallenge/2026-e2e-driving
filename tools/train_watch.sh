#!/usr/bin/env bash
# 장시간 학습 감시 + 자동 재개.
#
# 학습이 죽으면 latest.pth 에서 알아서 이어서 다시 띄운다.
# train.py 는 --resume-from 파일이 없으면 무시하므로(tools/train.py:158)
# 최초 실행과 재개에 같은 명령을 쓸 수 있다.
#
# 컨테이너 "밖"(호스트 셸 또는 WSL)에서 실행한다.
#
#   bash tools/train_watch.sh exp_baseline_fp16_b6
#   bash tools/train_watch.sh exp_baseline_fp16_b6 "data.samples_per_gpu=5"
#
# 백그라운드로 돌리려면:
#   setsid nohup bash tools/train_watch.sh <이름> > /dev/null 2>&1 < /dev/null &
set -u
NAME=${1:?"config 이름을 넣어라 (확장자 없이)"}
EXTRA=${2:-}
EPOCHS=${EPOCHS:-10}
MAX_RETRY=${MAX_RETRY:-20}
WD="work_dirs/$NAME"

docker exec etri mkdir -p "$WD"
WLOG="$WD/watch.log"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$HOME/ETRI_Challenge/$WLOG"; }

log "감시 시작  config=$NAME  epochs=$EPOCHS  extra='$EXTRA'"

for i in $(seq 1 "$MAX_RETRY"); do
  ITER=$(docker exec etri sh -c "grep -o 'Epoch \[[0-9]*\]\[[0-9]*/' $WD/train.log 2>/dev/null | tail -1" || true)
  log "시도 $i/$MAX_RETRY  (마지막 진행: ${ITER:-없음})"

  docker exec etri bash -c "python tools/train.py \
      projects/configs/VAD/$NAME.py \
      --work-dir $WD --no-validate \
      --resume-from $WD/latest.pth \
      --cfg-options runner.max_epochs=$EPOCHS total_epochs=$EPOCHS $EXTRA \
      >> $WD/train.log 2>&1"
  rc=$?

  if [ "$rc" -eq 0 ]; then
    log "정상 종료 (학습 완료)"
    exit 0
  fi

  # 체크포인트가 없으면 재시작해도 처음부터라 무한 반복이 된다. 확인 후 중단.
  if ! docker exec etri test -f "$WD/latest.pth"; then
    log "rc=$rc 로 죽었는데 latest.pth 가 없다. 첫 체크포인트 전에 실패한 것이므로 중단한다."
    docker exec etri tail -20 "$WD/train.log"
    exit 1
  fi

  log "rc=$rc 로 죽음. 60초 후 latest.pth 에서 재개한다."
  sleep 60
done

log "재시도 $MAX_RETRY 회 소진. 중단."
exit 1
