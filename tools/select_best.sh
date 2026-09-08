#!/usr/bin/env bash
# 체크포인트 풀에서 최선 고르기 (docs/WORKFLOW.md 6장).
# work_dirs/<이름>/epoch_N.pth 전부를 제출과 동일한 추론 설정(train_watch 자동 평가와 같은 인자)으로
# 평가하고 L2_avg 오름차순 표를 만든다. 이미 결과가 있는 epoch(eval_epoch_N.log 에 결과 줄)은
# 건너뛰므로 중단 후 다시 실행해도 된다. 4080 에서 받은 eval_epoch_N.log 를 미리 넣어 두면 그만큼 준다.
#
# 컨테이너 "밖"(호스트 셸)에서 실행한다. 서브컴(3060)은 체크포인트당 약 8분.
#   bash tools/select_best.sh <work_dir 이름> [config 이름=exp_full_tgtbev]
#   setsid nohup bash tools/select_best.sh exp_full_tgtbev_4080 > /dev/null 2>&1 < /dev/null &
set -u
CTR=${ETRI_CONTAINER:-etri}
NAME=${1:?"work_dirs 아래 디렉터리 이름을 넣어라"}
CFG=${2:-exp_full_tgtbev}
WD="work_dirs/$NAME"
HWD="$HOME/ETRI_Challenge/$WD"
cd "$HOME/ETRI_Challenge/ETRI_E2E_Driving_Challenge"
LOG="$HWD/select_best.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "선택 평가 시작  dir=$NAME  config=$CFG"
for N in $(seq 1 30); do
  CK="epoch_$N.pth"
  [ -f "$HWD/$CK" ] || continue
  EL="$HWD/eval_epoch_$N.log"
  if [ -f "$EL" ] && grep -q '^-10' "$EL"; then log "  $CK 결과 있음 - 건너뜀"; continue; fi
  log "  $CK 평가 중"
  if docker exec "$CTR" sh -c "python tools/eval_protocol.py projects/configs/VAD/$CFG.py $WD/$CK --scenarios 0 --eval-stride 10 --offsets=-10,-5,0 --history-bev-only --fp16 > $WD/eval_epoch_$N.log 2>&1"; then
    log "  결과 [$CK]: $(grep '^-10' "$EL" | head -1)"
  else
    log "  !! 평가 실패 [$CK] - $EL 확인"
  fi
done

echo
printf '%-12s %7s %7s %7s %9s\n' ckpt 1s 2s 3s L2_avg | tee "$HWD/select_best.txt"
for N in $(seq 1 30); do
  EL="$HWD/eval_epoch_$N.log"
  [ -f "$EL" ] || continue
  R=$(grep '^-10' "$EL" | head -1); [ -z "$R" ] && continue
  printf '%-12s %s\n' "epoch_$N" "$(echo "$R" | awk '{print $2, $3, $4, $5}')"
done | sort -k5 -n | tee -a "$HWD/select_best.txt"
echo
log "완료. 맨 윗줄(L2_avg 최소)이 후보. 다음: bash tools/make_submission.sh $NAME <epoch_N.pth>"
