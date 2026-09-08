#!/usr/bin/env bash
# 제출 JSON 생성 (docs/WORKFLOW.md 6장 순서 그대로): ablation -> FLOPs -> 추론 -> 형식 검사.
# 서브컴(3060) 실측: ablation 약 30분 + FLOPs 1분 + 추론 9분.
# 결과: work_dirs/<이름>/submission_<이름>_<ckpt>.json  (형식 검사 통과 시에만 제출)
#
# 컨테이너 "밖"(호스트 셸)에서 실행한다.
#   bash tools/make_submission.sh <work_dir 이름> <체크포인트 파일명> [config 이름=exp_full_tgtbev]
#   setsid nohup bash tools/make_submission.sh exp_full_tgtbev_4080 epoch_9.pth > /dev/null 2>&1 < /dev/null &
set -u
CTR=${ETRI_CONTAINER:-etri}
NAME=${1:?"work_dirs 아래 디렉터리 이름을 넣어라"}
CK=${2:?"체크포인트 파일명을 넣어라 (예: epoch_9.pth)"}
CFG=${3:-exp_full_tgtbev}
WD="work_dirs/$NAME"
HWD="$HOME/ETRI_Challenge/$WD"
CFGP="projects/configs/VAD/$CFG.py"
# 목표점 조건 모델은 반드시 tgt pkl(13차원). pre pkl 은 목표점이 없어 조용히 조건 없는 모델로 제출된다.
ANN="data/etri/vad_etri_tgt_infos_temporal_test.pkl"
TAG=${CK%.pth}
OUT="$WD/submission_${NAME}_${TAG}.json"
cd "$HOME/ETRI_Challenge/ETRI_E2E_Driving_Challenge"
LOG="$HWD/make_submission_${TAG}.log"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run() { docker exec "$CTR" sh -c "$1"; }

[ -f "$HWD/$CK" ] || { log "!! $HWD/$CK 없음"; exit 1; }
log "제출 생성 시작  dir=$NAME  ckpt=$CK  config=$CFG"

log "1/4 영상 ablation (약 30분)"
run "python tools/ablate_image.py $CFGP $WD/$CK --scenarios 10 > $WD/ablation_${TAG}.log 2>&1" \
  || { log "!! ablation 실패 - $HWD/ablation_${TAG}.log 확인"; exit 1; }
grep -E '배로 악화|제출하면 안 된다|판정을 보류|스크립트 문제' "$HWD/ablation_${TAG}.log" | tee -a "$LOG"
if grep -q '제출하면 안 된다\|판정을 보류\|스크립트 문제' "$HWD/ablation_${TAG}.log"; then
  log "!! ablation 불합격. 제출 생성을 중단한다."; exit 1
fi

log "2/4 FLOPs 측정"
run "python tools/measure_flops.py $CFGP --ann-file $ANN --out $OUT > $WD/flops_${TAG}.log 2>&1" \
  || { log "!! FLOPs 실패 - $HWD/flops_${TAG}.log 확인"; exit 1; }
grep -E 'FLOPs \(|wrote' "$HWD/flops_${TAG}.log" | tee -a "$LOG"

log "3/4 테스트 1,125 clip 추론 (약 10분)"
run "python tools/etri_test_submit.py $CFGP $WD/$CK --ann-file $ANN --out $OUT > $WD/submit_${TAG}.log 2>&1" \
  || { log "!! 추론 실패 - $HWD/submit_${TAG}.log 확인"; tail -3 "$HWD/submit_${TAG}.log"; exit 1; }
tail -1 "$HWD/submit_${TAG}.log" | tee -a "$LOG"

log "4/4 형식 검사"
python3 - "$HWD/submission_${NAME}_${TAG}.json" <<'EOF' | tee -a "$LOG"
import json, sys
d = json.load(open(sys.argv[1]))
fl = d.pop('__flops__', None)
bad = [k for k, v in d.items() if len(v) != 6 or any(len(p) != 2 for p in v)]
print('clip %d (1125 필요) | 형태 불량 %d | __flops__ %s (cut-off 7053 G)' % (len(d), len(bad), fl))
ok = len(d) == 1125 and not bad and fl is not None and fl <= 7053e9
print('형식 검사 ' + ('통과' if ok else '!! 실패'))
sys.exit(0 if ok else 1)
EOF
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] && log "완료: $HWD/submission_${NAME}_${TAG}.json  (제출 전 팀에 알릴 것)" || log "!! 형식 검사 실패. 제출 금지."
exit "$rc"
