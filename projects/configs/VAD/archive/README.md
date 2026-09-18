# 폐기 config — 제출 라인이 아니다

여기 있는 config 들은 **2026-08-27 운영국 답변 이전에** 돌린 실험이며, 그 답변으로 금지가
확정된 입력 통로(자차 과거 상태·목표점을 planner 에 직접/단순 임베딩으로 입력 =
`ego_his_encoder` / `ego_lcf_feat_idx`)를 켠다. **폐기했고 제출에 쓰지 않는다.**
심사 시 제출 라인과 혼동되지 않도록 별도 디렉터리로 분리해 둔다.

| config | 무엇을 했나 | 왜 폐기했나 |
|---|---|---|
| `exp_jeung_0818_egostate.py` | 자차 속도를 planner 에 직접 입력 | 8/27 판정으로 구조 자체가 금지 |
| `exp_target_0824.py` | 목표점을 planner 에 직접 입력 | 같은 통로. 영상 가림 검사도 0.94배로 실패(영상을 안 봄) |
| `exp_target_vel_0824.py` | 목표점 + 속도를 planner 에 직접 입력 | 같은 통로. epoch 7 에서 중단 |

제출 라인은 `../exp_full_tgtbev.py` 이며, 목표점을 **BEV(공통 feature) 생성 단계의 조건
입력으로만** 쓴다. 정보 흐름과 판정 근거는 `docs/JUDGE_NOTES.md`, 경위는
`docs/WORKFLOW.md` 0장에 있다.
