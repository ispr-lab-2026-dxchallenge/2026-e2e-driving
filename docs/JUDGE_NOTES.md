# 2차 심사용 설명서 — 제출 모델의 구조와 정보 흐름

ETRI 2026 자율주행 AI 챌린지 / E2E Driving 트랙. 이 문서는 심사위원이 제출 코드를 열었을 때
확인할 사항을 미리 정리한 것이다. **베이스라인(제공된 VAD) 대비 변경은 파일 2개 약 25줄과
config 3줄이 전부**이고, 그 변경이 규정의 어느 판정에 해당하는지를 원문과 함께 밝힌다.
불리한 사실도 숨기지 않는다(§5).

## 1. 한 줄 요약

> 5초 목표점을 **BEV(공통 feature) 생성 단계의 조건 입력**으로만 쓴다. planner 는 목표점도
> 자차 상태도 수치로 받지 않는다. 최종 궤적은 영상에서 만들어진 BEV → 객체·지도 query 를
> 거쳐 나온다.

목표점이 모델에 들어가는 자리는 **단 한 곳**이다 (`VAD_transformer.py:287-296`).
그 바로 윗줄(`:282-286`)이 베이스라인이 기본으로 켜 두는 can_bus(자차 속도·가속도·요레이트)
주입이며, 우리는 **같은 자리에 같은 형태로** 목표점을 더한다.

```python
# VAD_transformer.py:282-296  (윗 3줄이 베이스라인, 아랫부분이 우리 추가분)
can_bus = self.can_bus_mlp(can_bus)[None, :, :]
bev_queries = bev_queries + can_bus * self.use_can_bus          # 베이스라인 기본값 True
if self.use_target_cond:
    tp = kwargs.get('target_point', None)                       # 5초 목표점 (2차원)
    if tp is not None:
        t_emb = self.target_mlp(tp)[None, :, :]
        if self.training and float(torch.rand(1)) < self.target_cond_drop:
            t_emb = t_emb * 0.0                                 # condition dropout 0.5
        bev_queries = bev_queries + t_emb
```

## 2. 정보 흐름 표 (줄 번호는 제출 코드 기준)

| 정보 | 들어가는 자리 | planner 에 수치로 들어가는가 | 근거 |
|---|---|---|---|
| 카메라 영상 6대 | backbone → FPN → BEV encoder | — (모든 출력의 출처) | 베이스라인 |
| 자차 상태 (can_bus: 속도·가속도·요레이트, pose 차분) | BEV query 가산 `VAD_transformer.py:283-286` | **아니오** (BEV 경유) | 베이스라인 기본값 `use_can_bus=True` |
| **5초 목표점** | BEV query 가산 `VAD_transformer.py:287-296` | **아니오** (BEV 경유) | 우리 추가분 |
| 과거 궤적 `ego_his_trajs` | 학습 손실에만 (§5) | **아니오** | — |
| `ego_lcf_feat` (자차 상태 벡터) | 목표점 2차원을 꺼내는 용도 `VAD_head.py:602-603` | **아니오** | 나머지 11차원은 미사용 |

**금지 통로는 꺼져 있다.** 운영국이 2026-08-27 에 "베이스라인이 지원하지만 기본으로 쓰지 않는
구조"로 지목한 두 통로는 기본값 `None` 그대로다 (`VAD_head.py:137` `ego_his_encoder=None`,
`:147` `ego_lcf_feat_idx=None`). 이 값이 `None` 이면 planner 입력은
`[ego_agent_query, ego_map_query]` 두 개뿐이다 (`VAD_head.py:855-860`). 제출 config 에도
명시해 두었다.

**planner 블록 안에는 목표점 참조가 없다.** `VAD_head.py:785-876` (ego_query 생성 →
ego_agent_decoder → ego_map_decoder → ego_fut_decoder) 구간에서 `target_point`,
`ego_lcf_feat`, `ego_his_trajs` 를 읽는 코드는 없다. `grep -n "target_point" VAD_head.py`
결과는 `:602-603`(BEV 로 넘기려고 꺼내는 곳) 한 군데다.

## 3. 규정 판정과의 대조 (운영국 답변 원문)

| 날짜 | 운영국 답변 (발췌) | 우리 구성 |
|---|---|---|
| 2026-08-19 | "자차의 과거 상태와 목표 지점만으로 궤적이 결정되는 구성은 신경망이어도 금지. 최종 궤적은 영상 인식 결과에서 나와야 하고 과거 상태와 목표 지점은 보조 입력에 한한다" | 영상을 0 으로 가리면 L2 가 1.43~2.08배 악화된다(§4). 목표점을 지워도 1.27배에 그친다 — 영상이 주다 |
| 2026-08-25 | "목표점을 condition 입력으로 (방향·후보 선택에 영향) — 허용" / "목표점 + 자차 상태만으로 최종 궤적 생성 — 금지" | 조건 입력으로만 쓴다. 후처리 보정·하이브리드 없음(§5) |
| 2026-08-27 | "판정 기준은 과거 궤적이나 status 가 **planner 에 직접 또는 단순 임베딩으로 입력**되어 최종궤적을 출력하는 여부 ... 여러 task 의 공통 특징을 향상하는 등의 **간접적 활용은 허용**" | planner 직접 입력 없음. BEV 는 검출·지도·플래닝이 공유하는 공통 feature 다 |
| 2026-08-28 | (질의) "target point 를 planner 입력이 아닌 **BEV/scene feature 형성 단계의 조건 입력**으로 사용하는 간접적 활용은 허용되는지" → (답변) "**네 이해하신 내용이 맞습니다**" | 정확히 이 구성이다 |
| 2026-09-02 | (타 팀의 goal-as-query 질의를 불허하며) "goal 은 **planner 의 경로가 아닌 인식이나 공통 feature 등의 향상 단계에서 활용**하시거나 후보 궤적 중에서 선택하는 방식으로 권장" | 권장된 자리에 넣었다. planner 내부(query/key/value/bias 어디에도) 목표점이 없다 |

## 4. 영상 의존성 측정 (ablation)

제출 프로토콜과 **똑같은 조건**(3프레임 −10/−5/0, history-BEV-only, fp16, 전체 검증 1,053점)에서
입력을 교란해 측정했다 (`tools/ablate_window.py`).

| 교란 | L2_avg | 배율 |
|---|---|---|
| 없음 | 0.2941 | 1.00 |
| 영상 전체를 0 으로 | 0.4199 | **1.43배** |
| 영상을 다른 프레임으로 섞음 | 0.3543 | 1.20배 |
| 목표점 제거 | 0.3740 | 1.27배 |
| 목표점을 다른 것으로 교체 | 1.1044 | 3.75배 |

긴 프로토콜(300프레임 연속 full forward, `tools/ablate_image.py`)에서는 영상 0 이 **2.08배**다.
참고로 목표점을 쓰지 않는 순수 비전 모델의 같은 프로토콜 영상 0 배율이 2.05배이므로, 우리
모델의 영상 의존도는 순수 비전 모델과 같은 수준이다.

**단위 테스트** (`tests/`): ① `target_point=None` 이면 순수 비전 모델과 출력이 완전히 같다
② BEV 를 고정하면 목표점을 바꿔도 궤적이 바뀌지 않는다 ③ planner·decoder 모듈에
`target_point` 가 전달되지 않는다.

## 5. 숨기지 않는 사실 3가지

1. **목표점 정보는 BEV 를 거쳐 planner 출력에 간접 전파된다.** 그것이 조건화의 목적이다.
   "planner 가 목표점의 영향을 전혀 받지 않는다"고 주장하지 않는다. 우리가 주장하는 것은
   "목표점이 planner 모듈의 입력이 아니다"이며, 이는 2026-08-28·09-02 답변이 허용한 형태다.
2. **자차 상태(can_bus)도 같은 경로로 BEV 에 들어간다.** 이는 베이스라인 기본 동작이며
   (`use_can_bus=True`), 우리가 켠 것이 아니다. "테스트 데이터로부터 자차 상태를 전혀
   입력받지 않는다"는 표현은 쓰지 않는다.
3. **제출 체크포인트의 planner 꼬리는 VO(영상 기반 자차 운동 추정) 헤드와 함께 학습됐다.**
   그 헤드는 현재/직전 프레임의 FPN **영상 특징만** 입력으로 받으며(can_bus·과거 궤적
   무접촉, 출처 테스트 4종 `tests/test_vo_provenance.py` 통과), 2026-09-08 답변이 허용한
   형태다. 다만 **추론 시에는 사용하지 않는다** — 제출 config(`exp_full_tgtbev.py`)는 VO
   모듈을 아예 만들지 않고, 제출 체크포인트에서 `vo_*` 가중치는 제거했다. 학습 중 그 출력을
   확률적으로(p=0.3) 넣었다 뺐다 한 것이 정규화로 작용해 검증 L2 가 0.326 → 0.298 로
   좋아졌고, 그 효과만 남은 상태다(경위는 `docs/RESULTS.md` 10장).

## 6. 제출 모델 구성

- 학습: 전체 train split(337 시나리오) 10 epoch → epoch 5·6 가중치 평균(model soup)으로
  트렁크 확정 → 트렁크 동결 후 planner 꼬리만 2 epoch 미세조정(`tools/train.py` 의
  `train_only`) → 서로 다른 실행의 꼬리 3개를 가중치 평균.
- 가중치 평균은 구조·입력·FLOPs·추론 시간이 단일 모델과 동일하다(앙상블 아님).
- 추론: 3프레임(−10/−5/0), 이력 프레임은 BEV 생성까지만, fp16. 모두 2026-08-25 운영국
  허용 항목이다. FLOPs 479.4 GFLOPs (컷오프 7,053G).
- 검증 L2_avg 0.2941 / 리더보드 환산 약 0.274 (검증 − 0.02, 제출 3회로 확인).

## 7. 재현

```bash
# 1) 검증 점수 재현 (제출과 동일 프로토콜)
python tools/eval_protocol.py projects/configs/VAD/exp_full_tgtbev.py <ckpt> \
    --scenarios 0 --eval-stride 10 --offsets=-10,-5,0 --history-bev-only --fp16
# 2) 영상 의존성 재측정
python tools/ablate_window.py projects/configs/VAD/exp_full_tgtbev.py <ckpt> --scenarios 0
# 3) 제출 JSON 생성 (ablation → FLOPs → 추론 → 형식 검사)
bash tools/make_submission.sh <work_dir 이름> <ckpt 파일명> exp_full_tgtbev
```

상세한 실험 경위와 기각된 시도는 `docs/RESULTS.md`, 규정 판정 대장은 `docs/WORKFLOW.md` 0장에 있다.
