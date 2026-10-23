# results/ 폴더 안내

표(CSV/JSON)와 그림만 커밋한다. 이벤트별 예측(`preds_*.csv`, `inner_preds.csv.gz`), 시퀀스(`*.npz`), 대부분의 실행 로그는
gitignore다. 수치의 해석과 확인/추측 구분은 `docs/FINDINGS.md`의 해당 절을 본다.

| 폴더 | 내용 | FINDINGS 절 | 만든 스크립트 |
|---|---|---|---|
| `figures/` | 논문용 그림(300 dpi): 경계(a), t_r 민감도(b), 학습 곡선(c), 교차로별 경계(d), 리드타임(l_a–l_c), 두 단계 판단(n), 통합 검증(p) | F, L, N, P | `paper_materials.py`, `leadtime_dz_figures.py`, `staged_dz_figure.py`, `pooled_eval.py` |
| `figures/data/` | 그림 데이터 CSV와 D절 표(`table_D_*.csv`) | F, D | 같음 |
| `fluid_fi/` | FLUID FI 151건 벤치마크(규칙 A/B, LR, MLP, LSTM, Transformer), 오프셋 민감도, 그림 1–4 | FLUID FI 벤치마크 | `run_benchmark.py` |
| `fluid_fi/leader_follower/` | 선두/추종 분리 평가, 위험 쌍; `leader/`의 `model_compare`, `noise`, `seq_dva`, `external_sind_sample` 하위 실험 | 선두/추종, 모델 비교, 잡음, 외부 시험 | `leader_follower.py`, `model_compare.py`, `noise_robustness.py`, `seq_dva.py`, `external_sind_sample.py` |
| `fluid_fi/leadtime_dz/` | FLUID 띠 26건 리드타임 표 | L | `leadtime_dz.py` |
| `v2xseq/` | V2X-Seq 정지선 보정표 `stopline_shift.csv` (10 % 분위), Q2 민감도용 `stopline_shift_q05.csv`, `_q20.csv` | V2X-Seq 적합성, Q | `v2xseq_calibrate.py` |
| `v2xseq/review/` | 교차로별 선별표 `per_site.csv` (이벤트별 라벨 감사표는 미포함, README 6절로 재생성) | V2X-Seq 적합성 | `v2xseq_review.py`, `v2xseq_ambiguous_go.py` |
| `v2xseq/eval/v3`, `v4`, `v4_excl95` | 외부 시험·LOIO·민감도·ablation(v3 = 정지선 보정 전, v4 = 보정 후, v4_excl95 = 선을 6 m 안에서 넘고 멈춘 애매한 통과 선두 70건 제외) | V2X-Seq 평가 | `v2xseq_eval.py`, `v2xseq_summary.py` |
| `v2xseq/eval/v4/followup/` | 후속 A(가속도)·B(딜레마존)·C(t_r) | V2X-Seq 후속 | `v2xseq_followup.py` |
| `v2xseq/eval/v4/calibrated_rule/` | 23폴드 보정 규칙 A, 쌍별 비교, t_r 고정 민감도 | D | `v2xseq_followup.py --tasks D` |
| `v2xseq/leadtime_dz/` | V2X-Seq 띠 509건 리드타임·확률 분석 | L | `leadtime_dz.py` |
| `v2xseq/staged/` | 두 단계 판단(N, N2) | N, N2 | `staged_dz.py`, `staged_n2.py` |
| `v2xseq/v11/` | 논문 v1.1 보완(Q): 라벨 경계 사례(q1), 정지선 보정 민감도(q2), θ 민감도(q3), 확정 시점의 여유(q4), 위험 오류 9건의 여유(q6) (이벤트별 표는 미포함) | Q | `v2xseq_v11.py`, `v2xseq_calibrate.py --lead-q`, `v2xseq_extract.py` |
| `sind/` | SinD 공개 샘플 4모델 비교, 지도 겹침 그림 (이벤트표는 미포함, README 6절로 재생성) | SinD | `run_sind.py`, `sind_check_alignment.py` |
| `sind/full/` | SinD v2.0 전체 선별, 창춘 선두 102건 경계·평가 | G | `sind_full.py` |
| `womd/w0/` | WOMD 타당성 점검(샤드 2개) 요약 (이벤트·트랙별 표는 미포함, README 6절로 재생성) | W0 | `womd_w0.py` |
| `womd/w1/` | WOMD 전체 추출 요약, 최종 평가(`final/gkf`, `final/loio_int30`, `final/loio_int10`), validation 중간 결과 | W1, W1 최종 | `womd_w1_*.py` |
| `womd/w1_leads/` | 리드타임 포함 재추출 요약 | L | `womd_w1_run.py --leads`, `womd_w1_aggregate.py` |
| `womd/leadtime_dz/` | WOMD 띠 1,055건 리드타임·확률 분석 | L | `leadtime_dz.py` |
| `womd/staged/` | 두 단계 판단(N, N2) | N, N2 | `staged_dz.py`, `staged_n2.py` |
| `pooled/` | 네 데이터 통합 검증: 구성(P1), 경계(1단계), 합동 LOIO(P2), 데이터셋 하나 빼기(P3), 데이터셋별 비교 | P | `pooled_eval.py` |
| `pooled/staged/` | 통합 띠 1,605건 두 단계 판단(N2) | N2 | `staged_n2.py` |
| `pooled/fix/` | P-fix: 비WOMD t_50 보정 규칙 A, 시점별 LODO vs 자체 LOIO, WOMD 제외 합동 LOIO | P-fix | `pooled_fix.py`, `pooled_eval.py` |
| `pooled/fix/sind_self_leadtime_dz/` | SinD 창춘 자체 LOIO(녹화 5폴드) 리드타임 | P-fix | `leadtime_dz.py` |
| `pooled/fix/staged_no_womd_fix/` | WOMD 제외 통합 띠 550건 두 단계 판단 | P-fix | `staged_n2.py` |
| `ngsim_peachtree/` | NGSIM Peachtree 이벤트·4모델 비교, 신호 오프셋 k = −1.5 s 민감도, 녹색 출발 점검 (제외 데이터) | NGSIM | `run_ngsim.py`, `ngsim_green_start.py` |
| `ngsim_lankershim/` | NGSIM Lankershim int2 점검·평가 (제외 데이터) | NGSIM | `lankershim_check.py`, `lankershim_eval.py` |
| `cctv_pilot/` | AI Hub 영상 1개 시퀀스의 추적 품질 요약(집계값만; 원 영상·프레임은 재배포 금지라 미포함) | `docs/E_cctv_pilot.md` | `cctv_pilot.py`, `cctv_pilot_analyze.py` |
