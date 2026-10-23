# 신호 정보를 이용한 황색 시작 시점 선행 차량의 정지·통과 예측

**English summary**
- Task: an SPaT-equipped automated vehicle predicts whether the human-driven vehicle ahead of it in its lane stops or goes at yellow onset.
- Threshold: using the SPaT yellow as the threshold gives macro-F1 0.673 on 1,867 V2X-Seq leaders; the drivers' boundary is d/v ≈ 2.67 s (2.63–2.70 s over 23 folds, 2.662 s pooled over three datasets), and calibrating the rule to it gives 0.831.
- Dilemma zone: a two-stage decision (commit early only when P ≤ 0.05 or ≥ 0.95, otherwise at 2.0 s) matches waiting until 2.0 s (Δ −0.009 / +0.002, n.s.) with a mean commit time of 1.41 s / 1.445 s.

**한 줄 결론: 앞차 예측은 신호가 아니라 운전자를 기준으로, 한 번이 아니라 두 단계로.**

모든 수치의 출처와 검증 등급(확인/추측)은 [`docs/FINDINGS.md`](docs/FINDINGS.md)에 있다. 이 README의 숫자는 FINDINGS와
`results/` 파일에서 그대로 옮겼다.

## 1. 문제 정의

- C-ITS **SPaT**(신호 위상·잔여시간)를 받는 자율주행차가, 같은 차로 바로 앞의 **사람이 운전하는 선행 차량**이
  황색 시작 시점에 **정지할지 통과할지**를 예측한다.
- 이벤트: 황색 시작 시각에 정지선 상류에 있는 직진 차량 1대 = 이벤트 1건. 같은 차로에 정지선을 아직 넘지 않은 앞차가
  없는 차량을 **선두(leader)**로 보고, 선두만 평가한다(추종 차량은 FLUID에서 선두가 서면 33/33 모두 섰다).
- 입력: 정지선까지 거리 d, 속도 v, 가속도 a, 황색 잔여시간 t_r(SPaT), 2 s 이력.
- 비교 대상
  - **규칙 A**: d/v < t_r 이면 통과 (t_r = 신호 기록의 황색 길이).
  - **보정 규칙 A**: d/v < t_50 이면 통과. t_50은 학습 교차로에서 d/v 단변량 로지스틱의 50 % 경계(−β₀/β₁, d/v ≤ 12 s).
  - 학습 모델: LR(d, v, a), 띠 전용 LR(d, v, a), MLP(d, v, a), LSTM(d, v, a 또는 x, y, v, a).
- **딜레마존(띠)**: 황색 시작 시점 d/v가 2.5–5.5 s인 이벤트.
- 평가: 교차로 하나 빼기(LOIO) 교차 검증, macro-F1, 주기 단위 부트스트랩 95 % CI(WOMD는 영역 단위).

## 2. 데이터

**원자료는 저장소에 포함하지 않는다.** 각 배포처에서 받아 `data/raw/` 아래에 둔다(`data/raw/`는 gitignore). 이벤트별
궤적·예측 파일과, 이 데이터들에서 만든 이벤트별 요약표도 커밋하지 않는다(재생성 방법은 6절).

| 데이터 | 역할 | 규모 (선두 이벤트) | 입수 방법 | 라이선스 | 배치 경로 |
|---|---|---|---|---|---|
| **V2X-Seq TFD** (DAIR-V2X-Seq, 베이징 이좡, 노변 라이다·카메라) | 주 데이터 | 23개 교차로, 1,867건 (띠 509) | DAIR-V2X 배포처에 신청 후 다운로드 (분할 7z) | 배포처 이용 약관을 따름 ([DAIR-V2X-Seq](https://github.com/AIR-THU/DAIR-V2X-Seq)) | `data/raw/v2xseq/full/extracted/V2X-Seq-TFD/` |
| **FLUID** (중국 쉬안청, 드론) | 보조 | FI 교차로 1곳, 88건 (띠 26) | figshare doi `10.6084/m9.figshare.29974954` v2, `derived_data` 64179079 (TI), 64179085 (FI), 64179082 (FIDRT) | CC BY 4.0 | `data/raw/fluid/{TI,FI,FIDRT}/` |
| **SinD v2.0** (중국 6개 교차로, 드론) | 보조 | 창춘 1곳, 102건 (띠 15) | 전체본은 SOTIF-AVLab에 신청, 공개 샘플은 `github.com/SOTIF-AVLab/SinD` | 비상업·신청제 | `data/raw/sind/full/SinD_v2.0_CSV_Dataset/`, 샘플 `data/raw/sind/` |
| **WOMD v1.3.1** (Waymo Open Motion, 차량 탑재 인지) | 보조 (경계 추정에서 제외) | 808개 교차로, 3,527건 (띠 1,055) | `gs://waymo_open_dataset_motion_v_1_3_1/uncompressed/scenario/` (`gsutil`, 약관 동의 후), 신호 보충 자료 `github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement` | 배포처 이용 약관을 따름 ([Waymo Open Dataset](https://waymo.com/open/)) | `data/raw/womd/`, `data/raw/womd/michigan_tls/` |

- WOMD를 경계 추정에서 뺀 이유: 띠 안 후보의 68 %가 센서 범위 때문에 라벨을 받지 못해 남은 표본이 빨리 결판나는
  차량 쪽으로 치우친다. 또 운전자 경계(4.1 s)가 황색(4.1 s)과 같아 보정 효과를 볼 수 없다
  ([W1 최종](docs/FINDINGS.md#w1-최종--womd-v131-전체-validation--training-2026-10-15)).
- 데이터별 맞춤 방법(센서, 샘플링, t_r 산출, 정지선 기준, 폴드 단위)은
  [P절 표](docs/FINDINGS.md#p-전체-데이터-통합-검증--v2x-seq--fluid--sind--womd-2026-10-15)에 있다. 네 데이터 모두 10 Hz이고
  재샘플하지 않았다.

**제외한 데이터**

| 데이터 | 제외 이유 |
|---|---|
| NGSIM (Peachtree, Lankershim) | Peachtree는 신호 로그와 궤적의 시계 차이가 ±1.5 s까지만 좁혀지고, 이 차이가 라벨 분포와 모델 순위를 바꾼다. Lankershim은 공식 배포 ZIP이 서버에서 잘려 있고(Wayback 사본만 온전), 신호 시각이 약 ±4 s 불확실하며 교차로 1곳(int2)만 맞는다. |
| Waterloo Multi-Agent: Intersection | 정지선 앞 가시 거리가 9–33 m라 황색 시점 접근 차량의 2 s 이력과 d/v 2.5–5.5 s 띠를 담지 못한다. 황색 이벤트 7건. 이용 조건 문구가 데이터셋 페이지에 없다. |
| DRIFT (The-DRIFT, 한국) | 신호 기록이 없다. |

NGSIM·Waterloo의 상세 점검은 FINDINGS 앞부분("NGSIM", "데이터 후보")에 있다. NGSIM 입수 경로는 아래 "부록 A"에 남겼다.

## 3. 핵심 결과

### a) 운전자 경계: 황색 길이가 아니라 d/v ≈ 2.7 s

| 항목 | 값 | 출처 |
|---|---|---|
| V2X-Seq t_50 (1,862건, 통과 1,292) | **2.674 s** [2.585, 2.764] (기록 황색 4.0 s) | P 1단계 |
| V2X-Seq 23폴드 LOIO의 폴드별 t_50 | **2.63–2.70 s** (평균 2.674, 표준편차 0.016) | D |
| FLUID t_50 (88건) | 2.437 s [1.924, 3.146] (황색 3.0 s) | P 1단계 |
| SinD 창춘 t_50 (102건) | 2.722 s [2.170, 3.502] (황색 3.0 s) | P 1단계 |
| **3개 통합 (V2X-Seq + FLUID + SinD, 2,052건)** | **2.662 s [2.580, 2.751]** | P 1단계 |
| 데이터셋 더미 LR 검정 (기준 V2X-Seq) | 절편 p = 0.064, 기울기 p = 0.053, 절편 + 기울기 p = 0.022 | P 1단계 |
| (참고) WOMD 포함 4개 | 3.506 s [3.348, 3.661] | P 1단계 |

![V2X-Seq 선두의 d/v별 통과 확률과 50 % 경계](results/figures/fig_a_go_probability_vs_tt.png)

### b) 규칙 A vs 보정 규칙 A vs 학습 모델 (V2X-Seq 23폴드 LOIO)

macro-F1, 괄호는 보정 규칙 A 대비 차이 [95 % CI]. 출처: [D절](docs/FINDINGS.md#d-보정-규칙-a의-공정-평가--23폴드-loio-2026-10-13).

| 모델 | 전체 (1,867) | 딜레마존 (509) |
|---|---|---|
| 규칙 A, 기록 t_r | 0.673 (−0.159 [−0.183, −0.132]) | 0.479 (−0.130 [−0.196, −0.061]) |
| **보정 규칙 A, t_r\*** | **0.831** [0.812, 0.849] | **0.611** [0.560, 0.662] |
| 띠 전용 LR (d, v, a) | — | 0.650 (+0.040 [−0.014, +0.096]) |
| LR (d, v, a) 전체 학습 | 0.847 (+0.016 [−0.003, +0.033]) | 0.500 (−0.111 [−0.166, −0.055]) |
| MLP (d, v, a) | 0.862 (+0.030 [+0.015, +0.046]) | 0.654 (+0.044 [−0.004, +0.093]) |
| LSTM (x, y, v, a) | 0.855 (+0.024 [+0.006, +0.045]) | 0.614 (+0.003 [−0.053, +0.062]) |
| LSTM (d, v, a) | 0.845 (+0.014 [−0.005, +0.032]) | 0.636 (+0.027 [−0.028, +0.087]) |

### c) 리드타임별 딜레마존 macro-F1 (황색 시작 후 +0 ~ +2 s)

띠 소속은 황색 시작 시점에 고정하고, 예측 시점 L의 (d, v, a)와 2 s 이력을 쓴다. 괄호는 보정 규칙 A 대비, \* = 95 % CI가 0
제외. 출처: [L절](docs/FINDINGS.md#l-딜레마존-리드타임--확률-분석-2026-10-15).

| 데이터 · 모델 | +0 s | +0.5 s | +1.0 s | +1.5 s | +2.0 s |
|---|---|---|---|---|---|
| V2X-Seq 보정 규칙 A (t50(L) = 2.67/2.23/1.84/1.45/1.09 s) | 0.611 | 0.632 | 0.666 | 0.732 | 0.788 |
| V2X-Seq 띠 전용 LR | 0.650 (+0.040) | 0.718 (+0.088\*) | 0.744 (+0.079\*) | 0.825 (+0.095\*) | 0.853 (+0.065\*) |
| V2X-Seq MLP | 0.651 (+0.042) | 0.710 (+0.079\*) | 0.733 (+0.068\*) | 0.854 (+0.123\*) | 0.878 (+0.091\*) |
| V2X-Seq LSTM | 0.631 (+0.023) | 0.713 (+0.082\*) | 0.724 (+0.058\*) | 0.861 (+0.130\*) | 0.858 (+0.071\*) |
| WOMD 보정 규칙 A (t50(L) = 4.11/3.36/2.80/2.30/1.90 s) | 0.538 | 0.607 | 0.692 | 0.753 | 0.834 |
| WOMD 띠 전용 LR | 0.686 (+0.149\*) | 0.727 (+0.121\*) | 0.853 (+0.162\*) | 0.902 (+0.151\*) | 0.938 (+0.105\*) |
| WOMD MLP | 0.706 (+0.170\*) | 0.759 (+0.154\*) | 0.871 (+0.180\*) | 0.920 (+0.169\*) | 0.946 (+0.113\*) |
| WOMD LSTM | 0.652 (+0.114\*) | 0.785 (+0.180\*) | 0.873 (+0.183\*) | 0.920 (+0.168\*) | 0.941 (+0.108\*) |

V2X-Seq 띠 509건(+2.0 s 508건), WOMD 띠 1,055건(+2.0 s 1,032건). FLUID 26건 값은 L절에 있다.

![리드타임별 딜레마존 macro-F1](results/figures/fig_l_a_leadtime_f1.png)

### d) 두 단계 판단, θ = 0.05 (N2)

각 시점(0, 0.5, 1.0, 1.5 s)에 P ≤ θ 또는 P ≥ 1 − θ이면 그 시점에 확정하고, 아니면 2.0 s에 0.5 임계로 확정한다.
위험 오류 = 실제 정지를 통과로 확정한 비율(실제 정지 대비), 보수 오류 = 실제 통과를 정지로 확정한 비율(실제 통과 대비).
출처: [N2절](docs/FINDINGS.md#n2-단계적-판단-보완--θ-선택-규칙-수정--평균-대기-맞춤-비교--θ005-고정-2026-10-15),
[P-fix 3](docs/FINDINGS.md#p-fix-통합-검증-보완-2026-10-16).

| 데이터 · 모델 (띠 n) | macro-F1 [95 % CI] | Δ vs 2.0 s 일괄 [95 % CI] | 위험 오류 | 보수 오류 | 평균 확정 시각 |
|---|---|---|---|---|---|
| V2X-Seq · MLP (509) | 0.870 [0.831, 0.905] | −0.009 [−0.022, +0.004] | 0.024 | 0.288 | 1.41 s |
| 3개 통합 (V2X-Seq + FLUID + SinD) · MLP (550) | 0.880 [0.844, 0.911] | +0.002 [−0.010, +0.015] | 0.030 | 0.250 | 1.445 s |

![두 단계 판단: 평균 확정 시각 vs macro-F1](results/figures/fig_n_staged_tradeoff.png)

### e) 데이터셋 간 전이 (P-fix, 데이터셋 하나 빼기, 띠 macro-F1)

LODO = 나머지 세 데이터셋으로 학습, 자체 = 시험 데이터셋 안의 LOIO. 보정 규칙 A의 t_50은 학습 데이터셋 중 WOMD가 아닌
이벤트로 추정했다. 규칙 A는 학습이 없어 두 값이 같다. 95 % CI는
[P-fix 2](docs/FINDINGS.md#p-fix-통합-검증-보완-2026-10-16)와 `results/pooled/fix/P3_lodo_band_by_lead_fix.csv`에 있다.

**V2X-Seq + FLUID + SinD (중국) 학습 → WOMD 시험** (띠 n = 1,055 / 1,055 / 1,054 / 1,052 / 1,032; t_50 = 2.662 / 2.217 / 1.818 / 1.443 / 1.083 s)

| 모델 (LODO / 자체) | +0 s | +0.5 s | +1.0 s | +1.5 s | +2.0 s |
|---|---|---|---|---|---|
| MLP | 0.422 / 0.706 | 0.606 / 0.759 | 0.710 / 0.871 | 0.838 / 0.920 | 0.909 / 0.946 |
| LSTM | 0.491 / 0.652 | 0.589 / 0.785 | 0.588 / 0.873 | 0.846 / 0.920 | 0.921 / 0.941 |
| 띠 전용 LR | 0.649 / 0.686 | 0.687 / 0.727 | 0.744 / 0.853 | 0.883 / 0.902 | 0.925 / 0.938 |
| 보정 규칙 A | 0.418 / 0.538 | 0.500 / 0.607 | 0.583 / 0.692 | 0.652 / 0.753 | 0.703 / 0.834 |
| 규칙 A | 0.560 | 0.569 | 0.597 | 0.659 | 0.729 |

**FLUID + SinD + WOMD (WOMD 중심) 학습 → V2X-Seq 시험** (띠 n = 509 / 509 / 509 / 509 / 508; t_50 = 2.571 / 2.077 / 1.681 / 1.352 / 1.001 s)

| 모델 (LODO / 자체) | +0 s | +0.5 s | +1.0 s | +1.5 s | +2.0 s |
|---|---|---|---|---|---|
| MLP | 0.618 / 0.651 | 0.642 / 0.710 | 0.662 / 0.733 | 0.769 / 0.854 | 0.752 / 0.878 |
| LSTM | 0.563 / 0.631 | 0.632 / 0.713 | 0.638 / 0.724 | 0.743 / 0.861 | 0.781 / 0.858 |
| 띠 전용 LR | 0.623 / 0.650 | 0.662 / 0.718 | 0.671 / 0.744 | 0.789 / 0.825 | 0.787 / 0.853 |
| 보정 규칙 A | 0.515 / 0.611 | 0.559 / 0.632 | 0.602 / 0.666 | 0.693 / 0.732 | 0.790 / 0.788 |
| 규칙 A | 0.479 | 0.481 | 0.545 | 0.635 | 0.732 |

네 데이터 합동 LOIO(35폴드)의 시점별 띠 macro-F1은 아래 그림과
[P절](docs/FINDINGS.md#p-전체-데이터-통합-검증--v2x-seq--fluid--sind--womd-2026-10-15)에 있다.

![통합 검증: 시점별 딜레마존 macro-F1 (통합과 데이터셋별)](results/figures/fig_p_pooled_leadtime_band.png)

## 4. 한계

- **실차 검증이 없다.** 모든 결과는 기록된 궤적으로 한 오프라인 평가다.
- **중국 자료 중심이다.** 경계 추정에 쓴 세 데이터(V2X-Seq, FLUID, SinD)는 모두 중국 도시(베이징, 쉬안청, 창춘)다. FLUID와
  SinD는 교차로 1곳씩(88건, 102건)이라 CI가 넓다.
- **WOMD 라벨 결손 편향.** 띠 안 후보의 68 %가 센서 범위 때문에 라벨을 받지 못했다. 남은 표본은 빨리 결판나는 차량 쪽으로
  치우쳐 있어, WOMD의 경계·보정 수치는 참고치다.
- **국내 적용에는 경계 재추정이 필요하다.** 운전자 경계는 데이터마다 다르고(t_50 2.437–2.722 s, WOMD 4.1 s), 국내 교차로 자료로는
  추정하지 않았다.
- 그 밖에 V2X-Seq 라벨이 접근로별 정지선 보정과 10–45 s의 라벨 지평에 의존하고, 노변 센서의 가속도에 잡음이 있다(FINDINGS
  "연구 질문에 대한 현재 답").

## 5. 실험 이력: 88건에서 기각한 가설이 1,867건에서 바뀐 과정

FLUID 선두 88건에서는 LR(d, v, a)이 규칙 A를 +0.185 이겼다. 운전자 경계(2.44 s [1.94, 3.04])가 황색 3 s와 가까워 경계를
보정해도 이득이 n.s.(+0.062 [−0.017, +0.140])였고, a를 빼면 이득이 사라져 "이득은 경계가 아니라 가속도에서 온다"고 결론지었다
([FLUID 벤치마크](docs/FINDINGS.md#fluid-fi-벤치마크--평가-파이프라인-시험-2026-10-11),
[F절](docs/FINDINGS.md#f-논문-마무리-재료--fluid-운전자-경계--교차-데이터셋-규칙-a--그림-4개-2026-10-13)).
V2X-Seq 선두 1,867건에서는 이 결론이 뒤집혔다. 가속도 효과는 −0.009(n.s., FLUID −0.121)였고
([V2X-Seq 후속 A](docs/FINDINGS.md#a-가속도-효과가-fluid0121와-v2x-seq0009에서-다른-원인)), 경계 2.67 s가 황색 4 s보다
1.3 s 짧아 경계 보정만으로 규칙 A가 0.673 → 0.831이 됐다([D절](docs/FINDINGS.md#d-보정-규칙-a의-공정-평가--23폴드-loio-2026-10-13)).
이후 L·N·N2·P·P-fix는 이 경계를 기준으로 딜레마존 리드타임, 두 단계 판단, 데이터셋 간 전이를 확인했다.

## 6. 재현 방법

### 환경

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
# 병렬 작업 시 스레드 과다 방지 (스크립트들은 spawn 풀을 쓴다)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

### 테스트

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`은 시스템에 설치된 외부 pytest 플러그인을 자동 로드하지 않게 한다. 이 개발 환경에서는 그런 플러그인이 `lark` 모듈을 찾지 못해 테스트 수집 전에 실패했다.

### 실행 순서

중간 이벤트 테이블과 이벤트별 예측은 `data/raw/` 아래(gitignore)에, 표·그림은 `results/` 아래에 쓴다. 각 스크립트 맨 위
docstring에 인자 설명이 있다. 소요 시간은 실행 기록이 남은 것만 적었다(16스레드 CPU, GPU 없음).

```bash
# 1) FLUID FI 선두 (0 s 이벤트 표, 리드타임 0-2 s 이벤트 표)
FI="--dataset fluid --root data/raw/fluid/FI/derived_data --site FI \
    --approaches W_approach,E_approach --light-map W_approach=light_E_s,E_approach=light_E_s"
.venv/bin/python scripts/run_benchmark.py $FI --split leader --schemes group \
    --out results/fluid_fi/leader_follower/leader --event-dir data/raw/fluid/results/benchmark_fi_leader
.venv/bin/python scripts/run_benchmark.py $FI --split leader --schemes group --leads 0,0.5,1,1.5,2 --offsets 0 \
    --out <임시 폴더> --event-dir data/raw/fluid/results/leadtime_fi_leader

# 2) V2X-Seq 선두 (정지선 보정표 results/v2xseq/stopline_shift.csv는 커밋되어 있음;
#    새로 만들려면 v2xseq_check.py -> v2xseq_calibrate.py)
.venv/bin/python scripts/v2xseq_extract.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \
    --shift-table results/v2xseq/stopline_shift.csv --out data/raw/v2xseq/results/leaders_v4 --workers 3
.venv/bin/python scripts/v2xseq_extract.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \
    --shift-table results/v2xseq/stopline_shift.csv --out data/raw/v2xseq/results/leaders_v4_leads --workers 4 --leads 0,0.5,1,1.5,2
.venv/bin/python scripts/v2xseq_eval.py --event-dir data/raw/v2xseq/results/leaders_v4 --out results/v2xseq/eval/v4 --tasks external,loio
.venv/bin/python scripts/v2xseq_followup.py --tasks A,C
.venv/bin/python scripts/v2xseq_followup.py --tasks B,B2
.venv/bin/python scripts/v2xseq_followup.py --tasks D --threads 8        # D: 23폴드 보정 규칙 A
.venv/bin/python scripts/paper_materials.py                              # F: 경계 그림 등 4개

# 3) SinD v2.0 (선별, 창춘 선두 102건)
.venv/bin/python scripts/sind_full.py --tasks extract --workers 6
.venv/bin/python scripts/sind_full.py --tasks eval

# 4) WOMD v1.3.1 (샤드 1,150개를 받아 처리하고 지움; 리드타임 재처리는 약 2.8시간)
.venv/bin/python scripts/womd_w1_run.py --splits validation,training --out data/raw/womd/w1_leads --leads 0,0.5,1,1.5,2
.venv/bin/python scripts/womd_w1_aggregate.py --raw data/raw/womd/w1_leads --splits validation,training \
    --tables data/raw/womd/w1_leads/tables_all --out results/womd/w1_leads
.venv/bin/python scripts/womd_w1_variants.py --tables data/raw/womd/w1_leads/tables_all --out results/womd/w1_leads   # *_gkf, *_int

# 5) L: 딜레마존 리드타임
.venv/bin/python scripts/leadtime_dz.py --event-dir data/raw/v2xseq/results/leaders_v4_leads --out results/v2xseq/leadtime_dz
.venv/bin/python scripts/leadtime_dz.py --event-dir data/raw/womd/w1_leads/tables_all_gkf --out results/womd/leadtime_dz
.venv/bin/python scripts/leadtime_dz.py --event-dir data/raw/fluid/results/leadtime_fi_leader --out results/fluid_fi/leadtime_dz --min-site 1
.venv/bin/python scripts/leadtime_dz_figures.py --runs v2xseq=results/v2xseq/leadtime_dz womd=results/womd/leadtime_dz fluid=results/fluid_fi/leadtime_dz

# 6) N / N2: 두 단계 판단
.venv/bin/python scripts/staged_dz.py --event-dir data/raw/v2xseq/results/leaders_v4_leads --preds results/v2xseq/leadtime_dz --out results/v2xseq/staged
.venv/bin/python scripts/staged_dz.py --event-dir data/raw/womd/w1_leads/tables_all_gkf --preds results/womd/leadtime_dz --out results/womd/staged
.venv/bin/python scripts/staged_dz_figure.py
.venv/bin/python scripts/staged_n2.py --event-dir data/raw/v2xseq/results/leaders_v4_leads --preds results/v2xseq/leadtime_dz \
    --inner results/v2xseq/staged/inner_preds.csv.gz --out results/v2xseq/staged
.venv/bin/python scripts/staged_n2.py --event-dir data/raw/womd/w1_leads/tables_all_gkf --preds results/womd/leadtime_dz \
    --inner results/womd/staged/inner_preds.csv.gz --out results/womd/staged

# 7) P: 네 데이터 통합 (models 단계 15,615 s, 약 4.3시간, --workers 7 --threads 2)
.venv/bin/python scripts/pooled_eval.py --tasks sind-leads,table,boundary,models,report --workers 7 --threads 2
.venv/bin/python scripts/staged_n2.py --event-dir data/raw/pooled/table --preds data/raw/pooled/table --preds-kind pooled \
    --inner-folds 10 --out results/pooled/staged

# 8) P-fix (WOMD 제외 합동 LOIO의 models 단계 3,544 s)
.venv/bin/python scripts/pooled_eval.py --tasks no-womd                     # -> data/raw/pooled/table_no_womd
.venv/bin/python scripts/pooled_eval.py --tasks models --schemes loio --table data/raw/pooled/table_no_womd \
    --out results/pooled/fix --workers 7 --threads 2
.venv/bin/python scripts/leadtime_dz.py --event-dir data/raw/pooled/sind_leads --out results/pooled/fix/sind_self_leadtime_dz --min-site 1
.venv/bin/python scripts/pooled_fix.py
.venv/bin/python scripts/staged_n2.py --event-dir data/raw/pooled/table_no_womd --preds data/raw/pooled/table_no_womd \
    --preds-kind pooled --inner-folds 10 --out results/pooled/fix/staged_no_womd_fix

# 9) Q: v1.1 보완 분석 (라벨 경계 사례, 정지선 보정 민감도, θ 민감도, 확정 시점의 여유; 선두 추출 1회 약 5분)
.venv/bin/python scripts/v2xseq_calibrate.py --check data/raw/v2xseq/results/check_v3 --out results/v2xseq/stopline_shift_q05.csv --lead-q 0.05
.venv/bin/python scripts/v2xseq_calibrate.py --check data/raw/v2xseq/results/check_v3 --out results/v2xseq/stopline_shift_q20.csv --lead-q 0.20
V2X=data/raw/v2xseq/full/extracted/V2X-Seq-TFD
.venv/bin/python scripts/v2xseq_extract.py --root $V2X --out data/raw/v2xseq/results/leaders_v11_noshift --workers 4   # 보정 없음
.venv/bin/python scripts/v2xseq_extract.py --root $V2X --shift-table results/v2xseq/stopline_shift_q05.csv --out data/raw/v2xseq/results/leaders_v11_q05 --workers 4
.venv/bin/python scripts/v2xseq_extract.py --root $V2X --shift-table results/v2xseq/stopline_shift_q20.csv --out data/raw/v2xseq/results/leaders_v11_q20 --workers 4
.venv/bin/python scripts/v2xseq_v11.py --tasks traj,q1,q2,q3,q4
```

### 이벤트별 요약표 재생성

이벤트표는 원자료로부터 스크립트를 실행해 재생성한다(저장소에는 포함하지 않음).

```bash
# V2X-Seq: results/v2xseq/review/events_audit.csv, sample_audit.csv
.venv/bin/python scripts/v2xseq_check.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \
    --out data/raw/v2xseq/results/check_v4 --workers 4 --shift-table results/v2xseq/stopline_shift.csv
.venv/bin/python scripts/v2xseq_review.py --root data/raw/v2xseq/full/extracted/V2X-Seq-TFD \
    --check data/raw/v2xseq/results/check_v4 --out results/v2xseq/review --shift-table results/v2xseq/stopline_shift.csv \
    --table --audit-all --sample 20 --seed 0
# V2X-Seq: results/v2xseq/review/ambiguous_go.csv
.venv/bin/python scripts/v2xseq_ambiguous_go.py
# SinD 공개 샘플: results/sind/events.csv
.venv/bin/python scripts/run_sind.py
# WOMD W0: results/womd/w0/{events,onsets,yellow_durations,accel_quality,stop_point_check}.csv
.venv/bin/python scripts/womd_w0.py --shards data/raw/womd/sample/training.tfrecord-0000*-of-01000 --out results/womd/w0
```

## 7. 저장소 구성

```
src/dzp/        데이터 로더(sind, fluid, v2xseq, womd, waterloo, ngsim), 정지선 기하, 황색 이벤트·라벨, 특징,
                모델(규칙 A/B, LR, MLP, LSTM, Transformer), 평가(LOIO, 주기 부트스트랩)
src/womd_protos/ WOMD scenario/map proto 축소본 (TensorFlow 없이 파싱)
scripts/        실험 스크립트. 각 파일 맨 위 docstring에 실험 이름(D, F, L, N, N2, P, P-fix 등)과 실행 예시
results/        표(CSV/JSON)와 그림. 폴더 설명은 results/INDEX.md
docs/FINDINGS.md  모든 수치와 확인/추측 구분 (실험 절별)
docs/ROADMAP.md   진행 계획·상태
docs/notes/       초기 학습 노트
tests/          라벨링·규칙 단위 테스트
```

## 부록 A. NGSIM 입수 경로 (제외 데이터, 점검 재현용)

- **Peachtree**: 궤적 `https://data.transportation.gov/resource/8ect-6jqj.csv?$where=location='peachtree'`
  (→ `data/raw/ngsim/peachtree/peachtree_trajectories.csv`), 신호 변화 로그는 dataset 8ect-6jqj 첨부
  "Peachtree-Street-Atlanta-GA.zip"의 `peachtree-street-signal-timing/signal-timing/signal-change/Original Text files/*.txt`
  (→ `data/raw/ngsim/peachtree/signal_change/`).
- **Lankershim**: data.transportation.gov의 지원 자료 ZIP은 서버에서 잘려 있다. 옛 NGSIM 커뮤니티 사이트의 Wayback 사본
  `https://web.archive.org/web/20150324165048id_/http://ngsim-community.org/Lankershim-Main-Data.zip`
  (2015-03-24 캡처, 375,226,084 B, sha256 `d615aae87efe0b845e46254da34ebe07bfa1f4ebc93206730230a7a6fca7240c`)
  → `data/raw/ngsim/lankershim/`.
- 실행: `scripts/run_ngsim.py`(k = 0), `scripts/run_ngsim.py -15`(k = −1.5 s), `scripts/lankershim_check.py`, `scripts/lankershim_eval.py`.
