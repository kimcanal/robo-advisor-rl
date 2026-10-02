# 강화학습 모듈 (김윤하 담당)

투자 환경 설계, PPO 학습, 자산 비중 산출을 담당하는 모듈. 전체 팀 아키텍처는
[프로젝트 루트 README](../README.md)를 참고.

## 현재 상태

- **더미 데이터**로 end-to-end 파이프라인 검증 완료 (데이터팀 CSV가 아직 없음)
- 환경/보상 3종/PPO 학습/백테스트 12지표/MVO 비교/SHAP/ANOVA **전부 코드로 존재하고 실행됨**
- ⚠️ 학습 스텝 수는 데모용으로 작게(3천~2만) 돌린 상태라 **성능 숫자 자체는 의미 없음**
  (요구사항의 10만 스텝 이상, 10자산 이상은 아직 안 돌림 — 시간이 오래 걸려서
  본 학습은 별도로 백그라운드에서 돌려야 함)

## 빠른 실행

```bash
cd robo-advisor
source .venv/bin/activate   # 없으면: python3 -m venv .venv && pip install -r rl/requirements.txt

# 1) 개별 보상함수 하나만 학습 (모델 + 학습곡선 저장)
python -m rl.train --reward mdd_penalty --timesteps 100000

# 2) 전체 파이프라인 데모 (학습 3종 + 백테스트 + ANOVA + SHAP까지 한 번에)
python -m rl.run_demo --n-assets 10 --timesteps 20000 --test-days 252

# 3) 유닛 테스트
python -m pytest rl/tests -v
```

`run_demo.py`는 회의/발표에서 "파이프라인이 실제로 끝까지 도는지" 보여주는
용도이며, `--timesteps`를 늘릴수록 결과가 그럴듯해지지만 시간도 늘어난다
(CPU 기준 20자산 x 10만 스텝은 꽤 오래 걸릴 수 있음 — 여유 있을 때 백그라운드로).

## 모듈 맵

| 파일 | 역할 |
|---|---|
| `config.py` | 수수료/슬리피지/윈도우/Safe-Guard 등 상수 |
| `data/dummy_data.py` | 상관관계 있는 더미 가격 생성 (GBM 기반) |
| `data/loader.py` | **데이터팀 인터페이스 계약** — `data/raw/{ticker}.csv` 우선, 없으면 더미로 자동 대체 |
| `data/fetch_real_data.py` | yfinance로 실제 ETF 10종(SPY/QQQ/IWM/EFA/EEM/AGG/TLT/HYG/GLD/VNQ) 5년+ 데이터를 받아 `data/raw/`에 저장 — 조윤상님 정식 파이프라인 전 **임시 실데이터 검증용** |
| `features.py` | 로그수익률, Z-score 정규화, RSI, MACD |
| `pipeline.py` | 위 조각들을 합쳐 env 입력(수익률/RSI/MACD, 인덱스 정렬)으로 조립 |
| `env/portfolio_env.py` | Gymnasium 커스텀 환경 (관측/행동 공간, Safe-Guard) |
| `rewards.py` | 보상 함수 3종 (simple / sharpe / mdd_penalty) |
| `train.py` | PPO 단일 학습 스크립트 (모델 + 학습곡선 저장) |
| `mvo.py` | MVO(Markowitz) 비교 기준, scipy 최적화 |
| `backtest.py` | 성과 지표 12종, Walk-Forward 윈도우 분할 |
| `stats_tests.py` | ANOVA(One-way/Two-way) + Tukey HSD — **설명·평가팀 인터페이스** |
| `regime.py` | 시장 국면(Bull/Flat/Bear) 규칙기반 탐지기(학습 불필요) — 검증3 Two-way ANOVA 국면 라벨용. `~/Dynamic_Regime_Portfolio-luca` 프로젝트의 `ScoreRegimeDetector`를 참고/이식(breadth 조건 제외, 핵심 4개 기술적 조건만) |
| `walk_forward.py` | Walk-Forward 백테스트(학습4년→테스트1년, 2윈도우, 재학습) + 일별 국면 라벨로 검증2·검증3 ANOVA까지 실행 |
| `shap_explain.py` | 정책 의사결정에 대한 SHAP Summary/Force Plot |
| `run_demo.py` | 위 전부를 잇는 end-to-end 데모 |
| `tests/` | pytest 14개 (env/reward/backtest/mvo/anova 스모크 테스트) |

## 설계 근거 (회의에서 설명할 포인트)

- **행동 공간**: 연속값 벡터에 softmax를 씌워 비중으로 변환 (합=1, 공매도 불가).
  이산적 리밸런싱 대신 연속을 택한 이유는 자산이 10개 이상일 때 이산 행동 공간이
  조합 폭발하기 때문.
- **관측 공간**: 과거 `window`일 로그수익률 + 현재 비중 + RSI + MACD 히스토그램.
  window는 기본 30일(요구 범위 20~60 내), 시퀀스를 그대로 펼쳐 MLP에 입력
  (LSTM/Transformer는 스코프 밖으로 미룸).
- **보상 3종**: `simple`(대조군) / `sharpe`(변동성 페널티) / `mdd_penalty`(누적낙폭
  페널티, lambda로 공격성 조절). 세 개를 실제로 비교해야 "왜 이 보상을 골랐는지"를
  ANOVA로 뒷받침할 수 있음.
- **Safe-Guard**: 낙폭이 `mdd_limit`(기본 15%)을 넘으면 즉시 에피소드 종료.
  `tests/test_env.py::test_safe_guard_triggers_on_large_drawdown`으로 검증됨.
- **거래비용**: 매 스텝 `turnover`(비중 변화량 합) x (수수료+슬리피지)를 net_return에서 차감.

## 설계 근거 및 출처 (리포트 Why 섹션용)

리포트가 요구하는 "왜 이 설정을 선택했는가"에 그대로 쓸 수 있도록 각 결정의
근거와 출처를 정리한다. 링크 대신 저자/연도/제목으로 표기 — 검색해서
원문을 직접 확인하고 인용할 것.

| 결정 | 근거 | 출처 |
|---|---|---|
| PPO 알고리즘 사용 | on-policy로 안정적이고, 연속 행동공간(비중 조절)에 표준적으로 쓰임. clipped objective로 학습 불안정성(발산) 억제 | Schulman et al., 2017, "Proximal Policy Optimization Algorithms" (arXiv:1707.06347) |
| 자산배분 RL에 PPO/Gym 커스텀 환경 적용 자체의 선례 | 금융 RL 분야에서 Gym 기반 커스텀 트레이딩 환경 + PPO/A2C 조합이 표준 베이스라인으로 자리잡음 | Liu et al., 2020, "FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading" (arXiv:2011.09607) |
| 행동공간을 softmax 연속값으로 (이산 대신) | 자산 10개 이상에서 이산 리밸런싱은 조합이 기하급수적으로 늘어남; softmax는 합=1·공매도금지 제약을 별도 프로젝션 없이 만족 | 위 FinRL 논문의 포트폴리오 배분 환경 설계와 동일한 관례 |
| 보상함수에 샤프비율 반영 (변형2) | 변동성 대비 초과수익을 극대화한다는 표준 위험조정수익 개념 | Sharpe, W.F., 1966, "Mutual Fund Performance," Journal of Business |
| 보상함수에 MDD 페널티 반영 (변형3) | 단순 변동성(샤프)과 달리 "연속 손실의 깊이"에 직접 페널티를 주어 꼬리위험을 억제 | Young, T.W., 1991 (Calmar Ratio 개념 — MDD 대비 수익 평가); 실무적으로는 손실 한도(스탑로스) 설계와 동일 발상 |
| 소르티노 비율(하방 변동성만 사용) | 샤프비율은 상방 변동성도 페널티로 잡는 한계가 있어, 투자자가 실제로 꺼리는 하방 변동성만 분리 | Sortino & van der Meer, 1991, "Downside Risk," Journal of Portfolio Management |
| VaR/CVaR 지표 채택 | 정규분포 가정 없이 실제 손실 분포의 꼬리 위험을 정량화; CVaR은 VaR을 넘는 손실의 기댓값이라 더 보수적 | Rockafellar & Uryasev, 2000, "Optimization of Conditional Value-at-Risk," Journal of Risk |
| MVO(Markowitz) 비교 기준 | 정적 최적화의 이론적 표준. DRL(동적)과의 성과 차이를 "정적 vs 동적"이라는 명확한 축으로 비교 가능 | Markowitz, H., 1952, "Portfolio Selection," Journal of Finance |
| MVO 공분산 252일 롤링 윈도우 | 실무에서 1년(약 252거래일)을 변동성/공분산 추정의 표준 관측 기간으로 사용(계절성 대비 충분히 길고, 구조변화 대비 너무 길지 않음) | 업계 표준 관행 (예: RiskMetrics 방법론류) |
| SHAP 기반 해석 | 게임이론 Shapley value를 모델-불가지론적으로 근사해, 각 피처가 예측에 기여한 정도를 가법적으로 분해 — 금융 XAI 규제 대응에 적합 | Lundberg & Lee, 2017, "A Unified Approach to Interpreting Model Predictions," NeurIPS (arXiv:1705.07874) |
| ANOVA + Tukey HSD | 3개 이상 그룹 평균 차이의 통계적 유의성을 한 번에 검정(개별 t-검정 반복 시 발생하는 다중비교 오류 방지); 유의하면 Tukey HSD로 어느 쌍이 다른지 사후 검정 | Fisher, R.A., 1925, "Statistical Methods for Research Workers" (ANOVA); Tukey, J.W., 1949, "Comparing Individual Means in the Analysis of Variance" (사후검정) |
| 관측 윈도우 N=20~60 | 미션 스펙 자체가 명시한 범위 — N이 작으면 단기 모멘텀에 민감(차원 낮아 학습 빠름), N이 크면 추세 반영(차원 커져 학습 느림). 실제 트레이드오프는 `experiments.py::window_sweep`으로 검증 | 과제 스펙 4-2 설계 가이드 + `rl/outputs/experiments/window_sweep.csv` 실험 결과 |
| RSI(14일)/MACD(12,26,9) 파라미터 | 기술적 분석에서 가장 널리 쓰이는 표준 파라미터 (교재/실무 관행) | Wilder, J.W., 1978, "New Concepts in Technical Trading Systems" (RSI); Appel, G. (MACD 창안자) |
| 거래수수료 0.015%/슬리피지 0.05% | 임의 선택이 아니라 **과제 스펙 4-2에 고정값으로 명시**된 제약 | 과제 스펙 (자체 근거 불필요, "요구사항 준수"가 근거) |
| lambda(MDD 페널티 강도) 탐색범위 0.5~5.0 | 과제 스펙이 권장 범위로 명시; 실제 단조적 트레이드오프(lambda↑ → 수익률↓, MDD↓)가 나오는지는 `experiments.py::lambda_sweep`으로 검증 | 과제 스펙 4-3 + `rl/outputs/experiments/lambda_sweep.csv`, `lambda_tradeoff.png` |
| 시장 국면(Bull/Flat/Bear) 판정 방식 | 이동평균(5/20/60일)·VWAP·변동성보정 ROC 4개 기술적 조건의 다수결 — 학습 없이 재현 가능하고 look-ahead 없음(전일까지 데이터만 사용) | 자체 구현이 아니라 참고 프로젝트에서 이식: `Dynamic_Regime_Portfolio-luca/models/score_model.py::ScoreRegimeDetector` (원본은 S&P500 전종목 breadth 조건 포함, 우리는 개별종목 유니버스가 없어 핵심 4개 조건만 이식) |
| 무위험이자율을 0 대신 실제 단기국채(BIL)로 | 위 참고 프로젝트 CONSTITUTION도 동일 원칙("하드코딩 상수 대신 BIL/SOFR 동적 사용")을 명시 — 두 출처가 같은 결론이라 근거가 탄탄함 | `Dynamic_Regime_Portfolio-luca/docs/CONSTITUTION.md` §5.4 + `rl/riskfree.py` |

### 국면 탐지기(`regime.py`) 검증

실제 역사적 사건과 대조해서 탐지기가 말이 되는 라벨을 내는지 확인함 (2026-09-30):

| 기간 | Bull/Flat/Bear 비율 | 실제 사실과 일치? |
|---|---|---|
| COVID 폭락 (2020-02~04) | bear 56% / bull 38% / flat 6% | ✅ 폭락+4월 반등 초입이 섞여 타당 |
| 2022 금리인상 약세장 | bear 54% / bull 26% / flat 20% | ✅ |
| 2021 강세장 | bull 77% / flat 18% / bear 5% | ✅ |
| 2023 회복장 | bull 53% / flat 24% / bear 22% | ✅ (상반기 혼조 반영) |

재현: `python3 -c "from rl.regime import load_spy_regime; ..."` (기간별 `.value_counts()` 확인)

## 남은 실험 (리포트 "실험" 섹션 근거 자료 생성용)

과제 스펙이 명시적으로 요구하는 두 실험을 `rl/experiments.py`에 자동화해뒀다.

```bash
# 보상 변형3의 lambda 트레이드오프 곡선 (스펙 4-3 요구사항)
python -m rl.experiments --which lambda --timesteps 15000
# -> rl/outputs/experiments/lambda_sweep.csv, lambda_tradeoff.png

# 관측 윈도우 N=20 vs 60 비교 (스펙 4-2 설계 가이드 요구사항)
python -m rl.experiments --which window --timesteps 15000
```

⚠️ 여기 쓴 timesteps(1.5만)도 여전히 데모 수준이다. **리포트에 실제로 넣을
숫자**는 본 학습(10만 스텝 이상, 10자산 이상, 여유 시간에 백그라운드로)을
돌린 뒤의 결과로 교체해야 한다 — 지금 이 실험은 "코드가 트레이드오프를
올바르게 재현하는지"를 확인하는 용도.

## 알려진 한계 / TODO (다음 단계)

1. 포트폴리오 수익률을 "자산별 로그수익률의 가중합"으로 근사 중 — 엄밀한 정의는
   아니지만 FinRL 등에서 흔히 쓰는 단순화. 리포트에 명시 필요.
2. Walk-Forward 자체는 `walk_forward.py`로 자동화 완료(윈도우 이동마다 실제 재학습).
   다만 이 스크립트는 **VecNormalize 도입 이전** 버전으로 돌린 결과가 마지막이라,
   VecNormalize 적용 후 재실행하면 수치가 바뀔 수 있음 — 다음 실행 후보.
3. SHAP은 `KernelExplainer` 기반이라 자산 수/윈도우가 커지면 느려짐. 10자산+윈도우
   30이면 관측 차원이 400에 육박 — 배경 표본을 줄이거나 `nsamples`를 낮출 것.
4. 실데이터 연결: `data/raw/{ticker}.csv`에 `Date`, `Close`/`Adj Close` 컬럼의 CSV를
   넣기만 하면 됨 (조윤상 파이프라인 완료 시). 지금은 `fetch_real_data.py`로 받은
   yfinance 데이터가 임시로 들어가 있음 — **yfinance는 비상업적 목적만 허용**이라
   `data/raw/*.csv`는 `.gitignore`로 제외해 리포지토리에는 올리지 않음(라이선스 준수).
   정식 데이터는 조윤상님 파이프라인 결과로 교체할 것.
5. 리스크 태그(RAG팀) 연동: 아직 관측 공간에 반영 안 함. 인터페이스 오면
   `env/portfolio_env.py`의 관측 공간에 축 하나 추가.
6. 관측 공간에 "현재 낙폭(drawdown)"이 빠져있음. mdd_penalty 보상이 낙폭에
   페널티를 주는데 정작 에이전트는 자기 낙폭 상태를 직접 볼 수 없어서, Safe-Guard를
   피하라고 학습시키기 어려운 구조적 한계로 보임 — 다음 개선 후보.
