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
| `features.py` | 로그수익률, Z-score 정규화, RSI, MACD |
| `pipeline.py` | 위 조각들을 합쳐 env 입력(수익률/RSI/MACD, 인덱스 정렬)으로 조립 |
| `env/portfolio_env.py` | Gymnasium 커스텀 환경 (관측/행동 공간, Safe-Guard) |
| `rewards.py` | 보상 함수 3종 (simple / sharpe / mdd_penalty) |
| `train.py` | PPO 단일 학습 스크립트 (모델 + 학습곡선 저장) |
| `mvo.py` | MVO(Markowitz) 비교 기준, scipy 최적화 |
| `backtest.py` | 성과 지표 12종, Walk-Forward 윈도우 분할 |
| `stats_tests.py` | ANOVA(One-way/Two-way) + Tukey HSD — **설명·평가팀 인터페이스** |
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

## 알려진 한계 / TODO (다음 단계)

1. 포트폴리오 수익률을 "자산별 로그수익률의 가중합"으로 근사 중 — 엄밀한 정의는
   아니지만 FinRL 등에서 흔히 쓰는 단순화. 리포트에 명시 필요.
2. Walk-Forward 백테스트는 함수(`walk_forward_windows`)만 있고, "윈도우 이동 시
   모델 재학습"까지 자동화된 스크립트는 아직 없음 — 실 데이터로 본 학습 돌릴 때 추가.
3. SHAP은 `KernelExplainer` 기반이라 자산 수/윈도우가 커지면 느려짐. 10자산+윈도우
   30이면 관측 차원이 400에 육박 — 배경 표본을 줄이거나 `nsamples`를 낮출 것.
4. 실데이터 연결: `data/raw/{ticker}.csv`에 `Date`, `Close`/`Adj Close` 컬럼의 CSV를
   넣기만 하면 됨 (조윤상 파이프라인 완료 시).
5. 리스크 태그(RAG팀) 연동: 아직 관측 공간에 반영 안 함. 인터페이스 오면
   `env/portfolio_env.py`의 관측 공간에 축 하나 추가.
