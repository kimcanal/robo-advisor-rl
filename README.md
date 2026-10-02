# 라스트댄스 — 에이전틱 RAG 및 강화학습 기반 통합 자율 로보어드바이저

> "데이터로 분석하고, 스스로 판단하고, 근거로 설명하는 투자"

이 문서는 팀 전체의 아키텍처 방향성과 역할 간 인터페이스를 정리한 것입니다.
각자 담당 모듈의 세부 구현은 하위 폴더의 README를 참고하세요.

## 팀 구성

| 역할 | 담당자 | 산출물 |
|---|---|---|
| 데이터·퀀트 | 조윤상 | 가격 데이터 수집/전처리, 기본 투자 전략 |
| 강화학습 | 김윤하 | 투자 환경(Gymnasium), PPO 학습, 자산 비중 산출 |
| 설명·평가 | 손석범 | SHAP 해석, 성과 비교·ANOVA 통계 검증 |
| 리서치·RAG | 장지원(팀장) | 뉴스 검색·분석, 출처 검증, 위험 정보 추출(LangGraph) |
| 백엔드·화면 | 임소현 | FastAPI, Streamlit 대시보드, 배포 |

## 왜 이렇게 나눴는가

미션 자체가 "① 시계열 예측/최적화(정량)"과 "② 텍스트 리서치/추론(정성)"이라는
성격이 다른 두 축을 하나의 파이프라인으로 묶는 구조라, 이 두 축을 독립적으로
개발하고 **리스크 태그**라는 하나의 인터페이스로만 연결하는 것이 병렬 개발에
유리합니다. 아래 아키텍처가 그 경계를 보여줍니다.

## 전체 아키텍처

```
[조윤상: 데이터·퀀트]                    [장지원: 리서치·RAG]
   가격 데이터 수집/전처리                   뉴스 수집 -> ChromaDB
   log return, RSI/MACD 등 피처             LangGraph(계획-수행-검증)
        │                                   Self-Correction
        │  data/raw/{ticker}.csv                │
        │  (Close/Adj Close 컬럼)               │  위험 이벤트 탐지
        ▼                                        ▼
┌───────────────────────┐            ┌─────────────────────────┐
│  김윤하: 강화학습 모듈   │◄──리스크 태그──│  리서치 에이전트 출력      │
│  Gymnasium 포트폴리오   │  (관측공간에    │  (인용구 포함 리포트,      │
│  환경 + PPO + 보상 3종  │   반영/모니터링) │   현재 생각 로그)         │
│  Safe-Guard(MDD 15%)   │            └─────────────────────────┘
└───────────┬────────────┘
            │ 비중(weights), 학습곡선, 백테스트 결과
            ▼
┌───────────────────────┐
│ 손석범: 설명·평가 모듈   │
│ SHAP(Summary/Force)    │
│ ANOVA 3종 + Tukey HSD  │
└───────────┬────────────┘
            │ 해석 결과, 통계 검증 결과
            ▼
┌─────────────────────────────────────────────┐
│  임소현: 백엔드·화면                            │
│  FastAPI (/health /optimize /explain          │
│           /research /backtest /anova)         │
│  Streamlit 7탭 (API 통신만, 모델 직접 로드 금지)  │
└─────────────────────────────────────────────┘
```

## 역할 간 인터페이스 계약 (중요 — 이것만 지키면 병렬 개발 가능)

1. **데이터팀 → 강화학습팀**
   `rl/data/raw/{ticker}.csv` 에 `Date`, `Close`(또는 `Adj Close`) 컬럼을 가진 CSV만
   두면 강화학습 모듈은 코드 변경 없이 실제 데이터로 전환됩니다.
   (해당 티커 CSV가 없으면 `rl/data/dummy_data.py`가 자동으로 더미 데이터로 대체합니다.
   지금은 `rl/data/fetch_real_data.py`로 받은 실제 ETF 10종으로 검증 중이며,
   정식 파이프라인이 오면 그걸로 교체하면 됩니다. raw CSV는 라이선스상
   `.gitignore`로 제외되어 있어 리포지토리를 받아도 각자 다시 받아야 합니다.)

2. **리서치·RAG팀 → 강화학습팀**
   위험 이벤트를 `{ticker, risk_score(0~1), tag, ts}` 형태로 넘기면
   (`rl/risk_tags.py` 스키마, CSV도 동일), `PortfolioEnv` 관측 끝의
   `portfolio_risk`(보유 비중 가중 평균) 축에 반영된다. 태그가 없으면 0.0.
   선택적으로 `risk_mdd_scale`으로 Safe-Guard MDD 한도를 위험도에 비례해 조일 수 있다.

3. **강화학습팀 → 설명·평가팀**
   `rl/backtest.py`의 `compute_metrics()`가 반환하는 12개 지표 dict와,
   `rl/stats_tests.py`의 `one_way_anova()`/`two_way_anova()`를 그대로 재사용하거나
   참고해서 대시보드에 연결하면 됩니다. SHAP은 `rl/shap_explain.py`에 정책
   래퍼까지 만들어뒀습니다.

4. **모든 모듈 → 백엔드팀**
   FastAPI는 각 모듈의 함수를 직접 import해서 감싸는 방식(모놀리식 배포 +
   MSA 스타일 API 분리)을 권장합니다. 학습된 모델은 `rl/outputs/models/*.zip`에
   저장되므로, `/optimize`, `/explain` 엔드포인트는 이 파일을 로드해서 추론하면 됩니다.

## 마일스톤 (Charter 기준)

- 제안 발표: 2026-10-12
- 중간 보고: 2026-11-09 주
- 완료 보고: 2026-12-14

## 현재 상태 (2026-10-03 기준)

- ✅ 강화학습 모듈: 환경/보상 3종/PPO+VecNormalize 학습/백테스트 12지표/MVO 비교/
  SHAP/ANOVA(One-way×2, Two-way) 전부 구현, **실제 ETF 10종 데이터로 검증 완료**
  (`rl/README.md` 참고)
- ✅ 버그 2개 발견·수정: 거래비용(turnover) 2배 과다계산, 무위험이자율 하드코딩(0)
- ✅ 시장 국면(Bull/Flat/Bear) 탐지기 추가 — 참고 프로젝트에서 핵심 로직만 이식,
  역사적 사건(코로나/2022 약세장/2021 강세장)과 대조 검증 완료
- ✅ VecNormalize 적용 전후 비교: 동일 시드(5~9) 기준 simple·mdd_penalty 보상함수가
  수익률↑·MDD↓ 동시 개선을 재현성 있게 확인 (`rl/outputs/experiments/` 참고)
- ✅ Walk-Forward(`walk_forward.py`)에 VecNormalize 적용·증분 CSV 저장·BIL/피처래그/
  베이스라인 수수료 공정성 반영. 기존 발표 수치는 구버전일 수 있으므로 Colab
  `colab_run.ipynb`로 재실행한 결과를 최종으로 쓸 것
- ✅ Colab `colab_run.ipynb`: 매 실행 `master` 신규 클론 → data(BIL/`SPY_ohlcv`) →
  데모 또는 풀 WF(≥100k×3보상×≥2윈도우) → 12지표(EW/MVO/SPY/KOSPI) → ANOVA →
  SHAP → risk-tag 스텁 → 결과 다운로드. GPU 불필요. Notion week-38 체크리스트 매핑 포함.
- ✅ 리스크 태그 스텁(`rl/risk_tags.py`) + SPY/KOSPI 벤치마크가 WF 비교표에 포함 (PR#2)
- ⏳ 동일가중 포트폴리오를 아직 절대수치로는 못 이김 — 원인 추적 중 (관측값에
  drawdown 미포함 등), 과제 스펙의 Why/How 문서화 요구에 맞춰 계속 기록 중
- ⏳ 데이터팀 실제 수집 파이프라인 대기 중 (연결되면 `rl/data/raw/`에 CSV만 추가)
- ✅ 리서치·RAG LangGraph형 스텁 확장 (`rag/`: plan→retrieve→tag_risk→verify→summarize, in-memory store, citations, API `/research` 연동) — 실 LLM/Chroma는 ⏳
- ✅ FastAPI/Streamlit/Docker 스켈레톤 확장 (`api/`, `streamlit_app/`, `rag/`, `Dockerfile`, CI workflow, `docs/report/outline.md`)
- ✅ 리스크 태그 → `PortfolioEnv.portfolio_risk` 배선 데모 API (`POST /risk-tags/apply`) — 교육용 스텁

## Notion week-38 submission checklist

팀 Notion 제출/중간 점검용 체크리스트. 세부 구현은 담당 모듈 README를 따른다.

| 섹션 | 상태 | 어디에 있나 |
|---|---|---|
| **Architecture** | ✅ 문서화 | 위 아키텍처 다이어그램 + 역할 인터페이스 계약 |
| **Reward rationale** | ✅ 문서화 | `rl/README.md` 설계 근거 (simple / sharpe / mdd_penalty + 출처 표) |
| **Docker placeholder** | ✅ 스켈레톤 | `Dockerfile`, `docker-compose.yml` (api + streamlit) |
| **Metrics** | ✅ 코드 | `rl/backtest.py::compute_metrics` (12지표), API `GET /backtest` |
| **ANOVA** | ✅ 코드+API | `rl/stats_tests.py` + API `POST /anova` (합성 시리즈 교육용) |
| **RAG / LangGraph stub** | ✅ 스텁 확장 | `rag/graph.py` (plan→retrieve→tag_risk→verify→summarize) + `rag/store.py` (in-memory Chroma-lite) + citations — 실 LLM/Chroma ⏳ |
| **Risk-tag wiring** | ✅ 스텁 API | `POST /risk-tags/apply` → `rl.risk_tags.risk_score_panel` + short `PortfolioEnv` obs (`portfolio_risk`) |
| **Performance targets** | ✅ 체크리스트 | `docs/performance_targets.md` (Colab 수치 placeholder) |
| **CI (GitHub Actions)** | ✅ workflow | `.github/workflows/ci.yml` (pytest on push/PR) |
| **Error analysis** | ⏳ 진행 | Walk-Forward·동일가중 미달 원인 추적 (`rl/README` 한계 6–7) |
| **Financial disclaimer** | ✅ | 아래 고지 + API/Streamlit 캡션 |

### API / UI quick start

```bash
source .venv/bin/activate
pip install -r rl/requirements.txt -r requirements-api.txt

# API (Swagger: http://127.0.0.1:8000/docs)
uvicorn api.main:app --reload --port 8000

# Streamlit (API만 호출, 모델 직접 로드 금지)
API_BASE_URL=http://127.0.0.1:8000 streamlit run streamlit_app/app.py

# Docker
docker compose up --build
```

Endpoints: `GET /health`, `POST /optimize`, `POST /explain` (artifact JSON or clear stub), `POST /research` (RAG plan/verify + citations), `POST /risk-tags/apply` (panel + env obs wiring), `GET /backtest` (synth SPY/KOSPI + latency_ms), `POST /anova` (합성 ANOVA).

## Colab 재검증 (Notion week-38)

[`colab_run.ipynb`](colab_run.ipynb) — Open in Colab 배지로 실행.

1. Setup: `rm -rf` 후 `master` 클론 + `rl/requirements.txt`
2. Data: `fetch_real_data` + **BIL / SPY_ohlcv / risk-free 상태 큰 출력**
3. Train: 짧은 `run_demo` **또는** 풀 `walk_forward --real --timesteps 120000` (≥100k, 보상 3종, ≥2 윈도우; 증분 CSV)
4. Metrics / ANOVA / SHAP / risk-tag 스텁 / 결과 다운로드 (`Path.exists` 가드)

**Disclaimer:** 교육·연구 목적. 백테스트 ≠ 미래 수익. 투자 자문 아님.

## Financial disclaimer (필수 고지)

**본 저장소는 교육·연구 목적의 데모입니다. 투자 자문이 아니며, 특정 증권의 매수·매도를 권유하지 않습니다.**

- 백테스트·시뮬레이션 성과는 **과거 데이터(또는 합성 데이터)에 기반**하며 **미래 수익을 보장하지 않습니다** (backtests ≠ future returns).
- 거래비용·슬리피지·유동성·세금·survivorship 등 실전 제약이 단순화되어 있을 수 있습니다.
- API 키(`.env.example`)와 Docker 스택은 로컬 실험용이며, 라이브 브로커 연동은 기본 비활성입니다.
- 실제 투자 결정은 본인 책임이며, 필요 시 자격 있는 전문가와 상담하세요.

