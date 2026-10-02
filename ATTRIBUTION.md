# 강화학습 모듈 — 참고자료 출처 및 코드 감사 결과

## 개요

강화학습(RL) 모듈 개발 중, 동료(영환/luca)가 공유해준 참고 프로젝트
(`Dynamic Regime Portfolio`)를 **일부 로직 이식 + 코드 자가검증 도구**로
활용했습니다. 핵심 RL 파이프라인(환경·보상·PPO 학습)은 전부 자체 구현이며,
아래 두 가지만 외부 참고를 거쳤습니다.

## 참고한 것 / 안 한 것

| 구분 | 내용 |
|---|---|
| ✅ 가져온 것 | 시장 국면(Bull/Flat/Bear) 규칙기반 판정 로직 — `models/score_model.py`를 간소화해서 이식 (학습 불필요, 이동평균·VWAP·ROC 기반) |
| ❌ 안 가져온 것 | GNN, HMM, entropy pooling 등 — 우리 과제 범위와 무관한 별도 시스템이라 제외 |

**출처 표기 문구** (리포트/Notion에 그대로 사용):

> 시장 국면 탐지 로직은 동료(영환, Dynamic Regime Portfolio 프로젝트)의
> 규칙기반 Score 모델을 간소화해서 이식했습니다 (출처 명시, 핵심 RL
> 파이프라인은 자체 구현).

실제 구현: [`rl/regime.py`](rl/regime.py)

## 코드 감사로 발견·수정한 버그 3개

| # | 버그 | 발견 경위 | 수정 내용 |
|---|---|---|---|
| 1 | 거래비용(turnover) 2배 과다계산 | 참고 프로젝트 `cost_model.py`와 비교 중 발견 | `turnover`를 매도+매수 합이 아니라 `/2`로 보정 ([`rl/env/portfolio_env.py`](rl/env/portfolio_env.py)) |
| 2 | 무위험이자율 0 하드코딩 | 참고 프로젝트 CONSTITUTION 원칙과 대조 | 실제 BIL 단기국채 수익률(연 2.63%)로 교체 ([`rl/riskfree.py`](rl/riskfree.py)) |
| 3 | MACD Look-ahead Bias | 참고 프로젝트 자체 코드감사 문서(`Code Weakness.md`)의 "전체기간 스케일러 fit" 지적과 동일 패턴 발견 | train/test 분할 전 정규화 로직 제거, VecNormalize로 대체 ([`rl/pipeline.py`](rl/pipeline.py)) |

## 검증만 하고 수정 안 한 것 (안심 포인트)

- **비용 이중차감 버그**: 참고 프로젝트엔 있었지만 우리 코드엔 없음 — 확인만 하고 종료
- **샤프비율/Information Ratio 공식**: 독립적으로 유도한 두 공식이 수학적으로 동일함을 검증

## 왜 이게 "베낀 것"이 아니라 "제대로 된 엔지니어링"인가

1. 가져온 로직은 금융공학 표준 기법(이동평균·VWAP)이지 독창적 알고리즘이 아님
2. 버그 3개는 전부 **우리 자신의 코드**에서 발견·수정한 것 — 참고자료는 "점검 체크리스트" 역할만 함
3. 전 과정이 이 리포지토리(`kimcanal/robo-advisor-rl`)의 커밋 이력에 투명하게 기록됨 — 자세한 내용은 `rl/README.md`의 "코드 감사 이력" 표 참고
