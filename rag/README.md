# RAG research agent stub (LangGraph-shaped · Notion week-38)

교육용 **상태머신 스텁**입니다. Notion 제출용 plan / exec / verify 흐름을 반영합니다:

`plan` → `retrieve` → `tag_risk` → `verify` → `summarize`

## 구성

| 모듈 | 역할 |
|---|---|
| `rag/graph.py` | 노드/엣지 상태머신 (LangGraph 개념 대응) |
| `rag/store.py` | **Chroma-lite** 인메모리 벡터스토어 (BoW + cosine, chromadb 불필요) |

## 계약 (RAG → RL env)

위험 이벤트는 `rl.risk_tags` 스키마와 동일해야 합니다:

```
{ticker: str, risk_score: float in [0,1], tag: str, ts: datetime-like}
```

이 태그는 `PortfolioEnv` 관측 끝의 **`portfolio_risk`**(보유 비중 가중 평균) 축으로 들어갑니다.
태그가 없으면 0.0 — 기존 RL 호출부는 깨지지 않습니다.

## Citations

`retrieve` 노드는 인메모리 코퍼스에서 히트를 고르고 `Citation` placeholder
(`doc_id`, `title`, `source`, `snippet`/`quote`, `score`)를 붙입니다.
실 서비스에서는 URL/뉴스 ID로 교체하면 됩니다.

## Verify (Self-Correction placeholder)

`verify` 노드는 스키마·태그 허용 집합·citation 존재 여부를 검사하고
`verify_ok` / `verify_notes`를 남깁니다 (실 LLM Self-Correction 자리표시자).

## API

FastAPI `POST /research`가 이 스텁을 호출합니다. 응답 필드:

- `risk_tags`, `report_excerpt`, `stub: true`
- `plan`, `node_trace`, `citations`, `verify_ok`, `verify_notes`, `env_contract`
- `latency_ms`

## 환경 변수 (optional)

| 변수 | 용도 |
|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | 실 LLM 연동 시 (현재 스텁 미사용) |
| `RAG_COLLECTION` | 향후 Chroma collection 이름 placeholder |
| `EXPLAIN_ARTIFACT_PATH` | `/explain`용 사전계산 SHAP JSON |
| `EXPLAIN_MODEL_PATH` | SB3 zip 경로 (API는 무거워서 기본 미실행) |

실제 LangGraph + ChromaDB로 교체할 때는 `rag.graph.run_research_graph`만
같은 계약으로 바꾸면 API/Streamlit은 그대로 둘 수 있습니다.
(`requirements-api.txt`에 langgraph/chromadb를 넣지 않음 — 의존성 부담 회피.)
