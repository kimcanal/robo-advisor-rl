# RAG research agent stub (LangGraph-shaped)

교육용 **상태머신 스텁**입니다. 노드 흐름은 LangGraph와 같은 형태입니다:

`retrieve` → `tag_risk` → `summarize`

- LLM API 키 불필요
- 출력 리스크 태그는 `rl.risk_tags` 스키마와 호환 (`ticker`, `risk_score`, `tag`, `ts`)
- FastAPI `POST /research`가 이 스텁을 호출하며 응답에 `stub: true`를 유지합니다

실제 LangGraph + 벡터스토어로 교체할 때는 `rag.graph.run_research_graph`만
같은 계약으로 바꾸면 API/Streamlit은 그대로 둘 수 있습니다.
(현재 `requirements-api.txt`에 langgraph를 넣지 않음 — 의존성 부담을 피하기 위함.)
