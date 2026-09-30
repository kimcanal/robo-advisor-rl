"""Shared constants for the RL portfolio module.

Values come from the mission spec (수수료/슬리피지/Safe-Guard 등).
Tune these here rather than hunting through the other modules.
"""

TRANSACTION_FEE = 0.00015   # 0.015%
SLIPPAGE = 0.0005           # 0.05%
WINDOW_SIZE = 30            # observation lookback window (거래일), 20~60 권장
MDD_LIMIT = 0.15            # Safe-Guard: 15% 초과 시 에피소드 조기 종료
MDD_LAMBDA_DEFAULT = 1.0    # 보상 변형3의 MDD 페널티 강도 (권장 탐색 0.5~5.0)
RISK_FREE_RATE = 0.0        # 샤프비율 계산용 (연율, 필요시 조정)
MAX_ASSET_WEIGHT = 0.4      # MVO 비교군 제약: 개별 자산 최대 비중
