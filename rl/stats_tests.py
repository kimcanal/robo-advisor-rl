"""ANOVA 통계 검증 3종 (요구사항 4-8) — 손석범(설명·평가) 담당 영역과 맞닿는 지점.

이 모듈은 "일별 수익률 시리즈들의 딕셔너리"를 입력으로 받아 바로 검정
결과를 낸다. RL 쪽에서 백테스트 결과(returns dict)만 넘겨주면 되므로,
설명·평가 담당자는 이 함수들을 그대로 가져다 쓰거나 참고해서 대시보드에
연결하면 된다 (인터페이스 계약).

p < 0.05 이면 Tukey HSD를 자동으로 수행한다. 결과는 statsmodels 요약 객체
(`tukey_hsd`)와 JSON으로 바로 쓸 수 있는 레코드 리스트(`tukey_records`)를
둘 다 돌려준다.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ALPHA = 0.05


def tukey_records(values, labels, alpha: float = ALPHA) -> tuple[object, list[dict]]:
    """Tukey HSD를 돌려 (statsmodels summary, JSON 레코드 리스트)를 반환."""
    tukey = pairwise_tukeyhsd(values, labels, alpha=alpha)
    table = tukey.summary()
    header = [str(h) for h in table.data[0]]
    records = []
    for row in table.data[1:]:
        rec = dict(zip(header, row))
        records.append(
            {
                "group1": str(rec["group1"]),
                "group2": str(rec["group2"]),
                "meandiff": float(rec["meandiff"]),
                "p_adj": float(rec["p-adj"]),
                "lower": float(rec["lower"]),
                "upper": float(rec["upper"]),
                "reject": bool(rec["reject"]),
            }
        )
    return table, records


def one_way_anova(groups: dict[str, pd.Series], alpha: float = ALPHA) -> dict:
    """검증 1(보상함수 3종) / 검증 2(DRL vs MVO vs 동일가중)에 공용으로 사용."""
    clean = {k: pd.Series(v).dropna() for k, v in groups.items()}
    series_list = [s.to_numpy() for s in clean.values()]
    f_stat, p_value = stats.f_oneway(*series_list)

    grand_mean = pd.concat(clean.values()).mean()
    ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in series_list)
    ss_total = sum(((s - grand_mean) ** 2).sum() for s in series_list)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else float("nan")

    result = {
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": eta_squared,
        "df_between": len(series_list) - 1,
        "df_within": int(sum(len(s) for s in series_list) - len(series_list)),
        "n_per_group": {k: int(len(s)) for k, s in clean.items()},
        "group_means": {k: float(s.mean()) for k, s in clean.items()},
        "significant": bool(p_value < alpha),
        "tukey_required": bool(p_value < alpha),
    }

    if p_value < alpha:
        labels, values = [], []
        for name, s in clean.items():
            values.extend(s.tolist())
            labels.extend([name] * len(s))
        table, records = tukey_records(values, labels, alpha=alpha)
        result["tukey_hsd"] = table
        result["tukey_records"] = records
    return result


def two_way_anova(df: pd.DataFrame, value_col: str, factor1: str, factor2: str) -> dict:
    """검증 3(시장 국면 x 전략) — 두 요인의 주효과+상호작용을 본다.

    df는 [value_col, factor1, factor2] 컬럼을 가진 long-format 테이블이어야 한다.
    예: value_col='daily_return', factor1='strategy', factor2='market_regime'
    eta_squared는 SS_effect / SS_total (Residual 포함 합) 이다.
    """
    formula = f"{value_col} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    ss_total = anova_table["sum_sq"].sum()
    eta_squared = (anova_table["sum_sq"] / ss_total).to_dict()

    return {"anova_table": anova_table, "eta_squared": eta_squared}
