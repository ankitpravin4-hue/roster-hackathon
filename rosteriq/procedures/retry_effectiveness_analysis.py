import pandas as pd


def retry_effectiveness_analysis(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare RUN_NO = 1 vs RUN_NO > 1 to understand retry impact on success.

    Expected columns:
      - RUN_NO
      - SCS_PCT or SCS_PERCENT or SCS_REC_CNT + TOT_REC_CNT
    """
    df = roster_df.copy()

    if "RUN_NO" not in df.columns:
        return df.iloc[0:0]

    base = df[df["RUN_NO"] == 1]
    retry = df[df["RUN_NO"] > 1]

    def _success_pct(sub: pd.DataFrame) -> float:
        cols = sub.columns
        if "SCS_PCT" in cols:
            return float(sub["SCS_PCT"].mean())
        if "SCS_PERCENT" in cols:
            return float(sub["SCS_PERCENT"].mean())
        if "SCS_REC_CNT" in cols and "TOT_REC_CNT" in cols:
            ratio = sub["SCS_REC_CNT"].sum() / max(sub["TOT_REC_CNT"].sum(), 1)
            return float(ratio * 100)
        return float("nan")

    base_scs = _success_pct(base)
    retry_scs = _success_pct(retry)
    lift = retry_scs - base_scs if pd.notna(base_scs) and pd.notna(retry_scs) else float("nan")

    summary = pd.DataFrame(
        [
            {
                "RUN_BUCKET": "RUN_NO = 1",
                "AVG_SUCCESS_PCT": base_scs,
            },
            {
                "RUN_BUCKET": "RUN_NO > 1",
                "AVG_SUCCESS_PCT": retry_scs,
            },
            {
                "RUN_BUCKET": "LIFT",
                "AVG_SUCCESS_PCT": lift,
            },
        ]
    )
    return summary

