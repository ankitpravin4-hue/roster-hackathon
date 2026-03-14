from typing import Dict, List

import pandas as pd


def detect_anomalies(roster_df: pd.DataFrame, market_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Detect key anomalies:
      - High rejection rate files (REJ_REC_CNT / TOT_REC_CNT > 0.3)
      - Stuck roster operations (IS_STUCK = True)
      - Stage duration anomalies (stage_duration > 2x avg)
      - Markets with SCS_PERCENT below 90
    Returns a dict of DataFrames keyed by anomaly type.
    """
    anomalies: Dict[str, pd.DataFrame] = {}

    df = roster_df.copy()
    cols = df.columns

    if "REJ_REC_CNT" in cols:
        if "TOT_REC_CNT" in cols:
            tot = df["TOT_REC_CNT"]
        else:
            tot = df["REJ_REC_CNT"]
        df["REJ_RATE"] = df["REJ_REC_CNT"] / tot.replace(0, pd.NA)
        anomalies["high_rejection_files"] = df[df["REJ_RATE"] > 0.3]

    if "IS_STUCK" in cols:
        anomalies["stuck_ro_operations"] = df[df["IS_STUCK"] == True]  # noqa: E712

    duration_candidates: List[str] = [
        "STAGE_DURATION",
        "TOTAL_DURATION",
        "DURATION_HOURS",
    ]
    for cand in duration_candidates:
        if cand in cols:
            avg = df[cand].mean()
            anomalies["stage_duration_anomalies"] = df[df[cand] > 2 * avg]
            break

    mcols = market_df.columns
    if "SCS_PERCENT" in mcols:
        anomalies["low_market_success"] = market_df[market_df["SCS_PERCENT"] < 90]
    elif "SCS_PCT" in mcols:
        anomalies["low_market_success"] = market_df[market_df["SCS_PCT"] < 90]

    return anomalies

