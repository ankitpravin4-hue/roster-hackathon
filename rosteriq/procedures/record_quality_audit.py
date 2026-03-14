import pandas as pd


def record_quality_audit(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate record quality metrics and flag files with low success percentage.

    Expected columns (if present):
      - FAIL_REC_CNT
      - REJ_REC_CNT
      - SKIP_REC_CNT
      - SCS_REC_CNT or SCS_CNT
      - TOT_REC_CNT
      - SCS_PCT or SCS_PERCENT
    """
    df = roster_df.copy()

    cols = df.columns

    fail = df["FAIL_REC_CNT"] if "FAIL_REC_CNT" in cols else 0
    rej = df["REJ_REC_CNT"] if "REJ_REC_CNT" in cols else 0
    skip = df["SKIP_REC_CNT"] if "SKIP_REC_CNT" in cols else 0
    scs = df["SCS_REC_CNT"] if "SCS_REC_CNT" in cols else (df["SCS_CNT"] if "SCS_CNT" in cols else 0)

    if "TOT_REC_CNT" in cols:
        tot = df["TOT_REC_CNT"]
    else:
        tot = fail + rej + skip + scs

    df["TOT_REC_CNT_COMPUTED"] = tot
    df["QUALITY_FAIL_RATE"] = (fail + rej) / df["TOT_REC_CNT_COMPUTED"].replace(0, pd.NA)
    df["QUALITY_REJ_RATE"] = rej / df["TOT_REC_CNT_COMPUTED"].replace(0, pd.NA)

    if "SCS_PCT" in cols:
        df["SCS_PCT_EFFECTIVE"] = df["SCS_PCT"]
    elif "SCS_PERCENT" in cols:
        df["SCS_PCT_EFFECTIVE"] = df["SCS_PERCENT"]
    else:
        df["SCS_PCT_EFFECTIVE"] = scs / df["TOT_REC_CNT_COMPUTED"].replace(0, pd.NA) * 100

    df["QUALITY_FLAG_LOW_SUCCESS"] = df["SCS_PCT_EFFECTIVE"] < 95

    return df

