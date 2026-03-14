from typing import Optional

import pandas as pd

from utils.helpers import join_roster_and_market


def market_health_report(
    roster_df: pd.DataFrame,
    market_df: pd.DataFrame,
    state: Optional[str] = None,
) -> pd.DataFrame:
    """
    Join datasets by state and compare rejection rate with market SCS_PERCENT.
    Flags markets where high rejection lines up with low success percent.
    """
    joined = join_roster_and_market(roster_df, market_df)

    if state and "STATE" in joined.columns:
        joined = joined[joined["STATE"] == state]

    # Compute file-level rejection rate.
    cols = joined.columns

    rej = joined["REJ_REC_CNT"] if "REJ_REC_CNT" in cols else pd.Series(0, index=joined.index)
    if "TOT_REC_CNT" in cols:
        tot = joined["TOT_REC_CNT"]
    else:
        # Fall back to a non-zero-safe denominator series (avoid scalar ints)
        tot = pd.Series(0, index=joined.index)

    joined["FILE_REJ_RATE"] = rej / tot.replace(0, pd.NA)

    if "SCS_PERCENT" in cols:
        joined["MARKET_SCS_PERCENT"] = joined["SCS_PERCENT"]
    elif "SCS_PCT" in cols:
        joined["MARKET_SCS_PERCENT"] = joined["SCS_PCT"]
    else:
        joined["MARKET_SCS_PERCENT"] = None

    joined["IS_MARKET_AT_RISK"] = (joined["FILE_REJ_RATE"] > 0.3) | (
        joined["MARKET_SCS_PERCENT"].fillna(100) < 90
    )

    return joined

