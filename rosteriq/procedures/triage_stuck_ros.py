from typing import Optional

import pandas as pd


def triage_stuck_ros(
    roster_df: pd.DataFrame,
    state: Optional[str] = None,
    organization: Optional[str] = None,
) -> pd.DataFrame:
    """
    Find all rows where IS_STUCK = True and rank by duration and red health flags.
    Expected columns (if present):
      - IS_STUCK (bool or equivalent)
      - one of the duration fields (e.g., PRE_PROCESSING_DURATION, ISF_GEN_DURATION, etc.)
      - one of the health fields (e.g., PRE_PROCESSING_HEALTH, ISF_GEN_HEALTH, etc.)
      - CNT_STATE
      - ORG_NM
    """
    df = roster_df.copy()

    required = ["RO_ID", "ORG_NM", "CNT_STATE", "LATEST_STAGE_NM", "FILE_STATUS_CD", "IS_STUCK"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Return empty frame with the expected output schema
        return pd.DataFrame(columns=["RO_ID", "ORG_NM", "CNT_STATE", "LATEST_STAGE_NM", "FILE_STATUS_CD"])

    # Dataset uses 0/1 for IS_STUCK; treat 1 as stuck
    stuck_mask = pd.to_numeric(df["IS_STUCK"], errors="coerce").fillna(0).astype(int) == 1
    df = df[stuck_mask]

    if state:
        df = df[df["CNT_STATE"].astype(str).str.upper() == str(state).upper()]
    if organization:
        df = df[df["ORG_NM"] == organization]

    # Return the exact fields requested for downstream reporting / LLM formatting
    stuck_df = df[["RO_ID", "ORG_NM", "CNT_STATE", "LATEST_STAGE_NM", "FILE_STATUS_CD"]].copy()
    return stuck_df

