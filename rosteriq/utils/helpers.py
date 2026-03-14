import os
from typing import Optional, Tuple, List

import duckdb
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def csv_path(filename: str) -> str:
    """
    Public helper for building CSV paths under the project's data directory.
    """
    return os.path.join(DATA_DIR, filename)


def filter_dataframe(
    df: pd.DataFrame,
    state: Optional[str] = None,
    organization: Optional[str] = None,
    lob: Optional[str] = None,
    pipeline_stage: Optional[str] = None,
    state_col: str = "CNT_STATE",
    org_col: str = "ORG_NM",
    lob_col: str = "LOB",
    stage_col: str = "LATEST_STAGE_NM",
) -> pd.DataFrame:
    """
    Generic filtering utility for the roster dataframes.
    Column names can be overridden to match your schema.
    """
    filtered = df.copy()

    if state and state_col in filtered.columns:
        filtered = filtered[filtered[state_col] == state]
    if organization and org_col in filtered.columns:
        filtered = filtered[filtered[org_col] == organization]
    if lob and lob_col in filtered.columns:
        filtered = filtered[filtered[lob_col] == lob]
    if pipeline_stage and stage_col in filtered.columns:
        filtered = filtered[filtered[stage_col] == pipeline_stage]

    return filtered


def join_roster_and_market(
    roster_df: pd.DataFrame,
    market_df: pd.DataFrame,
    roster_state_col: str = "CNT_STATE",
    market_state_col: str = "MARKET",
    suffixes: Tuple[str, str] = ("_ROSTER", "_MARKET"),
) -> pd.DataFrame:
    """
    Join the two datasets using a state/market key.
    By default assumes both use 'STATE' but can be overridden.
    """
    # Be resilient to schema differences (e.g., MARKET instead of STATE, or whitespace/case issues)
    roster = roster_df.copy()
    market = market_df.copy()
    roster.columns = roster.columns.astype(str).str.strip()
    market.columns = market.columns.astype(str).str.strip()

    def _resolve_key(df: pd.DataFrame, preferred: str) -> str:
        if preferred in df.columns:
            return preferred
        # Fall back to any column that matches preferred ignoring case/whitespace
        pref_norm = preferred.strip().upper()
        for c in df.columns:
            if c.strip().upper() == pref_norm:
                return c
        # Common fallback: MARKET
        for c in df.columns:
            if c.strip().upper() == "MARKET":
                return c
        return preferred  # will raise a clear KeyError downstream if missing

    left_key = _resolve_key(roster, roster_state_col)
    right_key = _resolve_key(market, market_state_col)

    if left_key not in roster.columns or right_key not in market.columns:
        raise KeyError(
            "Unable to join roster and market datasets. "
            f"Expected join keys like '{roster_state_col}'/'{market_state_col}' (or 'MARKET'). "
            f"Resolved keys: left='{left_key}' right='{right_key}'. "
            f"Roster columns: {list(roster.columns)}. "
            f"Market columns: {list(market.columns)}."
        )

    # Merge using the cleaned copies (so resolved keys actually exist)
    return roster.merge(
        market,
        left_on=left_key,
        right_on=right_key,
        how="left",
        suffixes=suffixes,
    )


def run_duckdb_query(
    roster_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    query: str,
    roster_table_name: str = "roster_processing_details",
    market_table_name: str = "aggregated_operational_metrics",
) -> pd.DataFrame:
    """
    Execute an ad-hoc DuckDB SQL query against the two dataframes.
    The dataframes are registered as in-memory tables.
    """
    con = duckdb.connect(database=":memory:")
    con.register(roster_table_name, roster_df)
    if market_df is not None:
        con.register(market_table_name, market_df)
    result = con.execute(query).fetchdf()
    con.close()
    return result


def list_unique_values(df: pd.DataFrame, columns: List[str]) -> dict:
    """
    Helper to list unique values for filters in the UI.
    """
    uniques = {}
    for col in columns:
        if col in df.columns:
            uniques[col] = sorted(df[col].dropna().unique().tolist())
    return uniques

