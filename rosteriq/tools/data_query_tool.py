from typing import Optional

import pandas as pd

from utils.helpers import filter_dataframe, run_duckdb_query


def filter_data(
    roster_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame] = None,
    state: Optional[str] = None,
    organization: Optional[str] = None,
    lob: Optional[str] = None,
    pipeline_stage: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for common filtering operations on the roster dataset.
    """
    return filter_dataframe(
        roster_df,
        state=state,
        organization=organization,
        lob=lob,
        pipeline_stage=pipeline_stage,
    )


def sql_query(
    roster_df: pd.DataFrame,
    market_df: Optional[pd.DataFrame],
    query: str,
) -> pd.DataFrame:
    """
    Execute an arbitrary DuckDB SQL query against the two datasets.
    """
    return run_duckdb_query(roster_df, market_df, query)

