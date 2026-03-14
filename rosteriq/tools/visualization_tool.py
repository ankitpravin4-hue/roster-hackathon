from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


def pipeline_health_heatmap(
    roster_df: pd.DataFrame,
    org_col: str = "ORG_NM",
    stage_col: str = "LATEST_STAGE_NM",
    health_col: str = "PRE_PROCESSING_HEALTH",
):
    """
    Organization vs Pipeline Stage heatmap, colored by health flag.
    """
    if not {org_col, stage_col, health_col}.issubset(roster_df.columns):
        return go.Figure()

    pivot = (
        roster_df[[org_col, stage_col, health_col]]
        .dropna()
        .pivot_table(index=org_col, columns=stage_col, values=health_col, aggfunc="first")
    )

    health_map = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    z = pivot.replace(health_map)

    fig = go.Figure(
        data=go.Heatmap(
            z=z.values,
            x=z.columns.astype(str),
            y=z.index.astype(str),
            colorscale=[
                [0.0, "green"],
                [0.5, "yellow"],
                [1.0, "red"],
            ],
            colorbar=dict(title="Health"),
        )
    )
    fig.update_layout(
        title="Pipeline Health Heatmap",
        xaxis_title="Pipeline Stage",
        yaxis_title="Organization",
    )
    return fig


def record_quality_breakdown(
    roster_df: pd.DataFrame,
    org_col: str = "ORG_NM",
    scs_col: str = "SCS_REC_CNT",
    fail_col: str = "FAIL_REC_CNT",
    skip_col: str = "SKIP_REC_CNT",
    rej_col: str = "REJ_REC_CNT",
):
    """
    Stacked bar chart showing SCS / FAIL / SKIP / REJ by organization.
    """
    for col in [scs_col, fail_col, skip_col, rej_col]:
        if col not in roster_df.columns:
            roster_df[col] = 0

    agg = (
        roster_df.groupby(org_col)[[scs_col, fail_col, skip_col, rej_col]]
        .sum()
        .reset_index()
    )
    fig = go.Figure()
    fig.add_bar(x=agg[org_col], y=agg[scs_col], name="SCS")
    fig.add_bar(x=agg[org_col], y=agg[fail_col], name="FAIL")
    fig.add_bar(x=agg[org_col], y=agg[skip_col], name="SKIP")
    fig.add_bar(x=agg[org_col], y=agg[rej_col], name="REJ")
    fig.update_layout(
        barmode="stack",
        title="Record Quality Breakdown by Organization",
        xaxis_title="Organization",
        yaxis_title="Record Count",
    )
    return fig


def market_success_trend(
    market_df: pd.DataFrame,
    date_col: str = "MONTH",
    scs_pct_col: str = "SCS_PERCENT",
    state: Optional[str] = None,
):
    """
    Line chart Month vs SCS_PERCENT with threshold line at 95%.
    """
    df = market_df.copy()
    if state and "STATE" in df.columns:
        df = df[df["STATE"] == state]

    if scs_pct_col not in df.columns and "SCS_PCT" in df.columns:
        scs_pct_col = "SCS_PCT"

    if not {date_col, scs_pct_col}.issubset(df.columns):
        return go.Figure()

    df = df.sort_values(by=date_col)
    fig = px.line(df, x=date_col, y=scs_pct_col, color="STATE" if "STATE" in df.columns else None)
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="95% target")
    fig.update_layout(
        title="Market Success Trend",
        xaxis_title="Month",
        yaxis_title="Success Percent",
    )
    return fig


def retry_effectiveness_chart(summary_df: pd.DataFrame):
    """
    Bar chart comparing first iteration vs retry success using the summary
    produced by retry_effectiveness_analysis.
    """
    if not {"RUN_BUCKET", "AVG_SUCCESS_PCT"}.issubset(summary_df.columns):
        return go.Figure()

    fig = px.bar(
        summary_df[summary_df["RUN_BUCKET"].isin(["RUN_NO = 1", "RUN_NO > 1"])],
        x="RUN_BUCKET",
        y="AVG_SUCCESS_PCT",
        title="Retry Effectiveness",
        labels={"AVG_SUCCESS_PCT": "Average Success %"},
    )
    return fig


def stuck_ro_tracker(stuck_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Simple helper that returns the top N stuck ROs ranked by severity.
    The sorting logic is handled by triage_stuck_ros; this function just truncates.
    """
    return stuck_df.head(top_n)

