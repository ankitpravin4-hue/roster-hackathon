import os
from typing import Optional

import pandas as pd
import streamlit as st

# NOTE:
# This file is executed as `streamlit run rosteriq/app.py` with the
# working directory set to the `rosteriq/` folder. That means we should
# use local (relative) imports, not `from rosteriq...`, otherwise
# the `rosteriq` package will not be found in deployed environments.
from agent import RosterIQAgent
from dashboards.market_success_trend import build_market_success_trend
from dashboards.pipeline_heatmap import build_pipeline_heatmap
from dashboards.record_quality_chart import build_record_quality_chart
from dashboards.retry_lift_chart import build_retry_lift_chart
from procedures.retry_effectiveness_analysis import retry_effectiveness_analysis
from procedures.triage_stuck_ros import triage_stuck_ros
from tools.visualization_tool import stuck_ro_tracker
from utils.helpers import csv_path, filter_dataframe, list_unique_values
@st.cache_data(show_spinner=False)
def _load_roster_df() -> pd.DataFrame:
    path = csv_path("roster_processing_details.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected roster_processing_details.csv in {os.path.dirname(path)}, but it was not found."
        )
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_market_df() -> pd.DataFrame:
    path = csv_path("aggregated_operational_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected aggregated_operational_metrics.csv in {os.path.dirname(path)}, but it was not found."
        )
    return pd.read_csv(path)


def _load_data() -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    try:
        roster_df = _load_roster_df()
        market_df = _load_market_df()
        return roster_df, market_df
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Place `roster_processing_details.csv` and `aggregated_operational_metrics.csv` "
            "in the `data/` folder and refresh the app."
        )
        return None


def main() -> None:
    st.set_page_config(
        page_title="RosterIQ – Provider Roster Intelligence Agent",
        layout="wide",
    )

    st.title("RosterIQ – Provider Roster Intelligence Agent")

    # Load data
    data = _load_data()
    if data is None:
        return

    roster_df, market_df = data

    # Normalize dataset schema
    roster_df.columns = roster_df.columns.str.strip()
    market_df.columns = market_df.columns.str.strip()

    # SECTION 1 – Filters (sidebar)
    st.sidebar.header("Filters")
    uniques = list_unique_values(roster_df, ["CNT_STATE", "ORG_NM", "LOB", "LATEST_STAGE_NM"])

    state = st.sidebar.selectbox("State", ["(All)"] + uniques.get("CNT_STATE", []))
    if state == "(All)":
        state = None
    organization = st.sidebar.selectbox(
        "Organization", ["(All)"] + uniques.get("ORG_NM", [])
    )
    if organization == "(All)":
        organization = None
    lob = st.sidebar.selectbox("Line of Business (LOB)", ["(All)"] + uniques.get("LOB", []))
    if lob == "(All)":
        lob = None
    pipeline_stage = st.sidebar.selectbox(
        "Pipeline Stage", ["(All)"] + uniques.get("LATEST_STAGE_NM", [])
    )
    if pipeline_stage == "(All)":
        pipeline_stage = None

    st.sidebar.markdown("---")
    st.sidebar.subheader("About RosterIQ")
    st.sidebar.write(
        "RosterIQ is an AI copilot for provider roster operations. "
        "Use the filters to focus on a market or organization, then ask natural language questions."
    )

    # Filtered views for metrics/dashboards
    filtered_roster_df = filter_dataframe(
        roster_df,
        state=state,
        organization=organization,
        lob=lob,
        pipeline_stage=pipeline_stage,
    )
    filtered_market_df = market_df.copy()
    if state and "STATE" in filtered_market_df.columns:
        filtered_market_df = filtered_market_df[filtered_market_df["STATE"] == state]

    # Initialize agent – uses full datasets but is aware of filter context
    agent = RosterIQAgent(roster_df, market_df)

    # SECTION 2 – AI Question Box
    st.subheader("Ask the RosterIQ Agent")
    default_q = "Which roster operations are stuck and what is the bottleneck?"
    question = st.text_input(
        "Ask a question about roster pipeline health, quality, or market performance:",
        value=default_q,
    )
    analyze_clicked = st.button("Analyze", type="primary")

    response = ""
    analytics = {}

    if analyze_clicked and question.strip():
        with st.spinner("Analyzing roster operations with RosterIQ..."):
            response, analytics = agent.answer(
                question,
                state=state,
                organization=organization,
                lob=lob,
                pipeline_stage=pipeline_stage,
            )

    # SECTION 3 – AI Answer
    st.markdown("### AI Answer")
    if response:
        st.markdown(response)
    else:
        st.info("Ask a question above and click **Analyze** to see AI insights.")

    # SECTION 5 – Dashboard Metrics (use filtered data)
    st.markdown("### Dashboard Metrics")
    total_records = int(filtered_roster_df.get("TOT_REC_CNT", pd.Series([0])).sum())
    total_rows = len(filtered_roster_df)

    # Rejection rate
    rejection_rate_pct = None
    if "REJ_REC_CNT" in filtered_roster_df.columns:
        if "TOT_REC_CNT" in filtered_roster_df.columns:
            rej_sum = filtered_roster_df["REJ_REC_CNT"].sum()
            tot_sum = filtered_roster_df["TOT_REC_CNT"].sum()
            if tot_sum > 0:
                rejection_rate_pct = 100.0 * rej_sum / tot_sum

    # Markets at risk
    markets_at_risk = 0
    if "SCS_PERCENT" in filtered_market_df.columns:
        markets_at_risk = int((filtered_market_df["SCS_PERCENT"] < 90).sum())
    elif "SCS_PCT" in filtered_market_df.columns:
        markets_at_risk = int((filtered_market_df["SCS_PCT"] < 90).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Files (rows)", value=total_rows)
    col2.metric(
        "Total Records",
        value=f"{total_records:,}",
    )
    if rejection_rate_pct is not None:
        col3.metric("Rejection Rate", value=f"{rejection_rate_pct:.1f}%")
    else:
        col3.metric("Markets at Risk", value=str(markets_at_risk))

    # SECTION 4 – Data Insights
    st.markdown("### Data Insights")
    if analytics:
        stuck_df = analytics.get("triage_stuck_ros")
        if isinstance(stuck_df, pd.DataFrame) and not stuck_df.empty:
            st.markdown("#### Stuck Roster Operations")
            st.dataframe(stuck_df.head(200))

        quality_df = analytics.get("record_quality_audit")
        if isinstance(quality_df, pd.DataFrame) and not quality_df.empty:
            st.markdown("#### Record Quality Audit")
            st.dataframe(quality_df.head(200))

        market_df_report = analytics.get("market_health_report")
        if isinstance(market_df_report, pd.DataFrame) and not market_df_report.empty:
            st.markdown("#### Market Health Report")
            st.dataframe(market_df_report.head(200))

        anomalies = analytics.get("anomalies")
        if isinstance(anomalies, dict) and anomalies:
            st.markdown("#### Anomalies")
            for name, df_anom in anomalies.items():
                if isinstance(df_anom, pd.DataFrame) and not df_anom.empty:
                    st.markdown(f"**{name.replace('_', ' ').title()}**")
                    st.dataframe(df_anom.head(200))
    else:
        st.info(
            "After running an analysis, detailed tables for stuck operations, quality, "
            "market health, and anomalies will appear here."
        )


def generate_operational_report(
    roster_df: pd.DataFrame,
    market_df: pd.DataFrame,
    state: Optional[str],
    organization: Optional[str],
) -> str:
    """
    Generate a structured operational report for a given state/organization.
    """
    scope_desc = ""
    if state:
        scope_desc += f"{state} "
    else:
        scope_desc += "All States "
    if organization:
        scope_desc += f"– {organization}"
    else:
        scope_desc += "– All Organizations"

    scoped = roster_df.copy()
    if state and "CNT_STATE" in scoped.columns:
        scoped = scoped[scoped["CNT_STATE"] == state]
    if organization:
        org_col = "ORG_NM" if "ORG_NM" in scoped.columns else "ORG_NAME"
        if org_col in scoped.columns:
            scoped = scoped[scoped[org_col] == organization]

    total_files = len(scoped)

    # Average success rate
    scs_pct = None
    if "SCS_PCT" in scoped.columns:
        scs_pct = scoped["SCS_PCT"].mean()
    elif "SCS_PERCENT" in scoped.columns:
        scs_pct = scoped["SCS_PERCENT"].mean()
    elif "SCS_REC_CNT" in scoped.columns and "TOT_REC_CNT" in scoped.columns:
        scs_pct = (
            scoped["SCS_REC_CNT"].sum() / max(scoped["TOT_REC_CNT"].sum(), 1) * 100
        )

    # Top failing organizations by rejection rate
    top_fail_orgs = ""
    if "REJ_REC_CNT" in scoped.columns:
        if "TOT_REC_CNT" in scoped.columns:
            scoped["REJ_RATE"] = scoped["REJ_REC_CNT"] / scoped["TOT_REC_CNT"].replace(0, pd.NA)
        else:
            scoped["REJ_RATE"] = 0
        group_col = "ORG_NM" if "ORG_NM" in scoped.columns else "ORG_NAME"
        org_group = (
            scoped.groupby(group_col)["REJ_RATE"].mean().sort_values(ascending=False).head(5)
        )
        top_fail_orgs = "\n".join(
            [f"- {idx}: {val:.1%} rejection rate" for idx, val in org_group.items()]
        )

    # Bottleneck pipeline stage by average duration or stuck count
    bottleneck = "Unknown"
    if "LATEST_STAGE_NM" in scoped.columns:
        # Approximate bottleneck by counting stuck files per latest stage
        if "IS_STUCK" in scoped.columns:
            stage_perf = scoped.groupby("LATEST_STAGE_NM")["IS_STUCK"].sum()
            if not stage_perf.empty:
                bottleneck = stage_perf.sort_values(ascending=False).index[0]

    # Recommended actions
    recs = [
        "Prioritize triage of stuck roster operations in the bottleneck stage.",
        "Deep-dive into top failing organizations to identify schema or data quality issues.",
        "Align validation and business rules with CMS/Medicaid guidance to reduce external rejections.",
        "Monitor markets below 95% success for SLA and contractual impact.",
    ]

    title = f"{scope_desc} Market Health Report"
    md = f"## {title}\n\n"
    md += f"- **Total Files Processed**: {total_files}\n"
    if scs_pct is not None:
        md += f"- **Average Success Rate**: {scs_pct:.2f}%\n"
    else:
        md += "- **Average Success Rate**: Not available\n"

    md += "\n### Top Failing Organizations (by rejection rate)\n"
    md += top_fail_orgs or "No significant rejection patterns detected.\n"

    md += "\n### Pipeline Bottleneck\n"
    md += f"- **Likely Bottleneck Stage**: {bottleneck}\n"

    md += "\n### Recommended Actions\n"
    for r in recs:
        md += f"- {r}\n"

    return md


if __name__ == "__main__":
    main()

