import os
from typing import Optional

import pandas as pd
import streamlit as st
import plotly.express as px

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


def _market_state_col(df: pd.DataFrame) -> Optional[str]:
    """Return STATE or MARKET column name if present."""
    for c in ("STATE", "MARKET"):
        if c in df.columns:
            return c
    return None


def _run_proactive_monitoring(
    roster_df: pd.DataFrame, market_df: pd.DataFrame
) -> None:
    """Display st.warning() alerts for markets below SLA, stuck ROs, and abnormal stage delays."""
    alerts = []

    # Markets with SCS_PERCENT < 90
    scs_col = "SCS_PERCENT" if "SCS_PERCENT" in market_df.columns else (
        "SCS_PCT" if "SCS_PCT" in market_df.columns else None
    )
    state_col = _market_state_col(market_df)
    if scs_col and state_col:
        scs_series = pd.to_numeric(market_df[scs_col], errors="coerce")
        below = market_df.loc[scs_series < 90]
        for _, row in below.iterrows():
            market_name = row.get(state_col, "Unknown")
            pct = row.get(scs_col)
            if pd.notna(pct):
                try:
                    pct_val = float(pct)
                    alerts.append(
                        f"Market {market_name} success rate dropped below SLA ({pct_val:.2f}%)"
                    )
                except (TypeError, ValueError):
                    pass

    # Stuck roster operations (IS_STUCK == True or 1)
    if "IS_STUCK" in roster_df.columns:
        stuck_series = pd.to_numeric(roster_df["IS_STUCK"], errors="coerce").fillna(0)
        stuck_mask = stuck_series.astype(int) == 1
        if stuck_mask.any():
            state_col_roster = "CNT_STATE" if "CNT_STATE" in roster_df.columns else "STATE"
            if state_col_roster in roster_df.columns:
                stuck_states = roster_df.loc[stuck_mask, state_col_roster].dropna().unique().tolist()
                for s in stuck_states[:10]:  # cap to avoid spam
                    alerts.append(f"New stuck roster operation detected in {s}")
            else:
                alerts.append("New stuck roster operation detected")

    # Pipeline stage with abnormal average duration
    if "LATEST_STAGE_NM" in roster_df.columns and "DURATION_MINUTES" in roster_df.columns:
        dur = pd.to_numeric(roster_df["DURATION_MINUTES"], errors="coerce")
        roster_clean = roster_df.assign(DURATION_MINUTES=dur).dropna(subset=["DURATION_MINUTES"])
        roster_clean = roster_clean[roster_clean["DURATION_MINUTES"] >= 0]
        if not roster_clean.empty:
            stage_avg = roster_clean.groupby("LATEST_STAGE_NM")["DURATION_MINUTES"].mean()
            if len(stage_avg) >= 2:
                med = stage_avg.median()
                # Significant = e.g. > 2x median of stage averages
                threshold = max(med * 2.0, 1.0)
                abnormal = stage_avg[stage_avg >= threshold]
                for stage_name in abnormal.index[:5]:
                    alerts.append(f"{stage_name} stage showing abnormal delay")

    for msg in alerts:
        st.warning(f"ALERT: {msg}")


def _build_root_cause_chain(
    roster_df: pd.DataFrame, market_df: pd.DataFrame
) -> str:
    """Build markdown for Root Cause Analysis: Market → Stage → Orgs → Cause."""
    state_col = _market_state_col(market_df)
    scs_col = "SCS_PERCENT" if "SCS_PERCENT" in market_df.columns else (
        "SCS_PCT" if "SCS_PCT" in market_df.columns else None
    )
    if not state_col or not scs_col:
        return "Root cause chaining requires STATE/MARKET and SCS_PERCENT (or SCS_PCT) in the market dataset."

    scs_series = pd.to_numeric(market_df[scs_col], errors="coerce")
    low_markets = market_df.loc[scs_series < 90, state_col].dropna().unique().tolist()
    if not low_markets:
        return "No markets currently below 90% success rate; no root cause chain to display."

    roster_state_col = "CNT_STATE" if "CNT_STATE" in roster_df.columns else "STATE"
    org_col = "ORG_NM" if "ORG_NM" in roster_df.columns else "ORG_NAME"
    has_duration = "DURATION_MINUTES" in roster_df.columns and "LATEST_STAGE_NM" in roster_df.columns

    lines = ["### Root Cause Analysis", "", "**Root Cause Chain**", ""]

    for market_name in low_markets[:5]:
        lines.append(f"**Market:** {market_name}")
        lines.append("")
        lines.append("↓")
        lines.append("")

        # Filter roster to this market
        if roster_state_col not in roster_df.columns:
            lines.append("Pipeline Stage: (data not available)")
            lines.append("")
            continue
        sub = roster_df[roster_df[roster_state_col].astype(str).str.strip() == str(market_name).strip()]
        if sub.empty:
            lines.append("Pipeline Stage: (no roster data for this market)")
            lines.append("")
            continue

        bottleneck_stage = "Unknown"
        if has_duration:
            dur = pd.to_numeric(sub["DURATION_MINUTES"], errors="coerce")
            sub_clean = sub.assign(_dur=dur).dropna(subset=["_dur"])
            sub_clean = sub_clean[sub_clean["_dur"] >= 0]
            if not sub_clean.empty:
                stage_avg = sub_clean.groupby("LATEST_STAGE_NM")["_dur"].median()
                if not stage_avg.empty:
                    bottleneck_stage = stage_avg.idxmax()

        lines.append(f"**Pipeline Stage:** {bottleneck_stage}")
        lines.append("")
        lines.append("↓")
        lines.append("")

        # Orgs contributing most to delays
        lines.append("**Organizations Contributing Most:**")
        if has_duration and org_col in sub.columns:
            dur = pd.to_numeric(sub["DURATION_MINUTES"], errors="coerce")
            sub2 = sub.assign(_dur=dur).dropna(subset=["_dur"])
            sub2 = sub2[sub2["_dur"] >= 0]
            if not sub2.empty:
                org_med = sub2.groupby(org_col)["_dur"].median().sort_values(ascending=False).head(5)
                for org_name in org_med.index:
                    lines.append(f"- {org_name}")
            else:
                lines.append("- (no duration data)")
        else:
            lines.append("- (duration or organization column not available)")
        lines.append("")
        lines.append("↓")
        lines.append("")
        lines.append(f"**Likely Cause:** Processing bottleneck at {bottleneck_stage} stage")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip()


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

    # Proactive Monitoring – alerts at top of dashboard
    _run_proactive_monitoring(roster_df, market_df)

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

    # SECTION 6 – Data Visualizations
    st.markdown("### Data Visualizations")

    # A. Record Outcome Distribution (Pie Chart)
    outcome_fig = None
    if {"SCS_REC_CNT", "FAIL_REC_CNT", "REJ_REC_CNT"}.issubset(filtered_roster_df.columns):
        success_total = float(
            pd.to_numeric(filtered_roster_df["SCS_REC_CNT"], errors="coerce").fillna(0).sum()
        )
        fail_total = float(
            pd.to_numeric(filtered_roster_df["FAIL_REC_CNT"], errors="coerce").fillna(0).sum()
        )
        reject_total = float(
            pd.to_numeric(filtered_roster_df["REJ_REC_CNT"], errors="coerce").fillna(0).sum()
        )
        outcome_df = pd.DataFrame(
            {
                "Outcome": ["Success", "Fail", "Reject"],
                "Count": [success_total, fail_total, reject_total],
            }
        )
        outcome_fig = px.pie(
            outcome_df,
            names="Outcome",
            values="Count",
            title="Record Processing Results",
        )

    # B. Pipeline Stage Distribution (Pie Chart)
    stage_fig = None
    if "LATEST_STAGE_NM" in filtered_roster_df.columns:
        stage_counts = (
            filtered_roster_df["LATEST_STAGE_NM"]
            .astype(str)
            .fillna("Unknown")
            .value_counts()
            .reset_index()
        )
        stage_counts.columns = ["Stage", "Count"]
        stage_fig = px.pie(
            stage_counts,
            names="Stage",
            values="Count",
            title="Roster Pipeline Stage Distribution",
        )

    col1_viz, col2_viz = st.columns(2)
    with col1_viz:
        if outcome_fig is not None:
            st.plotly_chart(outcome_fig, use_container_width=True)
        else:
            st.info("Record outcome distribution is unavailable – missing SCS/FAIL/REJECT counts.")

    with col2_viz:
        if stage_fig is not None:
            st.plotly_chart(stage_fig, use_container_width=True)
        else:
            st.info("Pipeline stage distribution is unavailable – `LATEST_STAGE_NM` column missing.")

    # C. Average Processing Duration by Stage (Bar Chart)
    if {"LATEST_STAGE_NM", "DURATION_MINUTES"}.issubset(filtered_roster_df.columns):
        dur_df = filtered_roster_df.copy()
        dur_df["DURATION_MINUTES"] = pd.to_numeric(
            dur_df["DURATION_MINUTES"], errors="coerce"
        )
        dur_df = dur_df[dur_df["DURATION_MINUTES"].notna()]
        if not dur_df.empty:
            stage_duration = (
                dur_df.groupby("LATEST_STAGE_NM")["DURATION_MINUTES"]
                .mean()
                .reset_index()
                .sort_values("DURATION_MINUTES", ascending=False)
            )
            dur_fig = px.bar(
                stage_duration,
                x="LATEST_STAGE_NM",
                y="DURATION_MINUTES",
                title="Average Processing Time by Pipeline Stage",
                labels={"LATEST_STAGE_NM": "Pipeline Stage", "DURATION_MINUTES": "Avg Duration (minutes)"},
            )
            st.plotly_chart(dur_fig, use_container_width=True)
        else:
            st.info("No valid `DURATION_MINUTES` values available for duration chart.")
    else:
        st.info(
            "Average processing duration by stage is unavailable – "
            "`LATEST_STAGE_NM` and/or `DURATION_MINUTES` columns are missing."
        )

    # Market Performance Visualization – success rate by state/market
    st.markdown("### Market Performance Visualization")
    market_state_col = _market_state_col(market_df)
    scs_col_viz = "SCS_PERCENT" if "SCS_PERCENT" in market_df.columns else (
        "SCS_PCT" if "SCS_PCT" in market_df.columns else None
    )
    if market_state_col and scs_col_viz:
        agg = market_df.groupby(market_state_col, as_index=False).agg(
            **{"market_success_rate": (scs_col_viz, "mean")}
        )
        if "market_success_rate" in agg.columns:
            agg["market_success_rate"] = pd.to_numeric(agg["market_success_rate"], errors="coerce")
            agg = agg.dropna(subset=["market_success_rate"])
        if not agg.empty:
            agg["SLA_met"] = agg["market_success_rate"] >= 90
            fig_market = px.bar(
                agg,
                x=market_state_col,
                y="market_success_rate",
                title="Market Success Rate by State",
                color="SLA_met",
                color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
            )
            fig_market.update_layout(showlegend=False)
            st.plotly_chart(fig_market, use_container_width=True)
        else:
            st.info("No valid market success rate data available for visualization.")
    else:
        st.info(
            "Market success rate chart requires STATE or MARKET and SCS_PERCENT (or SCS_PCT) "
            "in the aggregated operational metrics dataset."
        )

    # SECTION 4 – Data Insights
    st.markdown("### Data Insights")

    # Root Cause Analysis – Market → Stage → Orgs → Cause
    root_cause_md = _build_root_cause_chain(roster_df, market_df)
    st.markdown(root_cause_md)

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

