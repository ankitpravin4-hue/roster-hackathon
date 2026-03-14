import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple
import warnings

load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI

from memory.episodic_memory import EpisodicMemory
from memory.procedural_memory import ProceduralRegistry
from memory.semantic_memory import SemanticMemory
from procedures.triage_stuck_ros import triage_stuck_ros
from procedures.record_quality_audit import record_quality_audit
from procedures.market_health_report import market_health_report
from procedures.retry_effectiveness_analysis import retry_effectiveness_analysis
from tools.anomaly_detection import detect_anomalies
from tools.data_query_tool import filter_data
from tools.web_search_tool import web_search


def _get_llm() -> ChatOpenAI:
    """
    Configure the underlying LLM.
    This is wired to OpenRouter by default but can be pointed to Gemini
    or another provider by changing base_url and model.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in the environment to use the agent."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("ROSTERIQ_MODEL_NAME", "openrouter/hunter-alpha")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,
    )


AGENT_PROMPT_TEMPLATE = """
You are RosterIQ – a memory-driven provider roster intelligence agent for a health plan.

You will be given:
- The user's question
- Active filter context
- Data-driven findings from procedures/tools (a summarized analytics string)
- Episodic memory snippets
- A semantic knowledge base

Your job: produce a **dashboard-ready operational intelligence report**.

User question:
{question}

Relevant filters / context:
{filter_context}

Data-driven findings from procedures/tools (summarized):
{analytics_summary}

Computed operational metrics (authoritative; use these numbers, NEVER invent your own):
{analytics_metrics}

Episodic memory (past similar investigations):
{episodic_context}

Domain knowledge:
{semantic_knowledge}

CRITICAL REASONING RULES:
- Treat values in `analytics_metrics` as the *only* source of truth for numbers. Never compute your own rates from prose.
- Let `market_success` represent the **market (file-level) success rate**; let `success_rate` represent the **record-level success rate**.
- Let `threshold` (typically 90) be the SLA threshold for market success.
- If `market_success` is not None **and** `market_success >= threshold`, the user's premise that success is "below threshold" is FALSE. In that case:
  - Do NOT perform root-cause analysis.
  - Explicitly say that the market is currently *above* threshold and give the numeric value.
  - You may still highlight emerging risks (e.g., top failing orgs, bottlenecks) but label them as **potential risk**, not confirmed root cause.
- When BOTH `market_success` and `success_rate` are present, immediately after the Market Summary add a short explanation clarifying why they can differ, along these lines:
  - "Market success rate measures the percentage of roster files that completed processing successfully."
  - "Record success rate measures the percentage of individual records within those files that processed successfully."
  - "Because a single failing record can cause an entire file to be marked as failed, the file-level success rate is often lower than the record-level success rate."
- Only label a pipeline stage as the **root-cause bottleneck** if BOTH are true:
  - `delay_factor > 2`, and
  - there is evidence of high failure/failure-rate (e.g., high `fail_rate` or high organization failure rates).
  Otherwise, describe the stage as a **potential performance anomaly**, e.g.:
  - "The ISF_GEN stage shows a significant processing delay (486× historical average). This indicates a performance anomaly and should be investigated as a potential bottleneck, but additional stage-level failure metrics are required to confirm whether it directly contributes to observed failures."
- If you compute or reference any organization-level "failure proxy" or "failure rate", always define it explicitly:
  - "Failure proxy = Failed Files / Total Files for that organization."
  and, for a 100% proxy, say that *all observed files for that org failed*.
- When organization metrics are available, add a small **Failure Distribution** table that shows, for each top organization:
  - client identifier, number of files, number of failed files, and its contribution to total failed files (e.g., 62% of all failures).
- If `reject_rate == 0` (or is None while failures exist), add an analytical diagnostic note:
  - "The absence of recorded rejects should be verified by reviewing validation-stage logs to confirm whether rejects are occurring but being classified as failures or that reject tracking is not implemented."
  Do not assume a specific cause; only recommend verification.
- When episodic memory mentions organizations that do **not** appear in the current failure distribution, clarify that explicitly:
  - e.g., "Previously flagged organizations do not appear in the current failure distribution. This may indicate remediation of earlier issues, inactivity in the current window, or reclassification under different client identifiers."
  Do not imply a causal connection unless the organizations actually appear in the current metrics.
- If multiple markets/orgs apply, list the **top 3–5** contributors (rank by severity: high failure rate, low success, long durations, stuck count).
- If a requested metric is not present in `analytics_metrics`, output **N/A** instead of guessing.
- Keep each section compact and scannable.

Return your answer using this exact template (fill what you can; use N/A when missing):

## Market Summary
- **Market**: <State or Market>
- **Market Success Rate**: <market_success percent>
- **Record Success Rate**: <success_rate percent>
- **Threshold**: <threshold, default 90>%
- If market_success >= threshold: clearly state that the market is above threshold and skip root-cause attribution.

## Pipeline Analysis
- **Total roster files processed**: <number>

### Top organizations contributing to failures
1. <Organization Name> – Rejection rate <percent>
2. <Organization Name> – Rejection rate <percent>
3. <Organization Name> – Rejection rate <percent>
(include up to 5 if relevant)

## Pipeline Bottleneck
- **Stage**: <PIPELINE_STAGE_NAME or "Potential Performance Anomaly">
  - When you mention this stage, follow the wording guidance above: describe it as a potential performance anomaly unless both delay_factor > 2 and failures are clearly associated with this stage.

## Performance Metrics
- **Average duration**: <minutes>
- **Historical average**: <minutes>
- **Delay factor**: <multiplier>

## Record Breakdown
- **Success**: <success_rate percent>
- **Fail**: <fail_rate percent>
- **Reject**: <reject_rate percent or 0; if 0, add diagnostic insight as described above>

## Insights
- <bullet points summarizing the main issue, clearly stating whether the user's premise is confirmed or disproven>
- <bullet points identifying systemic causes only when they are supported by the metrics>
- <include a short explanation of market vs record success rate when both are present, as described above>
- <include the reject-rate diagnostic note if reject_rate == 0 or None while failures exist>

## Recommended Actions
- <operational fixes tied directly to observed evidence (e.g., high delay_factor, high organization failure_rate)>
- <data / rule corrections based on the failure distribution and failure proxy definitions>
"""


class RosterIQAgent:
    """
    Main orchestrator for the RosterIQ intelligence agent.
    It wraps the LLM with:
      - episodic memory (ChromaDB)
      - procedural memory (analytics workflows)
      - semantic memory (knowledge base)
    and exposes a simple .answer() API for the Streamlit app.
    """

    def __init__(
        self,
        roster_df: pd.DataFrame,
        market_df: pd.DataFrame,
    ):
        self.roster_df = roster_df
        self.market_df = market_df

        self.llm = _get_llm()
        self.semantic_memory = SemanticMemory()
        self.episodic_memory = EpisodicMemory()
        self.procedures = ProceduralRegistry()

    def _build_filter_context(
        self,
        state: Optional[str],
        organization: Optional[str],
        lob: Optional[str],
        pipeline_stage: Optional[str],
    ) -> str:
        parts = []
        if state:
            parts.append(f"state = {state}")
        if organization:
            parts.append(f"organization = {organization}")
        if lob:
            parts.append(f"LOB = {lob}")
        if pipeline_stage:
            parts.append(f"pipeline stage = {pipeline_stage}")
        return ", ".join(parts) if parts else "no explicit filters"

    def _run_procedures(
        self,
        roster_df: pd.DataFrame,
        state: Optional[str],
        organization: Optional[str],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Execute key procedures and anomaly detection and return:
          - raw frames for the UI
          - text summary for the LLM prompt
        """
        analytics_frames: Dict[str, Any] = {}
        # Triage stuck ROs
        stuck = triage_stuck_ros(roster_df, state=state, organization=organization)
        analytics_frames["triage_stuck_ros"] = stuck

        # Record quality audit
        quality = record_quality_audit(roster_df)
        analytics_frames["record_quality_audit"] = quality

        # Market health report
        market = market_health_report(roster_df, self.market_df, state=state)
        analytics_frames["market_health_report"] = market

        # Retry effectiveness
        retry = retry_effectiveness_analysis(roster_df)
        analytics_frames["retry_effectiveness_analysis"] = retry

        # Anomalies
        anomalies = detect_anomalies(roster_df, self.market_df)
        analytics_frames["anomalies"] = anomalies

        # Build a text summary that the LLM can use
        summary_parts: List[str] = []
        if not stuck.empty:
            org_col = "ORG_NM" if "ORG_NM" in stuck.columns else "ORG_NAME"
            summary_parts.append(
                f"{len(stuck)} roster operations are currently marked as stuck; "
                f"worst examples include states/orgs like "
                f"{stuck.get('CNT_STATE', stuck.index).head(3).tolist()} / "
                f"{stuck.get(org_col, stuck.index).head(3).tolist()}."
            )
        if not quality.empty:
            high_rej = quality[quality.get("QUALITY_REJ_RATE", 0) > 0.3]
            if not high_rej.empty:
                org_col = "ORG_NM" if "ORG_NM" in high_rej.columns else "ORG_NAME"
                summary_parts.append(
                    f"{len(high_rej)} files have rejection rate > 30%; "
                    f"top offending orgs: {high_rej.get(org_col, high_rej.index).head(5).tolist()}."
                )
        if not market.empty:
            at_risk = market[market.get("IS_MARKET_AT_RISK", False) == True]  # noqa: E712
            if not at_risk.empty:
                states = at_risk.get("MARKET", at_risk.index).dropna().unique().tolist()
                summary_parts.append(
                    f"Markets at risk based on low success and/or high rejection include: {states}."
                )
        if isinstance(anomalies, dict):
            if "low_market_success" in anomalies:
                low_mkt = anomalies["low_market_success"]
                if not low_mkt.empty:
                    states = low_mkt.get("MARKET", low_mkt.index).dropna().unique().tolist()
                    summary_parts.append(
                        f"Markets with success percent below 90%: {states}."
                    )

        analytics_summary = "\n".join(summary_parts) if summary_parts else "No critical anomalies detected."
        return analytics_frames, analytics_summary

    def _build_episodic_context(self, question: str, state: Optional[str]) -> str:
        episodes = self.episodic_memory.retrieve_similar(market=state, issue=question, n_results=3)
        if not episodes:
            return "No prior related investigations found."

        lines = []
        for ep in episodes:
            lines.append(
                f"- On {ep.get('timestamp')}, issue '{ep.get('issue')}' in market "
                f"{ep.get('market') or 'All'} focused on stage {ep.get('stage') or 'N/A'} "
                f"with organizations {ep.get('organizations') or 'N/A'}. "
                f"Summary: {ep.get('summary') or ''}"
            )
        return "\n".join(lines)

    def _maybe_call_web_search(self, question: str) -> Optional[str]:
        """
        Opportunistically call Tavily when the question looks regulatory/definition-like.
        """
        trigger_keywords = [
            "CMS",
            "Medicaid",
            "compliance",
            "regulation",
            "validation rule",
            "standards",
        ]
        if not any(k.lower() in question.lower() for k in trigger_keywords):
            return None

        try:
            results = web_search(question, max_results=3)
        except Exception:
            return None

        snippets = []
        for r in results:
            title = r.get("title") or "Result"
            snippet = r.get("content") or r.get("snippet") or ""
            snippets.append(f"- {title}: {snippet[:300]}")
        return "\n".join(snippets) if snippets else None

    def answer(
        self,
        question: str,
        state: Optional[str] = None,
        organization: Optional[str] = None,
        lob: Optional[str] = None,
        pipeline_stage: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Main entry point called by the Streamlit app.
        Returns:
          - markdown answer from the LLM
          - analytics_frames: dict of DataFrames / structures the UI may use
        """
        # --- Compute key operational metrics from the dataframe BEFORE LLM ---
        # 1) Correct state filtering first: dataset uses CNT_STATE (e.g., "TX")
        base_df = self.roster_df.copy()
        base_df.columns = base_df.columns.astype(str).str.strip()

        if state and "CNT_STATE" in base_df.columns:
            filtered_df = base_df[
                base_df["CNT_STATE"].astype(str).str.upper() == state.upper()
            ].copy()
        else:
            filtered_df = base_df.copy()

        def _num(df_: pd.DataFrame, col: str) -> pd.Series:
            """Numeric helper that returns zeros when a column is missing."""
            if col not in df_.columns:
                return pd.Series(0, index=df_.index, dtype="float64")
            return pd.to_numeric(df_[col], errors="coerce").fillna(0)

        # 2) Record metrics – driven by the market dataset schema (OVERALL_* counts)
        total_files = int(len(filtered_df))

        mdf = self.market_df.copy()
        mdf.columns = mdf.columns.astype(str).str.strip()
        if state and "MARKET" in mdf.columns:
            metrics_df = mdf[mdf["MARKET"].astype(str).str.upper() == state.upper()].copy()
        else:
            metrics_df = mdf.copy()

        total_success = float(pd.to_numeric(metrics_df["OVERALL_SCS_CNT"], errors="coerce").fillna(0).sum()) if "OVERALL_SCS_CNT" in metrics_df.columns else 0.0
        total_fail = float(pd.to_numeric(metrics_df["OVERALL_FAIL_CNT"], errors="coerce").fillna(0).sum()) if "OVERALL_FAIL_CNT" in metrics_df.columns else 0.0
        total_records = total_success + total_fail
        total_reject = 0.0

        # Data integrity check: avoid silently showing all 0%
        if total_success + total_fail + total_reject == 0:
            warnings.warn(
                "Metrics Integrity Alert: record outcome metrics missing or zero "
                "(SCS_REC_CNT/FAIL_REC_CNT/REJ_REC_CNT).",
                RuntimeWarning,
            )

        if total_records > 0:
            success_rate = round((total_success / total_records) * 100.0, 1)
            fail_rate = round((total_fail / total_records) * 100.0, 1)
            reject_rate = round((total_reject / total_records) * 100.0, 1)
        else:
            success_rate = fail_rate = reject_rate = None

        # 3) Market success rate (use market dataset SCS_PERCENT and state-filtered MARKET)
        market_success = None
        if "SCS_PERCENT" in metrics_df.columns:
            valid = pd.to_numeric(metrics_df["SCS_PERCENT"], errors="coerce").dropna()
            if not valid.empty:
                market_success = float(valid.mean())

        # 4) Organization ranking – from market dataset by CLIENT_ID
        top_orgs: List[Dict[str, Any]] = []
        if "CLIENT_ID" in metrics_df.columns and {"OVERALL_SCS_CNT", "OVERALL_FAIL_CNT"}.issubset(
            metrics_df.columns
        ):
            org_stats = (
                metrics_df.groupby("CLIENT_ID")
                .agg(
                    total_success=("OVERALL_SCS_CNT", "sum"),
                    total_fail=("OVERALL_FAIL_CNT", "sum"),
                )
            )
            denom = org_stats["total_success"] + org_stats["total_fail"]
            org_stats["failure_rate"] = (
                org_stats["total_fail"] / denom.where(denom > 0, other=pd.NA) * 100.0
            ).fillna(0.0)

            top = org_stats.sort_values("failure_rate", ascending=False).head(3)
            for client_id, row in top.iterrows():
                top_orgs.append(
                    {
                        "client_id": str(client_id),
                        "failure_rate": round(float(row["failure_rate"]), 2),
                    }
                )

        # 5) Bottleneck detection: stage duration comparison (state-filtered)
        stage_avgs: Dict[str, float] = {}
        stage_duration_cols = {
            "PRE_PROCESSING": "PRE_PROCESSING_DURATION",
            "MAPPING_APPROVAL": "MAPPING_APROVAL_DURATION",
            "ISF_GEN": "ISF_GEN_DURATION",
            "DART_GEN": "DART_GEN_DURATION",
            "DART_REVIEW": "DART_REVIEW_DURATION",
            "DART_UI_VALIDATION": "DART_UI_VALIDATION_DURATION",
            "SPS_LOAD": "SPS_LOAD_DURATION",
        }

        def _clean_duration_series(s: pd.Series) -> pd.Series:
            """
            Keep only completed/valid duration values:
            - numeric coercion
            - non-null
            - non-negative
            - drop extreme outliers that commonly occur for unfinished stages
              (use a robust 99th percentile cap).
            """
            vals = pd.to_numeric(s, errors="coerce")
            vals = vals[vals.notna()]
            vals = vals[vals >= 0]
            if len(vals) < 20:
                return vals
            cap = float(vals.quantile(0.99))
            if cap <= 0:
                return vals
            return vals[vals <= cap]

        for stage_name, dur_col in stage_duration_cols.items():
            if dur_col in filtered_df.columns:
                # Only use rows where duration is present and reasonable
                vals = _clean_duration_series(filtered_df[dur_col])
                if not vals.empty:
                    stage_avgs[stage_name] = float(vals.mean())

        bottleneck_stage = max(stage_avgs, key=stage_avgs.get) if stage_avgs else None

        # Organizations contributing most to delay at the bottleneck stage
        top_delay_orgs: List[Dict[str, Any]] = []
        if bottleneck_stage is not None:
            bottleneck_col = stage_duration_cols[bottleneck_stage]
            if {"ORG_NM", "CNT_STATE", bottleneck_col}.issubset(filtered_df.columns):
                df_delay = filtered_df.copy()
                # Clean duration values for the bottleneck stage
                df_delay[bottleneck_col] = pd.to_numeric(
                    df_delay[bottleneck_col], errors="coerce"
                )
                # Only rows where duration exists and is non-negative
                df_delay = df_delay[df_delay[bottleneck_col].notna()]
                df_delay = df_delay[df_delay[bottleneck_col] >= 0]
                # Remove extreme outliers (likely unfinished or anomalous) using 95th percentile
                if len(df_delay) >= 20:
                    thresh = float(df_delay[bottleneck_col].quantile(0.95))
                    if thresh > 0:
                        df_delay = df_delay[df_delay[bottleneck_col] < thresh]

                # Human-readable file status mapping
                status_map = {
                    9: "STOPPED",
                    45: "DART_REVIEW",
                    49: "DART_GENERATION",
                    65: "SPS_LOAD",
                    99: "RESOLVED",
                }
                if "FILE_STATUS_CD" in df_delay.columns:
                    df_delay["FILE_STATUS"] = pd.to_numeric(
                        df_delay["FILE_STATUS_CD"], errors="coerce"
                    ).map(status_map).fillna(df_delay["FILE_STATUS_CD"].astype(str))

                # For organization ranking, use median duration (more robust to residual outliers)
                grp = (
                    df_delay.groupby(["ORG_NM", "CNT_STATE"])[bottleneck_col]
                    .median()
                    .sort_values(ascending=False)
                    .head(5)
                )
                for (org_nm, cnt_state), med_delay in grp.items():
                    # Most common human-readable file status for this org/state
                    status = None
                    if "FILE_STATUS" in df_delay.columns:
                        subset = df_delay[
                            (df_delay["ORG_NM"] == org_nm)
                            & (df_delay["CNT_STATE"] == cnt_state)
                        ]["FILE_STATUS"].dropna()
                        if not subset.empty:
                            status = subset.mode().iat[0]
                    top_delay_orgs.append(
                        {
                            "organization": str(org_nm),
                            "state": str(cnt_state),
                            "latest_stage": bottleneck_stage,
                            "file_status": status,
                            "avg_duration": float(med_delay),
                        }
                    )

        # Select a high-level diagnostic workflow from procedural memory for context
        selected_workflow = self.procedures.match_workflow(question)

        analytics_metrics = {
            "state": state,
            "total_files": total_files,
            "total_records": total_records,
            "success_rate": success_rate,
            "fail_rate": fail_rate,
            "reject_rate": reject_rate,
            "market_success": market_success,
            "top_orgs": top_orgs,
            "bottleneck_stage": bottleneck_stage,
            "stage_avgs": stage_avgs,
            "top_delay_orgs": top_delay_orgs,
            "threshold": 90.0,
            "selected_workflow": selected_workflow.get("description") if selected_workflow else None,
            "workflow_steps": selected_workflow.get("steps") if selected_workflow else None,
        }

        # Run procedures and anomaly tools using the same state-filtered dataframe
        analytics_frames, analytics_summary = self._run_procedures(
            roster_df=filtered_df,
            state=state,
            organization=organization,
        )

        episodic_context = self._build_episodic_context(question, state)
        semantic_knowledge = self.semantic_memory.as_bulleted_markdown()

        web_context = self._maybe_call_web_search(question)
        if web_context:
            analytics_summary = analytics_summary + "\n\nExternal web context:\n" + web_context

        filter_context = self._build_filter_context(state, organization, lob, pipeline_stage)

        prompt = AGENT_PROMPT_TEMPLATE.format(
            question=question,
            filter_context=filter_context,
            analytics_summary=analytics_summary,
            analytics_metrics=str(analytics_metrics),
            episodic_context=episodic_context,
            semantic_knowledge=semantic_knowledge,
        )

        llm_result = self.llm.invoke(prompt)
        # langchain_openai ChatOpenAI returns a BaseMessage-like object
        response = getattr(llm_result, "content", str(llm_result))

        # Store episodic memory about this investigation
        orgs_for_memory: List[str] = []
        for org_entry in analytics_metrics.get("top_delay_orgs", []):
            name = org_entry.get("organization")
            if name:
                orgs_for_memory.append(str(name))
        for org_entry in analytics_metrics.get("top_orgs", []):
            name = org_entry.get("client_id") or org_entry.get("org")
            if name:
                orgs_for_memory.append(str(name))

        self.episodic_memory.add_episode(
            market=state,
            issue=question,
            stage=analytics_metrics.get("bottleneck_stage"),
            organizations=orgs_for_memory,
            summary=response[:500],
        )

        analytics_frames["filtered_roster"] = filtered_df
        analytics_frames["analytics_metrics"] = analytics_metrics
        return response, analytics_frames

