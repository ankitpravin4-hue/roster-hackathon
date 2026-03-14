## RosterIQ – Memory Driven Provider Roster Intelligence Agent

## Problem

Healthcare provider roster pipelines process thousands of files across multiple stages 
(ingestion, validation, mapping, DART generation, SPS load, etc.). 

When failures occur, operations teams struggle to:

- Identify **stuck roster operations**
- Understand **why certain markets have low success rates**
- Trace issues back to **specific pipeline stages or organizations**

Traditional monitoring dashboards show metrics but **do not provide root cause analysis**.

## Solution

RosterIQ is a **memory-driven AI operations copilot** that analyzes roster pipeline health, 
connects operational data with market performance, and generates root-cause explanations 
using a combination of:

- Episodic memory
- Procedural analytical workflows
- Semantic domain knowledge

RosterIQ is a hackathon-ready AI agent that helps operations, network, and analytics teams understand and debug provider roster pipelines. It combines Streamlit, LangChain, Pandas, DuckDB, ChromaDB, Plotly, and Tavily web search into a single interactive experience.

### Features

- **Natural language questions** about roster pipeline health, stuck operations, quality issues, and market performance.
- **Episodic memory (ChromaDB)** that remembers past investigations and surfaces them when relevant.
- **Procedural memory** as reusable workflows (`triage_stuck_ros`, `record_quality_audit`, `market_health_report`, `retry_effectiveness_analysis`).
- **Semantic memory** explaining pipeline stages, health flags, error metrics, and LOB/source system concepts.
- **Plotly dashboards** for pipeline health, record quality, market success trends, retry effectiveness, and stuck RO tracking.
- **Tavily web search** for CMS/Medicaid rules, validation standards, and external context.
- **Proactive anomaly detection** for high rejection rates, stuck operations, long-running stages, and low market success.

### Project Structure

```text
rosteriq/
├── app.py                  # Streamlit entrypoint
├── agent.py                # RosterIQAgent orchestrator (LLM + memories + tools)
├── memory/
│   ├── episodic_memory.py  # ChromaDB-backed episodic memory
│   ├── procedural_memory.py# Registry of analytical workflows
│   └── semantic_memory.py  # Static knowledge base about pipeline concepts
├── tools/
│   ├── data_query_tool.py  # Filtering and DuckDB SQL
│   ├── visualization_tool.py # Plotly chart builders
│   ├── web_search_tool.py  # Tavily search integration
│   └── anomaly_detection.py# Rejection / duration / market anomaly detection
├── procedures/
│   ├── triage_stuck_ros.py           # IS_STUCK triage + ranking
│   ├── record_quality_audit.py       # Record quality metrics
│   ├── market_health_report.py       # Join roster + market metrics by state
│   └── retry_effectiveness_analysis.py # RUN_NO=1 vs RUN_NO>1 success lift
├── dashboards/
│   ├── pipeline_heatmap.py   # Org vs Stage heatmap wrapper
│   ├── record_quality_chart.py
│   ├── market_success_trend.py
│   └── retry_lift_chart.py
├── data/
│   └── roster_processing_details.csv
│   └── aggregated_operational_metrics.csv
├── utils/
│   └── helpers.py            # Data loading, filtering, and DuckDB helpers
├── requirements.txt
└── README.md
```

### Data Requirements

Place the following CSVs in the `data/` folder:

- `roster_processing_details.csv`
- `aggregated_operational_metrics.csv`

Recommended (but not strictly required) columns:

- **Common dimensions**: `STATE`, `ORG_NAME`, `LOB`, `PIPELINE_STAGE`, `RUN_NO`
- **Status / health**: `IS_STUCK`, `HEALTH_FLAG`, `STAGE_DURATION` or `TOTAL_DURATION`
- **Record metrics**: `TOT_REC_CNT`, `SCS_REC_CNT` or `SCS_CNT`, `FAIL_REC_CNT`, `SKIP_REC_CNT`, `REJ_REC_CNT`, `SCS_PCT` or `SCS_PERCENT`
- **Market metrics** (in `aggregated_operational_metrics.csv`): `STATE`, `MONTH`, `SCS_PERCENT` (or `SCS_PCT`)

The code defensively handles missing columns where possible (for example, computing success percent if only counts are provided).

### Setup

1. **Create and activate a virtual environment** (optional but recommended):

```bash
cd rosteriq-hackathon/rosteriq
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Add your data**:

- Copy `roster_processing_details.csv` and `aggregated_operational_metrics.csv` into the `data/` directory.

4. **Configure API keys**:

Set the following environment variables (e.g., in your shell profile or a `.env` you load manually):

- `OPENROUTER_API_KEY` – used by LangChain's `ChatOpenAI` client with the OpenRouter base URL.
- `OPENROUTER_BASE_URL` – optional; defaults to `https://openrouter.ai/api/v1`.
- Alternatively, you can use `OPENAI_API_KEY` if routing through OpenAI-compatible endpoints.
- `ROSTERIQ_MODEL_NAME` – optional; default is `gpt-4.1-mini`.
- `TAVILY_API_KEY` – required for web search (Tavily).

Example (macOS / Linux):

```bash
export OPENROUTER_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."
```

### Running the App

From the `rosteriq/` directory:

```bash
streamlit run app.py
```

This launches the Streamlit UI with:

- **Ask AI Agent** – free-form natural language interface to the RosterIQAgent.
- **Pipeline Dashboard** – pipeline health heatmap and stuck-RO tracker.
- **Record Quality Dashboard** – stacked bar charts for SCS/FAIL/SKIP/REJ by organization.
- **Market Performance** – success trend over time with 95% threshold and retry effectiveness.
- **Operational Reports** – generated state/org-specific market health reports.

### Architecture Overview

- **Agent (`agent.py`)**
  - Wraps a LangChain `LLMChain` with:
    - `EpisodicMemory` (ChromaDB collection) – stores `{timestamp, query, analysis, result_summary}` per question.
    - `ProceduralRegistry` – registry over Python functions that implement analytics workflows.
    - `SemanticMemory` – static knowledge base describing pipeline stages, flags, metrics, and LOB/source systems.
  - On each query:
    - Applies filters (state / org / LOB / stage).
    - Runs key procedures (stuck triage, quality audit, market health, retry effectiveness).
    - Runs anomaly detection.
    - Optionally calls Tavily web search for regulatory/definition-heavy questions.
    - Retrieves related episodic memories.
    - Prompts the LLM with all of the above and returns a Markdown answer.
    - Persists the new investigation back into episodic memory.

- **Memories (`memory/`)**
  - `episodic_memory.py`: Uses a persistent ChromaDB client with a single collection; documents are investigation summaries, metadata includes timestamps and queries; similarity search pulls prior related analyses.
  - `procedural_memory.py`: Maps names like `triage_stuck_ros` to the corresponding analytics functions to make them discoverable by the agent.
  - `semantic_memory.py`: Hard-coded semantic facts rendered as Markdown and injected into prompts.

- **Tools (`tools/`)**
  - `data_query_tool.py`: Simplified data filtering and ad-hoc DuckDB SQL queries across both dataframes.
  - `visualization_tool.py`: Plotly chart builders (heatmap, stacked bars, line chart with threshold, retry bar chart, stuck tracker).
  - `web_search_tool.py`: Tavily client wrapper for external web search.
  - `anomaly_detection.py`: Core anomaly detectors (high rejection, stuck ROs, long durations, low market success).

- **Dashboards (`dashboards/`)**
  - Lightweight wrappers that expose Plotly figures to the Streamlit app, decoupling visualization logic from UI wiring.

- **Streamlit UI (`app.py`)**
  - Manages CSV loading, sidebar filters, tabbed layout, and calling `RosterIQAgent.answer()`.
  - Displays agent responses alongside key dataframes and charts for transparency.
  - Includes a Markdown-based report generator for "Texas Market Health Report"-style outputs.

### Demo Flow Ideas

- Ask: **"Which roster operations are stuck?"** and drill into the stuck tracker in the Pipeline tab.
- Ask: **"Which organization has highest rejection rate?"** and review the Record Quality dashboard.
- Ask: **"Why is California market success rate dropping?"** while filtering `STATE = CA` in the sidebar; cross-check the Market Performance tab.
- Ask: **"Show me files with rejection rate > 30%"** and review the quality audit table.
- Ask: **"What pipeline stage is the bottleneck?"** and inspect pipeline health plus the report generator output.

This project is intentionally modular so you can:

- Swap the LLM provider/model via environment variables.
- Drop in new procedures and register them in `ProceduralRegistry`.
- Extend the semantic memory with more domain facts.
- Add new dashboards or tools without touching the core agent orchestration.

## Hackathon Bonus Features

### Proactive Monitoring
RosterIQ automatically scans both operational datasets and surfaces alerts for:

- Markets trending below SLA success thresholds
- Newly stuck roster operations
- Pipeline stages showing abnormal processing delays

### Root Cause Chaining
The agent automatically traces market performance issues through the pipeline:

Market ↓  
Pipeline Stage ↓  
Organizations ↓  
Likely Operational Cause

This helps teams understand **why market success is dropping** instead of just showing metrics.

### Memory-Driven AI Reasoning

RosterIQ uses three complementary memory systems:

| Memory Type | Purpose |
|-------------|--------|
| Episodic Memory | Stores past investigations and retrieves them when similar issues appear |
| Procedural Memory | Encodes analytical workflows like stuck-RO triage and retry analysis |
| Semantic Memory | Provides domain knowledge about pipeline stages, metrics, and flags |

System Architexture Diagram
## System Architecture

User Question
      ↓
Streamlit UI
      ↓
RosterIQ Agent (LangChain)
      ↓
┌───────────────┬───────────────┬───────────────┐
│ Episodic      │ Procedural    │ Semantic      │
│ Memory        │ Workflows     │ Knowledge     │
└───────────────┴───────────────┴───────────────┘
      ↓
Analytics Tools (Pandas + DuckDB)
      ↓
Visualizations (Plotly)
      ↓
Operational Insights