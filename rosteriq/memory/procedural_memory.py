import os
from typing import Any, Callable, Dict, List, Optional

import yaml

from procedures.triage_stuck_ros import triage_stuck_ros
from procedures.record_quality_audit import record_quality_audit
from procedures.market_health_report import market_health_report
from procedures.retry_effectiveness_analysis import retry_effectiveness_analysis


ProcedureFunc = Callable[..., Any]


class ProceduralRegistry:
    """
    Registry of reusable analytical procedures that the agent can invoke,
    plus access to high-level diagnostic workflows stored in YAML.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, ProcedureFunc] = {
            "triage_stuck_ros": triage_stuck_ros,
            "record_quality_audit": record_quality_audit,
            "market_health_report": market_health_report,
            "retry_effectiveness_analysis": retry_effectiveness_analysis,
        }
        self._workflows: Dict[str, Dict[str, Any]] = self._load_workflows()

    def _load_workflows(self) -> Dict[str, Dict[str, Any]]:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base_dir, "memory", "procedural_memory.yaml")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    @property
    def available_procedures(self) -> List[str]:
        return sorted(self._registry.keys())

    @property
    def available_workflows(self) -> List[str]:
        return sorted(self._workflows.keys())

    def get(self, name: str) -> ProcedureFunc:
        if name not in self._registry:
            raise KeyError(f"Unknown procedure: {name}")
        return self._registry[name]

    def get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        return self._workflows.get(name)

    def match_workflow(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Very lightweight routing from user question to a diagnostic workflow.
        """
        q = question.lower()
        if any(k in q for k in ["stuck", "stalled", "hung"]):
            return self._workflows.get("stuck_operations")
        if any(k in q for k in ["bottleneck", "slow", "delay", "latency"]):
            return self._workflows.get("pipeline_bottleneck")
        if any(k in q for k in ["success rate", "failure rate", "at risk", "market"]):
            return self._workflows.get("market_failure")
        return None

    def describe(self) -> str:
        """
        Human-readable description of procedures for prompting the LLM.
        """
        return (
            "triage_stuck_ros: Find all rows where IS_STUCK = True and rank by duration and key health fields.\n"
            "record_quality_audit: Evaluate record quality metrics and flag files with low success.\n"
            "market_health_report: Join roster and market-level metrics by state/market to highlight at-risk markets.\n"
            "retry_effectiveness_analysis: Compare initial vs retry runs to determine whether retries improve success."
        )

