import json
import os
from typing import Any, Dict


class SemanticMemory:
    """
    Semantic memory backed by a JSON knowledge file.
    Provides definitions for pipeline stages, metrics and SLA thresholds.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base_dir, "memory", "semantic_memory.json")
        self.data: Dict[str, Any] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)

    def get_stage_description(self, stage_name: str) -> str:
        stages = self.data.get("pipeline_stages", {})
        return stages.get(stage_name, "")

    def get_metric_definition(self, metric_name: str) -> str:
        metrics = self.data.get("metrics", {})
        return metrics.get(metric_name, "")

    def get_sla_threshold(self) -> float:
        sla = self.data.get("sla", {})
        return float(sla.get("market_success_threshold", 90.0))

    def as_bulleted_markdown(self) -> str:
        """
        Render the knowledge base as markdown that can be injected into prompts.
        """
        lines = ["Semantic knowledge about the roster pipeline:"]

        for stage, desc in self.data.get("pipeline_stages", {}).items():
            lines.append(f"- **Stage – {stage}**: {desc}")
        for metric, desc in self.data.get("metrics", {}).items():
            lines.append(f"- **Metric – {metric}**: {desc}")

        if "sla" in self.data and "market_success_threshold" in self.data["sla"]:
            lines.append(
                f"- **SLA – Market success threshold**: "
                f"{self.data['sla']['market_success_threshold']}% minimum success rate."
            )

        return "\n".join(lines)


