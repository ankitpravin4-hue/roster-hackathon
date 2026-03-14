import pandas as pd
from plotly.graph_objs import Figure

from tools.visualization_tool import pipeline_health_heatmap


def build_pipeline_heatmap(roster_df: pd.DataFrame) -> Figure:
    return pipeline_health_heatmap(roster_df)

