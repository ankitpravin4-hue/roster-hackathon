import pandas as pd
from plotly.graph_objs import Figure

from tools.visualization_tool import retry_effectiveness_chart


def build_retry_lift_chart(summary_df: pd.DataFrame) -> Figure:
    return retry_effectiveness_chart(summary_df)

