import pandas as pd
from plotly.graph_objs import Figure

from tools.visualization_tool import record_quality_breakdown


def build_record_quality_chart(roster_df: pd.DataFrame) -> Figure:
    return record_quality_breakdown(roster_df)

