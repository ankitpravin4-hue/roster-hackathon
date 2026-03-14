from typing import Optional

import pandas as pd
from plotly.graph_objs import Figure

from tools.visualization_tool import market_success_trend


def build_market_success_trend(market_df: pd.DataFrame, state: Optional[str] = None) -> Figure:
    return market_success_trend(market_df, state=state)

