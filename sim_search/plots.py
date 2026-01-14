import pandas as pd
from lightweight_charts import AbstractChart, JupyterChart

from sim_search.models import Window


def create_chart_impl(chart: AbstractChart, window: Window, show_projection: bool):
    if window.data.columns.size > 1:
        if not show_projection and window.train_cutoff:
            chart.set(window.data.loc[:window.train_cutoff])
        else:
            chart.set(window.data)
    else:
        line = chart.create_line()
        data = window.data
        if not show_projection and window.train_cutoff:
            line.set(data.loc[:window.train_cutoff])
        else:
            line.set(data)
    if window.train_cutoff:
        chart.vertical_span(window.train_cutoff, color='#E8F2FD')
    chart.fit()


def window_chart(window: Window, show_projection: bool = True, width: int = 800,
                 height: int = 400) -> JupyterChart:
    """Plot jupyter chart from a WindowMatch, with an optional projection"""
    chart = JupyterChart(width=width, height=height)
    create_chart_impl(chart, window, show_projection)
    return chart


def dataframe_chart(data: pd.DataFrame | pd.Series, width: int = 800, height: int = 400) -> JupyterChart:
    """Plot jupyter chart from a DataFrame"""
    if isinstance(data, pd.Series):
        data = data.to_frame('value')
    return window_chart(Window(data=data), False, width, height)
