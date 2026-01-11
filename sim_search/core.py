import pandas as pd
from typing import cast

PandasData = pd.DataFrame | pd.Series


def dt_idx(data: pd.DataFrame | pd.Series | pd.Index) -> pd.DatetimeIndex:
    if isinstance(data, pd.DatetimeIndex):
        return data
    idx = data if isinstance(data, pd.Index) else data.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, but index was {type(idx).__name__}")
    return cast(pd.DatetimeIndex, idx)
