import os
from datetime import time, datetime
from typing import cast, overload
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
from loguru import logger
from pandas.api.typing import DataFrameGroupBy, SeriesGroupBy

from . import config
from .core import dt_idx

CME = xcals.get_calendar("CME")
FUTURES = xcals.get_calendar("us_futures")
NYSE = xcals.get_calendar("NYSE")
NYSE_REGULAR_CLOSE = time(16)
NYT = ZoneInfo('America/New_York')
TZ_DEFAULT: ZoneInfo = ZoneInfo('UTC')


def default_tz() -> ZoneInfo:
    return TZ_DEFAULT


def set_default_tz(tz: str):
    global TZ_DEFAULT
    logger.info(f'Set timezone to {tz}')
    TZ_DEFAULT = ZoneInfo(tz)


def assertNoTz(data: pd.DataFrame | pd.Series | pd.Index) -> pd.DatetimeIndex:
    index = dt_idx(data)
    if index.tz is not None:
        raise ValueError("Data must not have a timezone")
    return index


def assertTz(data: pd.DataFrame | pd.Series | pd.Index) -> pd.DatetimeIndex:
    index = dt_idx(data)
    if index.tz is None:
        raise ValueError("Data must have a timezone")
    return index


def assertDefaultTz(data: pd.DataFrame | pd.Series | pd.Index) -> pd.DatetimeIndex:
    index = dt_idx(data)
    if index.tz is None:
        raise ValueError("Data must have a timezone")
    if index.tz != default_tz():
        raise ValueError(f"Data must have timezone {default_tz()}")
    return index


def interval_dtype(data: pd.DataFrame | pd.Series | pd.DatetimeIndex, closed='both'):
    pd.IntervalDtype(subtype=dt_idx(data).dtype, closed=closed)


@overload
def group_by_session(data: pd.DataFrame) -> DataFrameGroupBy:
    pass


@overload
def group_by_session(data: pd.Series) -> SeriesGroupBy:
    pass


def group_by_session(data: pd.DataFrame | pd.Series) -> DataFrameGroupBy | SeriesGroupBy:
    index = dt_idx(data)
    if index.tz is None:
        index = index.tz_localize(TZ_DEFAULT)
    # If you pass a timezone-naive index to the calendar, it will be localized to UTC. Make sure to pass a tz.
    sessions = FUTURES.minutes_to_sessions(index)
    return data.groupby(sessions)


def select_dayofweek(data: pd.DataFrame | pd.Series | pd.Index, dayofweek: int) -> list[pd.DataFrame]:
    """
    Return DataFrame for each session occurring on dayofweek (0=Monday...6=Sunday)
    """
    sessions = group_by_session(data)
    return [grp for key, grp in sessions if key.dayofweek == dayofweek]


# def select_dayofweek_index(data: pd.DataFrame | pd.Series | pd.Index, dayofweek: int) -> pd.IntervalIndex:
#     """
#     Return IntervalIndex with interval for each session occurring on dayofweek (0=Monday...6=Sunday)
#     """
#     sessions = group_by_session(data)
#     results = [(grp.index.min(), grp.index.max()) for key, grp in sessions if key.dayofweek == dayofweek]
#     return pd.IntervalIndex.from_tuples(results, closed='both', dtype=pd.IntervalDtype(subtype='datetime64[s, UTC]'))
#

def get_session(data: pd.DataFrame | pd.Series | pd.Index) -> pd.DatetimeIndex:
    if isinstance(data, pd.Index):
        data = data.to_series()
    return FUTURES.minutes_to_sessions(data.index)


def get_dayofweek_mask(data: pd.DataFrame | pd.Series, dayofweek: int) -> pd.Series:
    """
    Get bool mask filtered for day of week, 0=Monday, 6=Sunday.
    Use with df.loc[mask] to filter.
    """
    session = get_session(data.index)
    mask = session.dayofweek == dayofweek
    return pd.Series(data=mask, index=data.index)


def get_agg_map(data: pd.DataFrame | pd.Series) -> dict:
    agg_map = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    is_series = isinstance(data, pd.Series)
    if is_series:
        agg_map = {data.name: agg_map[cast(str, data.name)]}
    else:
        agg_map = {k: v for k, v in agg_map.items() if k in data.columns}
    return agg_map


def resample(data: pd.DataFrame | pd.Series, freq: str, origin: str = '18:00') -> pd.DataFrame:
    """Resample a DataFrame to a new frequency. Origin is interpreted in default timezone."""
    if len(data) == 0:
        return data

    # Origin is assumed default timezone. Convert to data timezone if different.
    start = data.index[0]
    ts_origin = datetime.combine(start.date(), time.fromisoformat(origin))
    if start.tz:
        ts_origin = ts_origin.replace(tzinfo=default_tz()).astimezone(start.tz)

    agg_map = get_agg_map(data)
    data = data.resample(freq, origin=ts_origin).agg(agg_map)
    data.dropna(inplace=True)
    # if is_series:
    #     data = data.squeeze()
    return data
