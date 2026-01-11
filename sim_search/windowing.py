"""
Window functions

Naming Conventions
https://learn.microsoft.com/en-us/azure/stream-analytics/stream-analytics-window-functions
- tumbling - fixed length, non overlapping buckets
- hopping  - same as above, but with overlap
- sliding - fixed length, may overlap, only when values change, irregular times
"""
from datetime import time, timedelta, datetime
from enum import IntFlag, auto

import numpy as np
import pandas as pd
from loguru import logger

from .core import PandasData
from .times import NYSE, FUTURES, NYSE_REGULAR_CLOSE, NYT, default_tz, assertTz, interval_dtype

# Allowed distance from expected start/end of windows and actual.
WINDOW_TOLERANCE = pd.Timedelta(minutes=15)


class WindowFilters(IntFlag):
    """
    Options to control window partition behavior.
    MISALIGNED - skip any windows which bounds are misaligned to larger than tolerance.
    SKIP_IRREGULAR - skip any windows that include irregular length days.
    SKIP_WEEKENDS - skip any windows that overlap the weekend.
    SKIP_HOLIDAYS - skip any windows that overlap holidays.
    """
    NONE = 0
    SKIP_MISALIGNED = auto()
    SKIP_IRREGULAR = auto()
    SKIP_WEEKENDS = auto()
    SKIP_HOLIDAYS = auto()


WINDOW_FILTER_DEFAULT = WindowFilters.SKIP_IRREGULAR | WindowFilters.SKIP_WEEKENDS | WindowFilters.SKIP_HOLIDAYS \
                        | WindowFilters.SKIP_MISALIGNED


def partition_time_anchored(data: pd.DataFrame | pd.Series | pd.Index, start: time, end: time, extend_sessions: int = 0,
                            exclude_filters=WINDOW_FILTER_DEFAULT) -> pd.IntervalIndex:
    """
    Partition the given data into windows from start to end time. Extend sessions will add an addition session gap to
    that already implied by the start and end time. Exclude any windows which match the exclude filters.
    For days offset, 0 means current day, 1 means next day, etc.
    Because this interacts with an external calendar, the input must be tz-aware and in the default timezone.
    Start and end time will be interpreted in default timezone.

    """
    if not len(data):
        return pd.IntervalIndex(data=[], closed='both', dtype=interval_dtype(data))
    index = assertTz(data)

    # Filter to only valid trading minutes to avoid errors
    minutes_range = FUTURES.minutes_in_range(index[0], index[-1])
    valid_mask = index.isin(minutes_range)
    valid_idx = index[valid_mask]
    if valid_idx.empty:
        return pd.IntervalIndex(data=[], closed='both', dtype=interval_dtype(data))

    # If start is greater than end, then this is an overnight session.
    if start > end:
        extend_sessions += 1
    windows = []
    for window_start_date in np.unique(valid_idx.date):
        window_start = pd.Timestamp.combine(window_start_date, start).tz_localize(default_tz())
        window_end_date = window_start_date + timedelta(days=extend_sessions)
        window_end = pd.Timestamp.combine(window_end_date, end).tz_localize(default_tz())
        if exclude_filters and _should_exclude_window(window_start, window_end, exclude_filters):
            logger.info(f'Excluding {window_start}-{window_end}')
            continue
        window: pd.DatetimeIndex = valid_idx[valid_idx.slice_indexer(window_start, window_end)]
        if not window.empty and _within_tolerance(window, window_start, window_end, exclude_filters):
            windows.append((window.min(), window.max()))
    return pd.IntervalIndex.from_tuples(windows, closed='both')


def _within_tolerance(window_idx: pd.DatetimeIndex, expected_start: datetime, expected_end: datetime,
                      filters: WindowFilters) -> bool:
    if WindowFilters.SKIP_MISALIGNED not in filters:
        return True
    return (abs(window_idx[0] - expected_start) < WINDOW_TOLERANCE and
            abs(window_idx[-1] - expected_end) < WINDOW_TOLERANCE)


def _should_exclude_window(start: pd.Timestamp, end: pd.Timestamp, filters: WindowFilters) -> bool:
    """
    Check if a window should be excluded based on the given filters.
    Returns True if the window should be excluded, False otherwise
    """
    # Get all dates covered by this window
    # Both sides inclusive by default
    dates_in_window = pd.date_range(start.date(), end.date(), freq='D')
    for date in dates_in_window:
        # If the range includes Saturday it crosses the weekend, Sunday sessions don't count.
        if WindowFilters.SKIP_WEEKENDS in filters and date.dayofweek == 5:
            return True

        # Check holidays using NYSE (requires timezone-naive date)
        date_naive = date.tz_localize(None)
        if WindowFilters.SKIP_HOLIDAYS in filters and not NYSE.is_session(date_naive):
            return True

        # Check irregular days (early closes) using NYSE
        if WindowFilters.SKIP_IRREGULAR in filters and _is_irregular(date_naive):
            return True

    return False


def _is_irregular(date: pd.Timestamp) -> bool:
    """Check if date is an irregular trading day (early close)"""
    if NYSE.is_session(date) and date in NYSE.schedule.index:
        session_close = NYSE.schedule.at[date, 'close'].tz_convert(NYT)
        return session_close.time() < NYSE_REGULAR_CLOSE
    return False


def normalize_window(data: PandasData, method='pct_change') -> PandasData:
    """
    Normalize a window using percentage change.

    Parameters
    ----------
    data : PandasData
        Window to normalize
    method : str
        Normalization method ('pct_change' or 'log_returns')

    Returns
    -------
    PandasData
        Normalized window
    """
    if method == 'pct_change':
        normalized = data.pct_change().fillna(0)
    elif method == 'log_returns':
        normalized = np.log(data / data.shift(1))
        normalized = pd.Series(np.nan_to_num(normalized), index=data.index)
    elif method == 'log':
        normalized = pd.Series(np.log(data), index=data.index)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normalized


def partition_tumbling(data: pd.DataFrame | pd.Series | pd.Index, window_len: int = 1, horizon_len: int = 1) \
        -> pd.IntervalIndex:
    total_window = window_len + horizon_len
    if not len(data):
        return pd.IntervalIndex(data=[], closed='both', dtype=interval_dtype(data))

    index = assertTz(data)
    intervals = []
    for i in range(0, len(data) - total_window + 1, window_len):
        left = index[i]
        right = index[i + window_len]
        intervals.append((left, right))
    return pd.IntervalIndex.from_tuples(intervals, closed='both')
