"""
Window functions

OPTIMIZED VERSION - Added generator support for memory efficiency.

Naming Conventions
https://learn.microsoft.com/en-us/azure/stream-analytics/stream-analytics-window-functions
- tumbling - fixed length, non overlapping buckets
- hopping  - same as above, but with overlap
- sliding - fixed length, may overlap, only when values change, irregular times
"""
from datetime import time, timedelta, datetime
from enum import IntFlag, auto
from typing import Iterator, Tuple, Optional

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


def normalize_rolling_zscore(data: PandasData, lookback: int = 20, min_periods: int = 1) -> PandasData:
    """
    Rolling Z-Score normalization using only past data (no lookahead bias).

    For each point t: z[t] = (x[t] - rolling_mean[t]) / rolling_std[t]
    where rolling stats use only data from [t-lookback+1, t].

    Parameters
    ----------
    data : PandasData
        Data to normalize (Series or DataFrame)
    lookback : int
        Number of periods for rolling window (default: 20)
    min_periods : int
        Minimum observations required for valid result (default: 1)

    Returns
    -------
    PandasData
        Z-score normalized data with no lookahead bias
    """
    rolling_mean = data.rolling(window=lookback, min_periods=min_periods).mean()
    rolling_std = data.rolling(window=lookback, min_periods=min_periods).std()
    # Avoid division by zero - replace 0 std with 1
    rolling_std = rolling_std.replace(0, 1)
    normalized = (data - rolling_mean) / rolling_std
    return normalized.fillna(0)


def normalize_window(data: PandasData, method='pct_change', **kwargs) -> PandasData:
    """
    Normalize a window using the specified method.

    Parameters
    ----------
    data : PandasData
        Window to normalize
    method : str
        Normalization method:
        - 'pct_change': Simple percentage change
        - 'log_returns': Log returns (ln(p_t / p_{t-1}))
        - 'log': Natural log of values
        - 'rolling_zscore': Rolling z-score (no lookahead bias)
    **kwargs : dict
        Additional arguments for specific methods:
        - rolling_zscore: lookback (int, default=20), min_periods (int, default=1)

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
    elif method == 'rolling_zscore':
        lookback = kwargs.get('lookback', 20)
        min_periods = kwargs.get('min_periods', 1)
        normalized = normalize_rolling_zscore(data, lookback=lookback, min_periods=min_periods)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return normalized


def partition_tumbling(data: pd.DataFrame | pd.Series | pd.Index, window_len: int = 1, horizon_len: int = 1) \
        -> pd.IntervalIndex:
    """
    Create non-overlapping (tumbling) windows.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series | pd.Index
        Input data with DatetimeIndex
    window_len : int
        Number of bars in each window
    horizon_len : int
        Number of bars needed after window for forecast horizon

    Returns
    -------
    pd.IntervalIndex
        Non-overlapping window intervals
    """
    total_window = window_len + horizon_len
    if not len(data):
        return pd.IntervalIndex(data=[], closed='both', dtype=interval_dtype(data))

    index = assertTz(data)
    intervals = []
    for i in range(0, len(data) - total_window + 1, window_len):
        left = index[i]
        right = index[i + window_len - 1]  # -1 to make right inclusive
        intervals.append((left, right))
    return pd.IntervalIndex.from_tuples(intervals, closed='both')


# =============================================================================
# GENERATOR-BASED SLIDING WINDOWS (Memory Efficient)
# =============================================================================

def iter_sliding_windows(
    data: pd.DataFrame | pd.Series | pd.Index,
    window_len: int = 30,
    step_size: int = 1,
    horizon_len: int = 1
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Generator that yields sliding window bounds one at a time.
    
    MEMORY EFFICIENT: Does NOT create a list of all windows.
    Use this for streaming/chunked processing of large datasets.
    
    Parameters
    ----------
    data : pd.DataFrame | pd.Series | pd.Index
        Input data with DatetimeIndex
    window_len : int
        Number of bars in each window (default: 30)
    step_size : int
        Number of bars to advance between windows (default: 1)
    horizon_len : int
        Number of bars needed after window for forecast horizon (default: 1)
        
    Yields
    ------
    tuple[pd.Timestamp, pd.Timestamp]
        (left, right) bounds of each window
        
    Examples
    --------
    >>> # Stream windows without loading all into memory
    >>> for left, right in iter_sliding_windows(df, window_len=60, step_size=5):
    ...     process_window(df.loc[left:right])
    """
    total_needed = window_len + horizon_len
    if not len(data):
        return
    
    index = assertTz(data)
    step_size = max(1, step_size)
    n_data = len(data)
    
    for i in range(0, n_data - total_needed + 1, step_size):
        left = index[i]
        right = index[i + window_len - 1]
        yield (left, right)


def iter_sliding_windows_indexed(
    data: pd.DataFrame | pd.Series | pd.Index,
    window_len: int = 30,
    step_size: int = 1,
    horizon_len: int = 1
) -> Iterator[Tuple[int, int, int]]:
    """
    Generator that yields sliding window INTEGER indices.
    
    More efficient than timestamp-based when you just need array slicing.
    
    Parameters
    ----------
    data : pd.DataFrame | pd.Series | pd.Index
        Input data with DatetimeIndex (used only for length)
    window_len : int
        Number of bars in each window
    step_size : int
        Number of bars to advance between windows
    horizon_len : int
        Number of bars needed after window for forecast horizon
        
    Yields
    ------
    tuple[int, int, int]
        (start_idx, end_idx, horizon_end_idx) for each window
        - start_idx: first index of window (inclusive)
        - end_idx: last index of window (inclusive)  
        - horizon_end_idx: last index of horizon (inclusive)
        
    Examples
    --------
    >>> # Efficient array slicing without timestamp lookups
    >>> for start, end, horizon_end in iter_sliding_windows_indexed(df, 60, 5, 20):
    ...     x = normalized_data[start:end+1]
    ...     y = normalized_data[end+1:horizon_end+1]
    """
    total_needed = window_len + horizon_len
    n_data = len(data)
    step_size = max(1, step_size)
    
    for i in range(0, n_data - total_needed + 1, step_size):
        start_idx = i
        end_idx = i + window_len - 1
        horizon_end_idx = i + window_len + horizon_len - 1
        yield (start_idx, end_idx, horizon_end_idx)


def count_sliding_windows(
    n_data: int,
    window_len: int = 30,
    step_size: int = 1,
    horizon_len: int = 1
) -> int:
    """
    Calculate the number of sliding windows without creating them.
    
    Useful for pre-allocating arrays or progress bars.
    
    Parameters
    ----------
    n_data : int
        Length of the data
    window_len : int
        Number of bars in each window
    step_size : int
        Number of bars to advance between windows
    horizon_len : int
        Number of bars needed after window for forecast horizon
        
    Returns
    -------
    int
        Number of windows that will be created
    """
    total_needed = window_len + horizon_len
    step_size = max(1, step_size)
    
    if n_data < total_needed:
        return 0
    
    return (n_data - total_needed) // step_size + 1


# =============================================================================
# ORIGINAL PARTITION SLIDING (Backward Compatible, with Optimizations)
# =============================================================================

def partition_sliding(
    data: pd.DataFrame | pd.Series | pd.Index,
    window_len: int = 30,
    step_size: int = 1,
    horizon_len: int = 1,
    max_windows: Optional[int] = None
) -> pd.IntervalIndex:
    """
    Create overlapping (sliding) windows with configurable step size.
    
    NOTE: For very large datasets (>1M bars), consider using iter_sliding_windows()
    or iter_sliding_windows_indexed() generators to avoid memory issues.

    Parameters
    ----------
    data : pd.DataFrame | pd.Series | pd.Index
        Input data with DatetimeIndex
    window_len : int
        Number of bars in each window (default: 30)
    step_size : int
        Number of bars to advance between windows (default: 1 = fully overlapping)
    horizon_len : int
        Number of bars needed after window for forecast horizon (default: 1)
    max_windows : int, optional
        Maximum number of windows to create (default: None = all).
        Use this to limit memory for very large datasets.

    Returns
    -------
    pd.IntervalIndex
        Overlapping window intervals

    Examples
    --------
    >>> # 100 bars, window=30, step=5, horizon=10
    >>> # Creates ~13 windows: (0-29), (5-34), (10-39), ...
    >>> partition_sliding(df, window_len=30, step_size=5, horizon_len=10)
    
    >>> # Limit to first 1000 windows for memory efficiency
    >>> partition_sliding(df, window_len=60, max_windows=1000)
    """
    total_needed = window_len + horizon_len
    if not len(data):
        return pd.IntervalIndex(data=[], closed='both', dtype=interval_dtype(data))

    index = assertTz(data)
    step_size = max(1, step_size)
    
    # Pre-calculate number of windows for efficient list allocation
    n_windows = count_sliding_windows(len(data), window_len, step_size, horizon_len)
    
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    
    # Log warning for large window counts
    if n_windows > 1_000_000:
        logger.warning(
            f"Creating {n_windows:,} windows will use significant memory. "
            f"Consider using iter_sliding_windows() generator or increasing step_size."
        )
    
    # Build intervals list
    intervals = []
    count = 0
    for i in range(0, len(data) - total_needed + 1, step_size):
        if max_windows is not None and count >= max_windows:
            break
            
        left = index[i]
        right = index[i + window_len - 1]
        intervals.append((left, right))
        count += 1

    return pd.IntervalIndex.from_tuples(intervals, closed='both')


def partition_sliding_chunked(
    data: pd.DataFrame | pd.Series | pd.Index,
    window_len: int = 30,
    step_size: int = 1,
    horizon_len: int = 1,
    chunk_size: int = 10000
) -> Iterator[pd.IntervalIndex]:
    """
    Create sliding windows in chunks to manage memory.
    
    Yields chunks of IntervalIndex objects for batch processing.
    
    Parameters
    ----------
    data : pd.DataFrame | pd.Series | pd.Index
        Input data with DatetimeIndex
    window_len : int
        Number of bars in each window
    step_size : int
        Number of bars to advance between windows
    horizon_len : int
        Number of bars needed after window for forecast horizon
    chunk_size : int
        Number of windows per chunk (default: 10000)
        
    Yields
    ------
    pd.IntervalIndex
        Chunk of window intervals
        
    Examples
    --------
    >>> # Process 10 years of data in manageable chunks
    >>> for chunk_intervals in partition_sliding_chunked(df, chunk_size=5000):
    ...     x_chunk, y_chunk, labels_chunk = prepare_panel_data(df, chunk_intervals)
    ...     process_chunk(x_chunk, y_chunk)
    """
    total_needed = window_len + horizon_len
    if not len(data):
        return
    
    index = assertTz(data)
    step_size = max(1, step_size)
    
    chunk_intervals = []
    
    for i in range(0, len(data) - total_needed + 1, step_size):
        left = index[i]
        right = index[i + window_len - 1]
        chunk_intervals.append((left, right))
        
        if len(chunk_intervals) >= chunk_size:
            yield pd.IntervalIndex.from_tuples(chunk_intervals, closed='both')
            chunk_intervals = []
    
    # Yield remaining intervals
    if chunk_intervals:
        yield pd.IntervalIndex.from_tuples(chunk_intervals, closed='both')
