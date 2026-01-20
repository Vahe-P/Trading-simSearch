"""
Builder for constructing WindowCollection from raw data.

This is the main entry point for creating properly enriched window data.
It handles:
- Window partitioning
- Normalization
- Volatility computation
- Calendar enrichment

Usage:
    builder = WindowCollectionBuilder(df)
    collection = (builder
        .with_time_anchored_windows(start=time(8), end=time(9,30))
        .with_normalization('log_returns')
        .with_volatility('garman_klass')
        .with_calendar_events()
        .build())
"""

from datetime import time
from typing import Optional, List
import numpy as np
import pandas as pd
from loguru import logger

from .datastructures import WindowData, WindowCollection
from .windowing import partition_time_anchored, partition_sliding, normalize_window
from .volatility import (
    compute_all_window_volatilities,
    compute_regime_thresholds,
    classify_all_regimes,
)
from .calendar_events import enrich_collection_with_calendar
from .config import SLIDING_WINDOW_MODE


class WindowCollectionBuilder:
    """
    Fluent builder for creating enriched WindowCollection objects.
    
    Example
    -------
    >>> builder = WindowCollectionBuilder(df)
    >>> collection = (builder
    ...     .with_time_anchored_windows(time(8), time(9, 30), extend_sessions=1)
    ...     .with_horizon(20)
    ...     .with_normalization('log_returns')
    ...     .with_volatility('garman_klass')
    ...     .with_calendar_events()
    ...     .build())
    >>> 
    >>> train, test = collection.split_train_test()
    """
    
    def __init__(self, df: pd.DataFrame, feature_col: str = 'close'):
        """
        Initialize builder with OHLC data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Market data with DatetimeIndex and OHLC columns
        feature_col : str
            Column to use for pattern matching (default: 'close')
        """
        self.df = df
        self.feature_col = feature_col
        
        # Configuration
        self.intervals: Optional[pd.IntervalIndex] = None
        self.horizon_len: int = 20
        self.norm_method: str = 'log_returns'
        self.vol_method: Optional[str] = None
        self.include_calendar: bool = False
        
    def with_time_anchored_windows(self, 
                                    start: time, 
                                    end: time, 
                                    extend_sessions: int = 0) -> 'WindowCollectionBuilder':
        """
        Use time-anchored windows (e.g., overnight session 8pm-9:30am).
        
        Parameters
        ----------
        start : time
            Window start time (in local timezone)
        end : time
            Window end time
        extend_sessions : int
            Number of sessions to extend (0 = same day, 1 = next day)
        """
        self.intervals = partition_time_anchored(
            self.df, start, end, extend_sessions
        )
        logger.info(f"Created {len(self.intervals)} time-anchored windows")
        return self
    
    def with_sliding_windows(self,
                             window_len: int = 60,
                             step_size: Optional[int] = None,
                             max_windows: Optional[int] = None) -> 'WindowCollectionBuilder':
        """
        Use sliding windows with configurable step size.
        
        The step_size is automatically determined by SLIDING_WINDOW_MODE config:
        - True (Sliding): step_size = 1 (window for every timestamp)
        - False (Anchored): step_size = bars_per_day (windows only at market open)
        
        Parameters
        ----------
        window_len : int
            Number of bars per window
        step_size : int, optional
            Bars to advance between windows. If None, determined by SLIDING_WINDOW_MODE.
            If provided, overrides the config setting.
        max_windows : int, optional
            Maximum windows to create
        """
        # Determine step_size based on config if not explicitly provided
        if step_size is None:
            if SLIDING_WINDOW_MODE:
                # Sliding mode: every bar (every minute)
                step_size = 1
            else:
                # Anchored mode: only market open (bars per day)
                step_size = self._calculate_bars_per_day()
                logger.info(f"Anchored mode: using step_size={step_size} (bars per day)")
        
        self.intervals = partition_sliding(
            self.df,
            window_len=window_len,
            step_size=step_size,
            horizon_len=self.horizon_len,
            max_windows=max_windows
        )
        logger.info(f"Created {len(self.intervals)} sliding windows (step_size={step_size})")
        return self
    
    def _calculate_bars_per_day(self) -> int:
        """
        Calculate average number of bars per trading day.
        
        This is used for anchored mode to ensure windows only start
        at market open (approximately once per day).
        
        Returns
        -------
        int
            Average bars per day (rounded)
        """
        if len(self.df) == 0:
            return 390  # Default for minute bars (6.5 hours * 60 minutes)
        
        # Get unique trading days
        dates = pd.to_datetime(self.df.index.date).unique()
        n_days = len(dates)
        
        if n_days == 0:
            return 390
        
        # Calculate average bars per day
        bars_per_day = len(self.df) / n_days
        
        # Round to nearest integer
        return int(round(bars_per_day))
    
    def with_horizon(self, horizon_len: int) -> 'WindowCollectionBuilder':
        """Set forecast horizon length."""
        self.horizon_len = horizon_len
        return self
    
    def with_normalization(self, method: str) -> 'WindowCollectionBuilder':
        """
        Set normalization method.
        
        Parameters
        ----------
        method : str
            'log_returns', 'pct_change', 'rolling_zscore', etc.
        """
        self.norm_method = method
        return self
    
    def with_volatility(self, method: str = 'garman_klass') -> 'WindowCollectionBuilder':
        """
        Enable volatility computation.
        
        Parameters
        ----------
        method : str
            'garman_klass' or 'parkinson'
        """
        self.vol_method = method
        return self
    
    def with_calendar_events(self) -> 'WindowCollectionBuilder':
        """Enable calendar event enrichment (FOMC, CPI, NFP)."""
        self.include_calendar = True
        return self
    
    def build(self) -> WindowCollection:
        """
        Build the WindowCollection with all configured options.
        
        Returns
        -------
        WindowCollection
            Fully enriched collection of windows
        """
        if self.intervals is None:
            raise ValueError("No windows defined. Call with_time_anchored_windows() or with_sliding_windows() first.")
        
        logger.info(f"Building WindowCollection with {len(self.intervals)} windows...")
        
        # Normalize the feature column
        normalized = normalize_window(self.df[self.feature_col], self.norm_method)
        # For Y, always use log returns regardless of X normalization
        y_normalized = normalize_window(self.df[self.feature_col], 'log_returns')
        
        # Compute volatilities if requested
        volatilities = None
        if self.vol_method:
            logger.info(f"Computing {self.vol_method} volatility...")
            volatilities = compute_all_window_volatilities(
                self.df, self.intervals, method=self.vol_method
            )
        
        # Build windows
        windows = []
        for i, interval in enumerate(self.intervals):
            try:
                # Extract X (normalized pattern)
                x_data = normalized.loc[interval.left:interval.right].values
                
                # Extract Y (future returns)
                cutoff_idx = normalized.index.get_loc(interval.right)
                y_data = y_normalized.iloc[cutoff_idx + 1:cutoff_idx + 1 + self.horizon_len].values
                
                # Skip if not enough horizon data
                if len(y_data) < self.horizon_len:
                    continue
                
                # Create WindowData
                window = WindowData(
                    idx=i,
                    x=x_data.reshape(1, -1),  # Shape: (1, n_timepoints) for aeon
                    y=y_data,
                    cutoff=interval.right,
                    interval=interval,
                    volatility=volatilities[i] if volatilities is not None else 0.0,
                )
                windows.append(window)
                
            except Exception as e:
                logger.debug(f"Skipping window {i}: {e}")
                continue
        
        collection = WindowCollection(windows)
        logger.info(f"Built collection with {len(collection)} valid windows")
        
        # NOTE: Regime classification is NO LONGER done here to avoid data leakage!
        # Regimes will be classified when split_train_test() is called,
        # using ONLY training data for threshold computation.
        # The vol_method is stored so split_train_test knows to classify.
        collection._vol_method = self.vol_method
        
        # Enrich with calendar events if requested
        if self.include_calendar:
            logger.info("Enriching with calendar events...")
            enrich_collection_with_calendar(collection)
        
        return collection
    
    def _classify_regimes(self, collection: WindowCollection):
        """Classify all windows into volatility regimes."""
        vols = collection.volatilities
        valid_vols = vols[~np.isnan(vols) & (vols > 0)]
        
        if len(valid_vols) < 10:
            logger.warning("Not enough valid volatilities for regime classification")
            return
        
        thresholds = compute_regime_thresholds(valid_vols)
        regimes = classify_all_regimes(vols, thresholds)
        
        # Update windows
        for window, regime in zip(collection.windows, regimes):
            window.regime = regime
        
        # Log distribution
        from .volatility import regime_summary
        summary = regime_summary(regimes)
        logger.info(f"Regime distribution: " + 
                   " | ".join(f"{k}: {v['count']} ({v['percentage']:.1f}%)" 
                             for k, v in summary.items()))


def build_collection_from_df(
    df: pd.DataFrame,
    window_start: time = time(8),
    window_end: time = time(9, 30),
    extend_sessions: int = 1,
    horizon_len: int = 20,
    norm_method: str = 'log_returns',
    vol_method: str = 'garman_klass',
    include_calendar: bool = True,
    feature_col: str = 'close'
) -> WindowCollection:
    """
    Convenience function to build a fully enriched WindowCollection.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLC market data
    window_start, window_end : time
        Time boundaries for windows
    extend_sessions : int
        Sessions to extend (1 = overnight)
    horizon_len : int
        Forecast horizon in bars
    norm_method : str
        Normalization method
    vol_method : str
        Volatility method
    include_calendar : bool
        Include FOMC/CPI/NFP events
    feature_col : str
        Column to use for patterns
        
    Returns
    -------
    WindowCollection
        Fully enriched collection
    """
    builder = WindowCollectionBuilder(df, feature_col=feature_col)
    
    return (builder
        .with_time_anchored_windows(window_start, window_end, extend_sessions)
        .with_horizon(horizon_len)
        .with_normalization(norm_method)
        .with_volatility(vol_method)
        .with_calendar_events() if include_calendar else builder
    ).build()
