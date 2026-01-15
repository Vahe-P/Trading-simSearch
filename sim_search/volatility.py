"""
Volatility Computation and Regime Classification.

This module provides efficient realized volatility estimators using OHLC data
and regime classification for filtering similarity search to same-regime patterns.

Key insight: We have full OHLC data but only use close price. Garman-Klass volatility
uses all OHLC information and is ~5x more statistically efficient than close-to-close.

Why compute realized vol instead of using VXN?
- VXN is daily implied vol (forward-looking, options-derived)
- We need intraday realized vol at the same granularity as our patterns
- Realized vol tells us what actually happened in each window
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from loguru import logger


def garman_klass_volatility(
    open_: pd.Series, 
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series
) -> pd.Series:
    """
    Garman-Klass (1980) volatility estimator.
    
    Uses all OHLC information for ~5x more efficient estimation than close-to-close.
    
    Formula: σ² = 0.5 × ln(H/L)² − (2ln2 − 1) × ln(C/O)²
    
    Parameters
    ----------
    open_, high, low, close : pd.Series
        OHLC price series with matching index
        
    Returns
    -------
    pd.Series
        Per-bar Garman-Klass variance estimate
        
    References
    ----------
    Garman, M. B., & Klass, M. J. (1980). On the estimation of security price 
    volatilities from historical data. Journal of Business, 53(1), 67-78.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    
    # GK variance formula
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    
    return gk_var


def parkinson_volatility(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Parkinson (1980) volatility estimator.
    
    Uses high-low range, ~5x more efficient than close-to-close.
    Simpler than Garman-Klass but doesn't use open/close.
    
    Formula: σ² = ln(H/L)² / (4 × ln(2))
    
    Parameters
    ----------
    high, low : pd.Series
        High and low price series
        
    Returns
    -------
    pd.Series
        Per-bar Parkinson variance estimate
    """
    log_hl = np.log(high / low)
    return (log_hl ** 2) / (4 * np.log(2))


def window_volatility(
    df: pd.DataFrame, 
    start: pd.Timestamp, 
    end: pd.Timestamp,
    method: str = 'garman_klass'
) -> float:
    """
    Compute realized volatility for a single window.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLC data with columns: open, high, low, close
    start, end : pd.Timestamp
        Window boundaries
    method : str
        Volatility estimator: 'garman_klass' or 'parkinson'
        
    Returns
    -------
    float
        Annualized volatility for the window (as standard deviation)
    """
    window = df.loc[start:end]
    
    if len(window) < 2:
        return 0.0
    
    if method == 'garman_klass':
        var_series = garman_klass_volatility(
            window['open'], window['high'], window['low'], window['close']
        )
    elif method == 'parkinson':
        var_series = parkinson_volatility(window['high'], window['low'])
    else:
        raise ValueError(f"Unknown volatility method: {method}")
    
    # Average variance, then sqrt for vol
    mean_var = var_series.mean()
    
    # Handle negative variance (can happen with GK in edge cases)
    if mean_var < 0:
        mean_var = abs(mean_var)
    
    return np.sqrt(mean_var)


def compute_all_window_volatilities(
    df: pd.DataFrame,
    intervals: pd.IntervalIndex,
    method: str = 'garman_klass'
) -> np.ndarray:
    """
    Compute volatility for all windows efficiently.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLC data
    intervals : pd.IntervalIndex
        Window boundaries
    method : str
        Volatility method
        
    Returns
    -------
    np.ndarray
        Array of volatilities, one per window
    """
    vols = np.zeros(len(intervals))
    
    for i, interval in enumerate(intervals):
        vols[i] = window_volatility(df, interval.left, interval.right, method)
    
    return vols


def compute_regime_thresholds(
    volatilities: np.ndarray,
    method: str = 'percentile',
    low_percentile: float = 33.0,
    high_percentile: float = 67.0
) -> Tuple[float, float]:
    """
    Compute thresholds for regime classification.
    
    Parameters
    ----------
    volatilities : np.ndarray
        Array of volatility values
    method : str
        Threshold method: 'percentile' or 'std'
    low_percentile, high_percentile : float
        Percentiles for LOW/HIGH cutoffs (only for 'percentile' method)
        
    Returns
    -------
    tuple[float, float]
        (low_threshold, high_threshold)
    """
    if method == 'percentile':
        low_thresh = np.percentile(volatilities, low_percentile)
        high_thresh = np.percentile(volatilities, high_percentile)
    elif method == 'std':
        mean_vol = np.mean(volatilities)
        std_vol = np.std(volatilities)
        low_thresh = mean_vol - 0.5 * std_vol
        high_thresh = mean_vol + 0.5 * std_vol
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    
    return (low_thresh, high_thresh)


def classify_regime(vol: float, thresholds: Tuple[float, float]) -> int:
    """
    Classify a single volatility value into a regime.
    
    Parameters
    ----------
    vol : float
        Volatility value
    thresholds : tuple[float, float]
        (low_threshold, high_threshold)
        
    Returns
    -------
    int
        Regime: 0=LOW, 1=MEDIUM, 2=HIGH
    """
    low_thresh, high_thresh = thresholds
    
    if vol < low_thresh:
        return 0  # LOW
    elif vol > high_thresh:
        return 2  # HIGH
    else:
        return 1  # MEDIUM


def classify_all_regimes(
    volatilities: np.ndarray,
    thresholds: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Classify all windows into regimes.
    
    Parameters
    ----------
    volatilities : np.ndarray
        Array of volatility values
    thresholds : tuple, optional
        Pre-computed thresholds. If None, computed from volatilities.
        
    Returns
    -------
    np.ndarray
        Array of regime labels (0=LOW, 1=MED, 2=HIGH)
    """
    if thresholds is None:
        thresholds = compute_regime_thresholds(volatilities)
    
    regimes = np.zeros(len(volatilities), dtype=int)
    for i, vol in enumerate(volatilities):
        regimes[i] = classify_regime(vol, thresholds)
    
    return regimes


def get_same_regime_indices(
    regimes: np.ndarray,
    query_regime: int,
    exclude_indices: Optional[List[int]] = None
) -> np.ndarray:
    """
    Get indices of windows in the same regime as query.
    
    Parameters
    ----------
    regimes : np.ndarray
        Array of regime labels
    query_regime : int
        Target regime to match
    exclude_indices : list, optional
        Indices to exclude (e.g., the query itself)
        
    Returns
    -------
    np.ndarray
        Indices of same-regime windows
    """
    same_regime = np.where(regimes == query_regime)[0]
    
    if exclude_indices is not None:
        exclude_set = set(exclude_indices)
        same_regime = np.array([i for i in same_regime if i not in exclude_set])
    
    return same_regime


REGIME_NAMES = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH'}
REGIME_COLORS = {
    0: 'rgba(0, 200, 0, 0.15)',      # Green for low vol
    1: 'rgba(255, 200, 0, 0.15)',    # Yellow for medium vol
    2: 'rgba(255, 50, 50, 0.15)'     # Red for high vol
}


def regime_summary(regimes: np.ndarray) -> dict:
    """
    Get summary statistics for regime distribution.
    
    Parameters
    ----------
    regimes : np.ndarray
        Array of regime labels
        
    Returns
    -------
    dict
        Summary with counts and percentages per regime
    """
    total = len(regimes)
    summary = {}
    
    for regime_id, name in REGIME_NAMES.items():
        count = np.sum(regimes == regime_id)
        summary[name] = {
            'count': count,
            'percentage': count / total * 100 if total > 0 else 0
        }
    
    return summary


def log_regime_distribution(regimes: np.ndarray, prefix: str = ""):
    """Log regime distribution for debugging."""
    summary = regime_summary(regimes)
    
    parts = []
    for name, stats in summary.items():
        parts.append(f"{name}: {stats['count']} ({stats['percentage']:.1f}%)")
    
    msg = f"{prefix}Regime distribution: " + " | ".join(parts)
    logger.info(msg)


def analyze_regime_transitions(
    regimes: np.ndarray,
    cutoffs: List[pd.Timestamp],
    current_regime: int
) -> dict:
    """
    Analyze regime transitions to detect instability.
    
    This tells you:
    - How long we've been in current regime (stability)
    - When the last transition happened
    - How many transitions in recent history
    
    Parameters
    ----------
    regimes : np.ndarray
        Array of regime labels for each window (historical)
    cutoffs : list
        Cutoff timestamps for each window
    current_regime : int
        The regime of the test/current window
        
    Returns
    -------
    dict containing:
        - current_regime: int (0=LOW, 1=MED, 2=HIGH)
        - current_regime_name: str
        - windows_in_current_regime: int (streak length)
        - regime_stability: str ("STABLE", "RECENT_CHANGE", "UNSTABLE")
        - last_transition_idx: int (when did regime last change)
        - last_transition_date: Timestamp or None
        - transitions_last_10: int (how many changes in last 10 windows)
        - transitions_last_20: int (how many changes in last 20 windows)
        - regime_history: list of (regime, count) for recent regimes
        - is_transitioning: bool (True if we just changed regime)
    """
    n = len(regimes)
    
    if n == 0:
        return {
            'current_regime': current_regime,
            'current_regime_name': REGIME_NAMES.get(current_regime, 'N/A'),
            'windows_in_current_regime': 0,
            'regime_stability': 'UNKNOWN',
            'last_transition_idx': None,
            'last_transition_date': None,
            'transitions_last_10': 0,
            'transitions_last_20': 0,
            'regime_history': [],
            'is_transitioning': False
        }
    
    # Count how many recent windows match current regime (streak from end)
    streak = 0
    for i in range(n - 1, -1, -1):
        if regimes[i] == current_regime:
            streak += 1
        else:
            break
    
    # Find last transition point
    last_transition_idx = None
    for i in range(n - 1, 0, -1):
        if regimes[i] != regimes[i - 1]:
            last_transition_idx = i
            break
    
    last_transition_date = cutoffs[last_transition_idx] if last_transition_idx is not None and last_transition_idx < len(cutoffs) else None
    
    # Count transitions in last N windows
    def count_transitions(arr):
        if len(arr) < 2:
            return 0
        return np.sum(arr[1:] != arr[:-1])
    
    transitions_last_10 = count_transitions(regimes[-10:]) if n >= 10 else count_transitions(regimes)
    transitions_last_20 = count_transitions(regimes[-20:]) if n >= 20 else count_transitions(regimes)
    
    # Determine stability
    if streak >= 10 and transitions_last_20 <= 2:
        stability = "STABLE"
    elif streak >= 5 and transitions_last_10 <= 1:
        stability = "STABLE"
    elif streak <= 2 or transitions_last_10 >= 3:
        stability = "UNSTABLE"
    else:
        stability = "RECENT_CHANGE"
    
    # Check if we just changed (current regime differs from most recent training)
    is_transitioning = (regimes[-1] != current_regime) if n > 0 else False
    
    # Build regime history (last 5 regime changes)
    regime_history = []
    current_run_regime = regimes[-1] if n > 0 else None
    current_run_count = 0
    
    for i in range(n - 1, -1, -1):
        if regimes[i] == current_run_regime:
            current_run_count += 1
        else:
            if current_run_regime is not None:
                regime_history.append((REGIME_NAMES.get(current_run_regime, '?'), current_run_count))
            current_run_regime = regimes[i]
            current_run_count = 1
        if len(regime_history) >= 5:
            break
    
    if current_run_regime is not None and len(regime_history) < 5:
        regime_history.append((REGIME_NAMES.get(current_run_regime, '?'), current_run_count))
    
    return {
        'current_regime': current_regime,
        'current_regime_name': REGIME_NAMES.get(current_regime, 'N/A'),
        'windows_in_current_regime': streak,
        'regime_stability': stability,
        'last_transition_idx': last_transition_idx,
        'last_transition_date': last_transition_date,
        'transitions_last_10': transitions_last_10,
        'transitions_last_20': transitions_last_20,
        'regime_history': regime_history,
        'is_transitioning': is_transitioning,
        'total_windows': n
    }
