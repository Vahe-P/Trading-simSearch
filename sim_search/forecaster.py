"""
Forecasting utilities using aeon KNeighborsTimeSeriesRegressor.

Supports multiple distance metrics including:
- DTW (Dynamic Time Warping)
- WDTW (Weighted DTW) - applies more weight to recent bars automatically
- Euclidean - fast but no time warping

NEW: Regime-aware similarity search that filters to same volatility regime
before running KNN, ensuring we only compare patterns from similar market conditions.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.distances import dtw_distance
from loguru import logger

from .windowing import normalize_window
from .clustering import cluster_paths
from .volatility import (
    compute_all_window_volatilities,
    compute_regime_thresholds,
    classify_regime,
    classify_all_regimes,
    get_same_regime_indices,
    log_regime_distribution,
    REGIME_NAMES
)


def prepare_panel_data(df: pd.DataFrame, intervals: pd.IntervalIndex, feature_col='close', horizon_len: int = 1,
                       norm_method='log_returns'):
    """
    Convert df to PanelDfList format.

    Parameters
    ----------
    df : pd.DataFrame
        Market data with datetime index
    intervals : pd.IntervalIndex
        Time intervals defining window boundaries (left=start, right=cutoff)
    feature_col : str
        Which column to use for forecasting (default: 'close')
    horizon_len : int
        Number of bars in forecast horizon (default: 1)
    norm_method : str
        Normalization method for X data ('log_returns', 'pct_change', etc.)

    Returns
    -------
    tuple
        (x_panel, y_df, labels) where:
        - x_panel: list of DataFrames (normalized historical windows)
        - y_df: DataFrame (normalized forecast horizons, always log_returns)
        - labels: list of Timestamps (cutoff/forecast origin for each window)
    """
    x_panel = []
    y_rows = []
    labels = []
    x_normalized: pd.Series = normalize_window(df[feature_col], norm_method)
    y_normalized: pd.Series = normalize_window(df[feature_col], 'log_returns')

    window_size = 0
    for interval in intervals:
        xn_normalized = x_normalized.loc[interval.left:interval.right].astype(np.float64)
        if not window_size:
            window_size = len(xn_normalized)
        elif window_size != len(xn_normalized):
            logger.warning(f"Window size {window_size} does not match expected length {len(xn_normalized)}")

        # Use the cutoff as the label
        labels.append(interval.right)
        xn_normalized.reset_index(inplace=True, drop=True)

        cutoff_idx = x_normalized.index.get_loc(interval.right)
        yn_normalized = y_normalized.iloc[cutoff_idx+1:cutoff_idx+1+horizon_len].astype(np.float64)
        yn_normalized.reset_index(inplace=True, drop=True)

        # Aeon expects (n_channels, n_timepoints), so transpose for univariate data
        x_panel.append(xn_normalized.to_frame().T)
        # Table spec has each instance as a row
        y_rows.append(yn_normalized)

    longest_x = max((len(x) for x in x_panel))
    x_idx = pd.Index(range(0, longest_x))
    for i, x in enumerate(x_panel):
        x_panel[i] = x.reindex(index=x_idx, fill_value=0)

    y_df = pd.DataFrame.from_dict(dict(zip(labels, y_rows)), orient='index')
    return x_panel, y_df, labels


def build_forecaster(n_neighbors=5, weights='uniform', distance='dtw'):
    """
    Build KNeighborsTimeSeriesRegressor with specified distance metric.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors to use
    weights : str
        Weight function ('uniform' or 'distance') 'distance' weighs by inverse of distance
    distance : str
        Distance metric ('dtw', 'euclidean', 'msm', etc.)

    Returns
    -------
    KNeighborsTimeSeriesRegressor
        Configured forecaster
    """
    forecaster = KNeighborsTimeSeriesRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        distance=distance,
    )
    return forecaster


def similarity_search(x_train, y_train, x_test, n_neighbors=5, impl='knn', distance='dtw', **kwargs):
    """
    Run similarity search to find nearest neighbors.

    Parameters
    ----------
    x_train : list of pd.DataFrame
        Training panel data (list of DataFrames)
    y_train : np.ndarray
        Training targets (n_samples, n_forecast_steps)
    x_test : pd.DataFrame
        Test data
    n_neighbors : int
        Number of neighbors to find
    impl : str
        Implementation strategy ('knn' for KNeighborsTimeSeriesRegressor)
    distance : str
        Distance metric ('dtw', 'euclidean', 'msm', etc.)
    **kwargs : dict
        Additional arguments to pass to the regressor

    Returns
    -------
    tuple
        (neighbor_indices, neighbor_distances) where:
        - neighbor_indices: list of arrays of neighbor indices for each test sample
        - neighbor_distances: list of arrays of distances for each test sample
    """
    if impl == 'knn':
        forecaster = KNeighborsTimeSeriesRegressor(
            n_neighbors=n_neighbors,
            weights='uniform',
            distance=distance,
            **kwargs
        )
        # We aren't using the y_train here, and not sure how to make the current data match the format.
        # https://www.sktime.net/en/latest/api_reference/data_format.html#table-mtype-specifications
        # It says that rows=instances, and columns=features.
        # Maybe we could just treat each time index as a separate feature.
        forecaster.fit(x_train, y_train)

        # Had to apply https://github.com/sktime/sktime/pull/8093
        distances, indices = forecaster.kneighbors([x_test])
        return indices[0], distances[0]
    else:
        raise ValueError(f"Unknown similarity search implementation: {impl}")


def regime_aware_similarity_search(
    x_train: List,
    y_train: np.ndarray,
    x_test,
    df: pd.DataFrame,
    intervals: pd.IntervalIndex,
    query_idx: int,
    n_neighbors: int = 10,
    distance: str = 'wdtw',
    distance_params: Optional[dict] = None,
    vol_method: str = 'garman_klass',
    min_same_regime: int = 5
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Two-stage similarity search with regime filtering.
    
    Stage 1: Filter windows to same volatility regime (no distance computation)
    Stage 2: Run KNN with WDTW on filtered windows only
    
    This ensures we only compare patterns from similar market conditions.
    A 2% drop in low-vol is very different from 2% in high-vol (think FOMC days).
    
    Parameters
    ----------
    x_train : list
        Training panel data (list of DataFrames)
    y_train : np.ndarray
        Training targets
    x_test : DataFrame
        Test data (query window)
    df : pd.DataFrame
        Original OHLC data for volatility computation
    intervals : pd.IntervalIndex
        Window intervals
    query_idx : int
        Index of query window in intervals
    n_neighbors : int
        Number of neighbors to find
    distance : str
        Distance metric: 'wdtw' (recommended), 'dtw', 'euclidean'
    distance_params : dict, optional
        Parameters for distance metric (e.g., {'g': 0.05} for wdtw)
    vol_method : str
        Volatility method: 'garman_klass' or 'parkinson'
    min_same_regime : int
        Minimum windows required in same regime before fallback to all
        
    Returns
    -------
    tuple
        (neighbor_indices, neighbor_distances, query_regime, all_regimes)
        - neighbor_indices: indices into ORIGINAL x_train (not filtered)
        - neighbor_distances: distances to neighbors
        - query_regime: 0=LOW, 1=MED, 2=HIGH
        - all_regimes: regime labels for all training windows
    """
    n_train = len(x_train)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: REGIME FILTER (no DTW here - just volatility classification)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Compute volatility for all windows (training + query)
    all_intervals = intervals[:query_idx + 1]  # Include query
    all_vols = compute_all_window_volatilities(df, all_intervals, method=vol_method)
    
    train_vols = all_vols[:query_idx]  # Training windows only
    query_vol = all_vols[query_idx]    # Query window
    
    # Compute thresholds from training data only (no look-ahead)
    thresholds = compute_regime_thresholds(train_vols)
    
    # Classify all windows
    train_regimes = classify_all_regimes(train_vols, thresholds)
    query_regime = classify_regime(query_vol, thresholds)
    
    # Get same-regime indices
    same_regime_idx = get_same_regime_indices(train_regimes, query_regime)
    
    logger.debug(
        f"Regime filter: Query={REGIME_NAMES[query_regime]} (vol={query_vol:.4f}), "
        f"Same regime: {len(same_regime_idx)}/{n_train} windows"
    )
    
    # Fallback if not enough windows in same regime
    if len(same_regime_idx) < min_same_regime:
        logger.warning(
            f"Only {len(same_regime_idx)} windows in regime {REGIME_NAMES[query_regime]}, "
            f"falling back to all {n_train} windows"
        )
        same_regime_idx = np.arange(n_train)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: KNN with WDTW on filtered set only
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Filter training data to same regime
    x_filtered = [x_train[i] for i in same_regime_idx]
    y_filtered = y_train[same_regime_idx] if y_train is not None else None
    
    # Set default distance params for WDTW
    if distance_params is None and distance == 'wdtw':
        distance_params = {'g': 0.05}  # Moderate tail weighting
    
    # Run KNN on filtered set
    k = min(n_neighbors, len(x_filtered))
    
    forecaster = KNeighborsTimeSeriesRegressor(
        n_neighbors=k,
        weights='uniform',
        distance=distance,
        distance_params=distance_params
    )
    
    # Fit on filtered data
    dummy_y = np.zeros(len(x_filtered)) if y_filtered is None else y_filtered
    if len(dummy_y.shape) > 1:
        dummy_y = dummy_y[:, 0]  # Use first column if multi-output
    
    forecaster.fit(x_filtered, dummy_y)
    
    # Get neighbors
    distances, indices = forecaster.kneighbors([x_test])
    
    # Map filtered indices back to original indices
    original_indices = same_regime_idx[indices[0]]
    
    return original_indices, distances[0], query_regime, train_regimes


def forecast_from_neighbors(horizons: list[np.ndarray], distances: np.ndarray, impl='avg'):
    """
    Generate forecast from neighbor predictions.

    Parameters
    ----------
    horizons : list of np.ndarray
        List of neighbor forecast horizons
    distances : list of np.ndarray
        List of neighbor distances
    impl : str
        Aggregation strategy:
        - 'avg': uniform average (like KNN with weights='uniform')
        - 'weighted-avg': inverse distance weighted average (like KNN with weights='distance')

    Returns
    -------
    np.ndarray
        Forecast
    """
    eps = 1e-10  # Small epsilon to avoid division by zero

    if impl == 'avg':
        # Uniform average
        forecast = np.mean(horizons, axis=0)
    elif impl == 'weighted-avg':
        # Inverse distance weighted average
        weights = 1.0 / (distances + eps)
        forecast = np.average(horizons, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown forecast implementation: {impl}")
    return forecast


def calculate_forecast_percentiles(horizons: np.ndarray, 
                                   percentiles: list[int] = None) -> dict:
    """
    Calculate percentile bands from neighbor forecast horizons.

    Use this to create probability cones showing the distribution of
    possible future paths based on historical similar patterns.

    Parameters
    ----------
    horizons : np.ndarray
        Array of shape (n_neighbors, horizon_len) containing forecast 
        horizons from each neighbor
    percentiles : list[int], optional
        Percentiles to calculate (default: [20, 50, 80])
        - p20: lower bound of expected range
        - p50: median forecast
        - p80: upper bound of expected range

    Returns
    -------
    dict
        Dictionary with keys like 'p20', 'p50', 'p80' containing arrays
        of length horizon_len

    Examples
    --------
    >>> horizons = np.array([[0.01, 0.02], [0.02, 0.01], [-0.01, 0.03]])
    >>> bands = calculate_forecast_percentiles(horizons)
    >>> bands['p50']  # Median forecast
    array([0.01, 0.02])
    """
    if percentiles is None:
        percentiles = [20, 50, 80]
    
    # Ensure horizons is a numpy array
    if isinstance(horizons, pd.DataFrame):
        horizons = horizons.to_numpy()
    elif not isinstance(horizons, np.ndarray):
        horizons = np.array(horizons)
    
    result = {}
    for p in percentiles:
        result[f'p{p}'] = np.percentile(horizons, p, axis=0)
    
    return result


def forecast_clusters(horizons: Union[np.ndarray, pd.DataFrame, List[pd.Series]], 
                     max_k: int = 3, min_k: int = 2) -> Dict[str, Any]:
    """
    Cluster neighbor future paths into distinct scenarios.
    
    Parameters
    ----------
    horizons : array-like
        Collection of future return paths from neighbors
    max_k : int
        Max clusters
    min_k : int
        Min clusters
        
    Returns
    -------
    dict
        Clustering results (labels, centers, probs, score)
    """
    # Ensure numpy array
    if isinstance(horizons, pd.DataFrame):
        data = horizons.to_numpy()
    elif isinstance(horizons, list):
        # Handle list of Series or arrays
        data = np.vstack([np.array(x) for x in horizons])
    else:
        data = np.array(horizons)
        
    return cluster_paths(data, max_k=max_k, min_k=min_k)


def score_forecast(y_pred, y_true):
    """
    Calculate forecast score metrics.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_forecast_steps) or (n_forecast_steps,)
    y_true : np.ndarray
        Actual values, shape (n_samples, n_forecast_steps) or (n_forecast_steps,)

    Returns
    -------
    dict
        Dictionary containing:
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'r2': R-squared score
    """
    # Ensure arrays are 1D for single sample
    if y_pred.ndim == 2 and y_pred.shape[0] == 1:
        y_pred = y_pred[0]
    if y_true.ndim == 2 and y_true.shape[0] == 1:
        y_true = y_true[0]

    # Calculate metrics
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true))

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def dtw_tail_plus_full(x, y, *, k=40, beta=0.7, window_full=0.1, window_tail=0.05):
    d_tail = dtw_distance(x[..., -k:], y[..., -k:], window=window_tail)
    d_full = dtw_distance(x, y, window=window_full)
    return beta * d_tail + (1 - beta) * d_full

