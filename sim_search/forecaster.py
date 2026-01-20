"""
Forecasting utilities using aeon KNeighborsTimeSeriesRegressor.

Supports multiple distance metrics including:
- DTW (Dynamic Time Warping)
- WDTW (Weighted DTW) - applies more weight to recent bars automatically
- Euclidean - fast but no time warping

NEW: Regime-aware similarity search that filters to same volatility regime
before running KNN, ensuring we only compare patterns from similar market conditions.

GPU SUPPORT: Automatic GPU acceleration when CuPy is available and USE_GPU=True.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from aeon.regression.distance_based import KNeighborsTimeSeriesRegressor
from aeon.distances import dtw_distance
from loguru import logger
import os

# GPU support (optional, via CuPy)
# Workaround for incomplete CUDA installation: patch os.add_dll_directory
_original_add_dll_directory = os.add_dll_directory
def _patched_add_dll_directory(path):
    """Handle missing CUDA bin directory gracefully."""
    if os.path.exists(path):
        return _original_add_dll_directory(path)
    # Return dummy context manager for missing directories
    class _DummyDllDir:
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    return _DummyDllDir()

# Apply patch before importing CuPy (on Windows with incomplete CUDA installs)
if os.name == 'nt':  # Windows only
    os.add_dll_directory = _patched_add_dll_directory

try:
    import cupy as cp
    # Test if GPU actually works (not just detected)
    try:
        test_arr = cp.array([1.0])
        _ = test_arr * 2
        GPU_AVAILABLE = True
        logger.info(f"GPU available: {cp.cuda.Device(0).compute_capability}")
    except RuntimeError as e:
        # GPU detected but runtime DLLs missing
        GPU_AVAILABLE = False
        cp = None
        logger.warning(f"GPU detected but runtime unavailable (NVRTC DLL missing). Install complete CUDA Toolkit. Error: {e}")
    else:
        GPU_AVAILABLE = cp.is_available()
        if GPU_AVAILABLE:
            logger.info(f"GPU available: {cp.cuda.Device(0).compute_capability}")
except (ImportError, RuntimeError, FileNotFoundError) as e:
    cp = None
    GPU_AVAILABLE = False
    logger.warning(f"GPU not available: {e}. Falling back to CPU.")

# Restore original function
if os.name == 'nt':
    os.add_dll_directory = _original_add_dll_directory

# Configuration: Enable GPU by default if available
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true" and GPU_AVAILABLE

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


# =============================================================================
# GPU-ACCELERATED DISTANCE CALCULATIONS
# =============================================================================

def _gpu_euclidean_distance_batch(x_train_gpu, x_test_gpu):
    """
    Compute Euclidean distances on GPU using batch operations.
    
    Parameters
    ----------
    x_train_gpu : cp.ndarray
        Training data on GPU, shape (n_train, n_features)
    x_test_gpu : cp.ndarray
        Test data on GPU, shape (n_test, n_features)
        
    Returns
    -------
    cp.ndarray
        Distances, shape (n_test, n_train)
    """
    if cp is None:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")
    
    # Expand dimensions for broadcasting: (n_test, 1, n_features) - (1, n_train, n_features)
    x_test_expanded = x_test_gpu[:, cp.newaxis, :]  # (n_test, 1, n_features)
    x_train_expanded = x_train_gpu[cp.newaxis, :, :]  # (1, n_train, n_features)
    
    # Compute squared differences and sum along feature axis
    diff_squared = (x_test_expanded - x_train_expanded) ** 2
    distances = cp.sqrt(cp.sum(diff_squared, axis=2))  # (n_test, n_train)
    
    return distances


def _gpu_wdtw_distance_batch(x_train_gpu, x_test_gpu, g=0.05):
    """
    Compute Weighted DTW distances on GPU using batched operations.
    
    This is a simplified batch version. For full DTW, we'd need to iterate
    or use more complex GPU kernels. This uses a weighted euclidean as approximation
    when g is high (focuses on recent bars).
    
    Parameters
    ----------
    x_train_gpu : cp.ndarray
        Training sequences on GPU, shape (n_train, seq_len)
    x_test_gpu : cp.ndarray
        Test sequences on GPU, shape (n_test, seq_len)
    g : float
        Weight decay parameter (higher = more weight on recent bars)
        
    Returns
    -------
    cp.ndarray
        Distances, shape (n_test, n_train)
    """
    if cp is None:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")
    
    seq_len = x_test_gpu.shape[1]
    
    # Create weights: exponentially decreasing from end to start
    # Higher g = more emphasis on recent bars
    weights = cp.exp(-g * cp.arange(seq_len, dtype=cp.float32)[::-1])
    weights = weights / cp.sum(weights)  # Normalize
    
    # Apply weights: element-wise multiplication along time dimension
    x_test_weighted = x_test_gpu * weights[cp.newaxis, :]
    x_train_weighted = x_train_gpu * weights[cp.newaxis, :]
    
    # Compute weighted euclidean distance (approximation to WDTW)
    # For true WDTW, would need dynamic programming on GPU
    diff_squared = (x_test_weighted[:, cp.newaxis, :] - x_train_weighted[cp.newaxis, :, :]) ** 2
    distances = cp.sqrt(cp.sum(diff_squared * weights[cp.newaxis, cp.newaxis, :], axis=2))
    
    return distances


def _gpu_knn_search(x_train_list, x_test_df, n_neighbors, distance='euclidean', distance_params=None):
    """
    GPU-accelerated KNN search using CuPy.
    
    Parameters
    ----------
    x_train_list : list of pd.DataFrame
        Training panel data
    x_test_df : pd.DataFrame
        Test data (single sample)
    n_neighbors : int
        Number of neighbors
    distance : str
        Distance metric
    distance_params : dict, optional
        Parameters for distance (e.g., {'g': 0.05} for wdtw)
        
    Returns
    -------
    tuple
        (indices, distances) as numpy arrays
    """
    # Check GPU availability dynamically
    if cp is None:
        raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x or cupy-cuda12x")
    
    if not cp.is_available():
        raise RuntimeError("CUDA device not available. Check your CUDA installation and GPU drivers.")
    
    # Convert to arrays and flatten time series
    x_train_arrays = []
    for df in x_train_list:
        arr = df.values.flatten() if hasattr(df, 'values') else np.array(df).flatten()
        x_train_arrays.append(arr)
    
    x_test_array = x_test_df.values.flatten() if hasattr(x_test_df, 'values') else np.array(x_test_df).flatten()
    
    # Handle variable-length sequences by padding to max length
    max_len = max(len(arr) for arr in x_train_arrays + [x_test_array])
    
    # Pad all arrays to same length (pad with last value, not zeros, to preserve signal)
    x_train_padded = []
    for arr in x_train_arrays:
        if len(arr) < max_len:
            padded = np.pad(arr, (0, max_len - len(arr)), mode='edge')  # Edge padding
        else:
            padded = arr[:max_len]  # Truncate if longer
        x_train_padded.append(padded)
    
    x_test_padded = x_test_array
    if len(x_test_padded) < max_len:
        x_test_padded = np.pad(x_test_padded, (0, max_len - len(x_test_padded)), mode='edge')
    elif len(x_test_padded) > max_len:
        x_test_padded = x_test_padded[:max_len]
    
    # Stack into matrices
    x_train_matrix = np.stack(x_train_padded)  # (n_train, seq_len)
    x_test_matrix = x_test_padded[np.newaxis, :]  # (1, seq_len)
    
    # Move to GPU
    x_train_gpu = cp.asarray(x_train_matrix.astype(np.float32))
    x_test_gpu = cp.asarray(x_test_matrix.astype(np.float32))
    
    # Compute distances on GPU
    if distance == 'euclidean':
        dist_matrix = _gpu_euclidean_distance_batch(x_test_gpu, x_train_gpu)
    elif distance == 'wdtw':
        g = distance_params.get('g', 0.05) if distance_params else 0.05
        dist_matrix = _gpu_wdtw_distance_batch(x_test_gpu, x_train_gpu, g=g)
    else:
        # Fallback: use euclidean for unsupported metrics
        logger.warning(f"GPU distance '{distance}' not implemented, using euclidean")
        dist_matrix = _gpu_euclidean_distance_batch(x_test_gpu, x_train_gpu)
    
    # Get k nearest neighbors
    distances_gpu = dist_matrix[0]  # First (and only) test sample
    k = min(n_neighbors, len(x_train_arrays))
    indices_gpu = cp.argsort(distances_gpu)[:k]
    distances_k_gpu = distances_gpu[indices_gpu]
    
    # Move back to CPU
    indices = cp.asnumpy(indices_gpu)
    distances = cp.asnumpy(distances_k_gpu)
    
    return indices, distances


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
    
    GPU acceleration is automatically used when available (CuPy installed and USE_GPU=True).
    Set environment variable USE_GPU=false to disable.

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
        Implementation strategy ('knn' for KNeighborsTimeSeriesRegressor, 'gpu' for GPU-accelerated)
    distance : str
        Distance metric ('dtw', 'euclidean', 'wdtw', 'msm', etc.)
    **kwargs : dict
        Additional arguments to pass to the regressor or GPU functions

    Returns
    -------
    tuple
        (neighbor_indices, neighbor_distances) where:
        - neighbor_indices: list of arrays of neighbor indices for each test sample
        - neighbor_distances: list of arrays of distances for each test sample
    """
    # Use GPU if available and enabled (check dynamically)
    use_gpu_flag = os.getenv("USE_GPU", "true").lower() == "true"
    gpu_actually_available = GPU_AVAILABLE  # Use the tested GPU_AVAILABLE flag
    
    if impl == 'gpu' or (use_gpu_flag and gpu_actually_available and impl == 'knn' and distance in ['euclidean', 'wdtw']):
        try:
            indices, distances = _gpu_knn_search(
                x_train, x_test, n_neighbors, distance=distance, distance_params=kwargs
            )
            if impl != 'gpu':  # Only log when auto-selecting GPU
                logger.debug(f"Using GPU acceleration for {distance} distance")
            return indices, distances
        except Exception as e:
            logger.warning(f"GPU search failed, falling back to CPU: {e}")
            # Fall through to CPU implementation
    
    # CPU implementation using aeon
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


def compute_signal_quality(
    neighbor_horizons: np.ndarray,
    neighbor_distances: np.ndarray,
    distance_threshold: float = None
) -> Dict[str, Any]:
    """
    Compute confidence and anomaly scores for the forecast.
    
    This transforms the system from just "LONG/SHORT" to:
    - Direction + Confidence + Anomaly detection
    
    Parameters
    ----------
    neighbor_horizons : np.ndarray
        Array of shape (n_neighbors, horizon_len) - the outcomes of neighbors
    neighbor_distances : np.ndarray
        Distance to each neighbor (lower = more similar)
    distance_threshold : float, optional
        Threshold above which to consider a match "poor"
        If None, uses median * 2 as adaptive threshold
        
    Returns
    -------
    dict containing:
        - direction: "LONG", "SHORT", or "UNCLEAR"
        - confidence: 0.0 to 1.0 (neighbor agreement)
        - anomaly_score: 0.0 to 1.0 (how unusual is this pattern)
        - signal: "TRADE", "CAUTION", or "NO_TRADE"
        - interpretation: human-readable explanation
    """
    n_neighbors = len(neighbor_horizons)
    
    # =========================================================================
    # CONFIDENCE: Do neighbors agree on direction?
    # =========================================================================
    neighbor_returns = np.sum(neighbor_horizons, axis=1)  # Total return per neighbor
    up_count = np.sum(neighbor_returns > 0)
    down_count = np.sum(neighbor_returns < 0)
    flat_count = np.sum(neighbor_returns == 0)
    
    agreement = max(up_count, down_count) / n_neighbors
    confidence = agreement  # 0.5 = split, 1.0 = unanimous
    
    # Direction based on majority
    if up_count > down_count:
        direction = "LONG"
        direction_pct = up_count / n_neighbors
    elif down_count > up_count:
        direction = "SHORT"
        direction_pct = down_count / n_neighbors
    else:
        direction = "UNCLEAR"
        direction_pct = 0.5
    
    # =========================================================================
    # ANOMALY: Are neighbors far away? (unusual pattern)
    # =========================================================================
    avg_distance = np.mean(neighbor_distances)
    min_distance = np.min(neighbor_distances)
    max_distance = np.max(neighbor_distances)
    distance_spread = max_distance - min_distance
    
    # Adaptive threshold: if distances are very small (like 1e-13), 
    # we need to scale appropriately
    if distance_threshold is None:
        # Use relative measure: how spread out are the distances?
        median_dist = np.median(neighbor_distances)
        distance_threshold = median_dist * 3 if median_dist > 0 else 1e-10
    
    # Anomaly score: what fraction of neighbors exceed threshold?
    poor_matches = np.sum(neighbor_distances > distance_threshold)
    anomaly_score = poor_matches / n_neighbors
    
    # Also consider if ALL distances are high (no good matches at all)
    if min_distance > distance_threshold:
        anomaly_score = 1.0  # All matches are poor
    
    # =========================================================================
    # SIGNAL: Combine confidence + anomaly into actionable signal
    # =========================================================================
    if confidence >= 0.7 and anomaly_score <= 0.3:
        signal = "TRADE"
        signal_strength = "STRONG" if confidence >= 0.8 else "MODERATE"
    elif confidence >= 0.6 and anomaly_score <= 0.5:
        signal = "CAUTION"
        signal_strength = "WEAK"
    else:
        signal = "NO_TRADE"
        signal_strength = "NONE"
    
    # =========================================================================
    # INTERPRETATION: Human-readable explanation
    # =========================================================================
    interpretations = []
    
    if confidence >= 0.8:
        interpretations.append(f"{int(direction_pct*100)}% of neighbors agree on {direction}")
    elif confidence >= 0.6:
        interpretations.append(f"Moderate agreement ({int(direction_pct*100)}%) on {direction}")
    else:
        interpretations.append(f"Neighbors disagree - no clear direction")
    
    if anomaly_score >= 0.7:
        interpretations.append("UNUSUAL PATTERN - no good historical matches")
    elif anomaly_score >= 0.4:
        interpretations.append("Some neighbors are poor matches - elevated uncertainty")
    else:
        interpretations.append("Good historical matches found")
    
    if signal == "NO_TRADE":
        interpretations.append(">> RECOMMENDATION: Stay flat or reduce exposure")
    elif signal == "CAUTION":
        interpretations.append(">> RECOMMENDATION: Trade with reduced size")
    else:
        interpretations.append(f">> RECOMMENDATION: {direction} signal with {signal_strength.lower()} conviction")
    
    return {
        'direction': direction,
        'confidence': confidence,
        'confidence_pct': direction_pct,
        'anomaly_score': anomaly_score,
        'signal': signal,
        'signal_strength': signal_strength,
        'interpretation': interpretations,
        'stats': {
            'up_count': int(up_count),
            'down_count': int(down_count),
            'flat_count': int(flat_count),
            'avg_distance': float(avg_distance),
            'min_distance': float(min_distance),
            'max_distance': float(max_distance),
            'distance_spread': float(distance_spread)
        }
    }


def calculate_excursion_metrics_per_neighbor(
    neighbor_indices: np.ndarray,
    neighbor_horizons: np.ndarray,
    df: pd.DataFrame,
    intervals: pd.IntervalIndex,
    entry_price: float,
    forecast_direction: bool
) -> Dict[str, Any]:
    """
    Calculate MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
    for each neighbor's forecast horizon using actual OHLC data.
    
    This measures the potential reward vs risk for each historical similar pattern,
    providing a better signal quality metric than just directional accuracy.
    
    Parameters
    ----------
    neighbor_indices : np.ndarray
        Indices of neighbors in the training set
    neighbor_horizons : np.ndarray
        Forecast horizon returns for each neighbor (normalized, typically log_returns)
    df : pd.DataFrame
        Original OHLC market data with DatetimeIndex
    intervals : pd.IntervalIndex
        Window intervals (used to map neighbor indices to cutoff times)
    entry_price : float
        Entry price at forecast origin (for calculating absolute MFE/MAE)
    forecast_direction : bool
        True for LONG (forecast > 0), False for SHORT (forecast <= 0)
        
    Returns
    -------
    dict
        Dictionary containing:
        - mfe_per_neighbor: array of MFE values for each neighbor
        - mae_per_neighbor: array of MAE values for each neighbor
        - avg_mfe: average MFE across all neighbors
        - avg_mae: average MAE across all neighbors
        - e_ratio: E-Ratio = Avg_MFE / Avg_MAE (higher = better risk/reward)
        
    Notes
    -----
    For LONG trades:
        - MFE = (Max High during horizon) - Entry Price
        - MAE = Entry Price - (Min Low during horizon)
    
    For SHORT trades:
        - MFE = Entry Price - (Min Low during horizon)
        - MAE = (Max High during horizon) - Entry Price
    """
    n_neighbors = len(neighbor_indices)
    mfe_list = []
    mae_list = []
    
    for neighbor_idx in neighbor_indices:
        try:
            # Get the cutoff time for this neighbor
            neighbor_cutoff = intervals[neighbor_idx].right
            
            # Find the position in df for this cutoff
            cutoff_loc = df.index.get_loc(neighbor_cutoff)
            
            # Extract OHLC data for the forecast horizon
            # Get horizon length from neighbor_horizons shape (assuming all neighbors have same length)
            if neighbor_horizons.ndim > 1:
                horizon_len = neighbor_horizons.shape[1]
            elif len(neighbor_indices) > 0:
                # Fallback: try to infer from first neighbor if 1D
                horizon_len = len(neighbor_horizons) // len(neighbor_indices) if len(neighbor_indices) > 0 else len(neighbor_horizons)
            else:
                horizon_len = len(neighbor_horizons)
            
            horizon_end = min(cutoff_loc + horizon_len + 1, len(df))
            horizon_data = df.iloc[cutoff_loc + 1:horizon_end]
            
            if len(horizon_data) == 0:
                # No data available for this horizon
                mfe_list.append(0.0)
                mae_list.append(0.0)
                continue
            
            # Extract high and low prices during horizon
            max_high = horizon_data['high'].max()
            min_low = horizon_data['low'].min()
            
            # Get entry price for this neighbor (close price at cutoff)
            neighbor_entry = df.loc[neighbor_cutoff, 'close']
            
            # Calculate MFE and MAE based on trade direction
            if forecast_direction:  # LONG
                mfe = max_high - neighbor_entry
                mae = neighbor_entry - min_low
            else:  # SHORT
                mfe = neighbor_entry - min_low
                mae = max_high - neighbor_entry
            
            mfe_list.append(float(mfe))
            mae_list.append(float(mae))
            
        except (KeyError, IndexError) as e:
            # If we can't find data for this neighbor, use zeros
            logger.debug(f"Could not calculate excursions for neighbor {neighbor_idx}: {e}")
            mfe_list.append(0.0)
            mae_list.append(0.0)
    
    mfe_per_neighbor = np.array(mfe_list)
    mae_per_neighbor = np.array(mae_list)
    
    # Calculate averages
    avg_mfe = np.mean(mfe_per_neighbor)
    avg_mae = np.mean(mae_per_neighbor)
    
    # Calculate E-Ratio (avoid division by zero)
    e_ratio = avg_mfe / avg_mae if avg_mae > 0 else 0.0
    
    return {
        'mfe_per_neighbor': mfe_per_neighbor,
        'mae_per_neighbor': mae_per_neighbor,
        'avg_mfe': float(avg_mfe),
        'avg_mae': float(avg_mae),
        'e_ratio': float(e_ratio)
    }


def calculate_e_ratio_from_excursions(
    mfe_values: np.ndarray,
    mae_values: np.ndarray
) -> float:
    """
    Calculate E-Ratio from arrays of MFE and MAE values.
    
    E-Ratio = Average(MFE) / Average(MAE)
    
    A high E-Ratio (> 2.0) indicates trades where potential reward significantly
    exceeds risk, even if directional accuracy is lower.
    
    Parameters
    ----------
    mfe_values : np.ndarray
        Array of MFE values
    mae_values : np.ndarray
        Array of MAE values
        
    Returns
    -------
    float
        E-Ratio value. Returns 0.0 if avg_mae is zero or arrays are empty.
    """
    if len(mfe_values) == 0 or len(mae_values) == 0:
        return 0.0
    
    avg_mfe = np.mean(mfe_values)
    avg_mae = np.mean(mae_values)
    
    if avg_mae == 0:
        return 0.0
    
    return float(avg_mfe / avg_mae)


def dtw_tail_plus_full(x, y, *, k=40, beta=0.7, window_full=0.1, window_tail=0.05):
    d_tail = dtw_distance(x[..., -k:], y[..., -k:], window=window_tail)
    d_full = dtw_distance(x, y, window=window_full)
    return beta * d_tail + (1 - beta) * d_full

