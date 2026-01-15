"""
Visualization functions for market forecast results.

This module provides Plotly-based visualizations for:
- Forecast bands with confidence intervals
- Neighbor-based scenario analysis
- Probability cones from historical returns
- Volatility regime analysis
"""

from typing import Optional, cast, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def forecast_from_origin(origin_price: float, returns: np.ndarray) -> np.ndarray:
    """
    Convert returns to price series starting from origin price.
    
    Parameters
    ----------
    origin_price : float
        Starting price
    returns : np.ndarray
        Array of returns (as decimals, e.g., 0.01 for 1%)
        
    Returns
    -------
    np.ndarray
        Price series including origin (length = len(returns) + 1)
    """
    # Create cumulative return factors
    cum_factors = np.concatenate([[1.0], np.cumprod(1 + returns)])
    return origin_price * cum_factors


def plot_forecast_bands(
    cutoff: pd.Timestamp,
    forecast_returns: np.ndarray,
    window_size: int,
    neighbor_horizons: Optional[pd.DataFrame] = None,
    neighbor_labels: Optional[list] = None,
    title: str = "Market Forecast with Confidence Bands",
    neighbor_subplots: bool = False,
    df_original: Optional[pd.DataFrame] = None,
    actual_returns: Optional[np.ndarray] = None,
    score_dict: Optional[dict] = None,
    plot_width: int = 1400,
    plot_height: int = 800
) -> go.Figure:
    """
    Create forecast visualization with confidence bands.
    
    Parameters
    ----------
    cutoff : pd.Timestamp
        Forecast cutoff time
    forecast_returns : np.ndarray
        Mean forecasted returns
    window_size : int
        Size of historical window
    neighbor_horizons : pd.DataFrame, optional
        Individual neighbor horizon returns
    neighbor_labels : list, optional
        Labels for neighbor traces
    title : str
        Plot title
    neighbor_subplots : bool
        Whether to show individual neighbor subplots
    df_original : pd.DataFrame, optional
        Original OHLC data for showing historical prices
    actual_returns : np.ndarray, optional
        Actual returns (if available)
    score_dict : dict, optional
        Dictionary containing score metrics (mse, rmse, mae, r2) to display in subtitle
    plot_width, plot_height : int
        Figure dimensions
        
    Returns
    -------
    go.Figure
        Plotly figure
    """
    # Create subtitle with score if provided
    subtitle = ""
    if score_dict:
        subtitle = f"RMSE: {score_dict['rmse']:.4f} | MAE: {score_dict['mae']:.4f} | R2: {score_dict['r2']:.4f}"
    
    # Determine number of rows
    n_rows = 2 if neighbor_subplots and neighbor_horizons is not None else 1
    row_heights = [0.7, 0.3] if n_rows == 2 else [1.0]
    
    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=row_heights,
        vertical_spacing=0.1,
        subplot_titles=[title, 'Similar Pattern Returns'] if n_rows == 2 else [title]
    )
    
    # Get origin price from original data if available
    if df_original is not None and cutoff in df_original.index:
        origin_price = df_original.loc[cutoff, 'close']
        
        # Get historical data for context
        cutoff_idx = df_original.index.get_loc(cutoff)
        hist_start = max(0, cutoff_idx - window_size)
        hist_df = df_original.iloc[hist_start:cutoff_idx + 1]
        
        # Add historical OHLC
        fig.add_trace(
            go.Candlestick(
                x=hist_df.index,
                open=hist_df['open'],
                high=hist_df['high'],
                low=hist_df['low'],
                close=hist_df['close'],
                name='Historical',
                increasing_line_color='#26A69A',
                decreasing_line_color='#EF5350'
            ),
            row=1, col=1
        )
    else:
        origin_price = 100.0  # Default
    
    # Create forecast time index
    horizon_length = len(forecast_returns)
    if df_original is not None:
        freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
        if freq:
            forecast_index = pd.date_range(start=cutoff, periods=horizon_length + 1, freq=freq)
        else:
            # Estimate frequency from data
            time_deltas = df_original.index.to_series().diff().dropna()
            median_delta = time_deltas.median()
            forecast_index = [cutoff + median_delta * i for i in range(horizon_length + 1)]
    else:
        forecast_index = list(range(horizon_length + 1))
    
    # Convert returns to prices
    forecast_prices = forecast_from_origin(origin_price, forecast_returns)
    
    # Add confidence bands if neighbor horizons provided
    if neighbor_horizons is not None:
        # Calculate percentile bands
        neighbor_prices = np.array([
            forecast_from_origin(origin_price, row) 
            for _, row in neighbor_horizons.iterrows()
        ])
        
        p10 = np.percentile(neighbor_prices, 10, axis=0)
        p25 = np.percentile(neighbor_prices, 25, axis=0)
        p75 = np.percentile(neighbor_prices, 75, axis=0)
        p90 = np.percentile(neighbor_prices, 90, axis=0)
        
        # 80% band (10-90)
        fig.add_trace(
            go.Scatter(
                x=list(forecast_index) + list(forecast_index)[::-1],
                y=list(p90) + list(p10)[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.1)',
                line=dict(color='rgba(0, 100, 255, 0)'),
                name='80% CI'
            ),
            row=1, col=1
        )
        
        # 50% band (25-75)
        fig.add_trace(
            go.Scatter(
                x=list(forecast_index) + list(forecast_index)[::-1],
                y=list(p75) + list(p25)[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(color='rgba(0, 100, 255, 0)'),
                name='50% CI'
            ),
            row=1, col=1
        )
    
    # Add forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_prices,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Add actual if provided
    if actual_returns is not None:
        actual_prices = forecast_from_origin(origin_price, actual_returns)
        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=actual_prices,
                mode='lines+markers',
                name='Actual',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # Add neighbor subplots
    if neighbor_subplots and neighbor_horizons is not None and n_rows > 1:
        for i, (idx, row) in enumerate(neighbor_horizons.iterrows()):
            label = neighbor_labels[i] if neighbor_labels else f"Neighbor {i+1}"
            neighbor_prices = forecast_from_origin(origin_price, row.values)
            
            fig.add_trace(
                go.Scatter(
                    x=forecast_index,
                    y=neighbor_prices,
                    mode='lines',
                    name=str(label)[:10],
                    line=dict(width=1),
                    opacity=0.5
                ),
                row=2, col=1
            )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{subtitle}</sup>" if subtitle else title,
            x=0.5
        ),
        height=plot_height,
        width=plot_width,
        template='plotly_white',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_scenarios(
    cutoff: pd.Timestamp,
    forecast_returns: np.ndarray,
    neighbor_horizons: pd.DataFrame,
    neighbor_labels: list,
    df_original: pd.DataFrame,
    title: str = "Scenario Analysis",
    plot_width: int = 1400,
    plot_height: int = 800
) -> go.Figure:
    """
    Create scenario analysis visualization showing individual neighbor paths.
    
    Parameters
    ----------
    cutoff : pd.Timestamp
        Forecast cutoff time
    forecast_returns : np.ndarray
        Mean forecasted returns
    neighbor_horizons : pd.DataFrame
        Individual neighbor horizon returns
    neighbor_labels : list
        Labels for each neighbor
    df_original : pd.DataFrame
        Original OHLC data
    title : str
        Plot title
    plot_width, plot_height : int
        Figure dimensions
        
    Returns
    -------
    go.Figure
        Plotly figure with scenario paths
    """
    fig = go.Figure()
    
    # Get origin price
    origin_price = df_original.loc[cutoff, 'close']
    
    # Create forecast time index
    horizon_length = len(forecast_returns)
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
    if freq:
        forecast_index = pd.date_range(start=cutoff, periods=horizon_length + 1, freq=freq)
    else:
        time_deltas = df_original.index.to_series().diff().dropna()
        median_delta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_length + 1)]
    
    # Add historical context
    cutoff_idx = df_original.index.get_loc(cutoff)
    hist_start = max(0, cutoff_idx - 50)
    hist_df = df_original.iloc[hist_start:cutoff_idx + 1]
    
    fig.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df['open'],
            high=hist_df['high'],
            low=hist_df['low'],
            close=hist_df['close'],
            name='Historical',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        )
    )
    
    # Add each neighbor scenario
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for i, (idx, row) in enumerate(neighbor_horizons.iterrows()):
        neighbor_prices = forecast_from_origin(origin_price, row.values)
        label = neighbor_labels[i] if neighbor_labels else f"Pattern {i+1}"
        
        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=neighbor_prices,
                mode='lines',
                name=str(label)[:16],
                line=dict(color=colors[i % len(colors)], width=1.5),
                opacity=0.7
            )
        )
    
    # Add mean forecast
    forecast_prices = forecast_from_origin(origin_price, forecast_returns)
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_prices,
            mode='lines+markers',
            name='Mean Forecast',
            line=dict(color='black', width=3, dash='dash'),
            marker=dict(size=8)
        )
    )
    
    fig.update_layout(
        title=title,
        height=plot_height,
        width=plot_width,
        template='plotly_white',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_probability_cone(
    cutoff: pd.Timestamp,
    forecast_returns: np.ndarray,
    neighbor_horizons: pd.DataFrame,
    df_original: pd.DataFrame,
    title: str = "Probability Cone",
    plot_width: int = 1400,
    plot_height: int = 800
) -> go.Figure:
    """
    Create probability cone visualization with percentile bands.
    
    Parameters
    ----------
    cutoff : pd.Timestamp
        Forecast cutoff time  
    forecast_returns : np.ndarray
        Mean forecasted returns
    neighbor_horizons : pd.DataFrame
        Individual neighbor horizon returns
    df_original : pd.DataFrame
        Original OHLC data
    title : str
        Plot title
    plot_width, plot_height : int
        Figure dimensions
        
    Returns
    -------
    go.Figure
        Plotly figure with probability cone
    """
    fig = go.Figure()
    
    origin_price = df_original.loc[cutoff, 'close']
    horizon_length = len(forecast_returns)
    
    # Create forecast index
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
    if freq:
        forecast_index = pd.date_range(start=cutoff, periods=horizon_length + 1, freq=freq)
    else:
        time_deltas = df_original.index.to_series().diff().dropna()
        median_delta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_length + 1)]
    
    # Historical context
    cutoff_idx = df_original.index.get_loc(cutoff)
    hist_start = max(0, cutoff_idx - 50)
    hist_df = df_original.iloc[hist_start:cutoff_idx + 1]
    
    fig.add_trace(
        go.Candlestick(
            x=hist_df.index,
            open=hist_df['open'],
            high=hist_df['high'],
            low=hist_df['low'],
            close=hist_df['close'],
            name='Historical',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        )
    )
    
    # Calculate percentile bands
    neighbor_prices = np.array([
        forecast_from_origin(origin_price, row.values) 
        for _, row in neighbor_horizons.iterrows()
    ])
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    bands = {p: np.percentile(neighbor_prices, p, axis=0) for p in percentiles}
    
    # 90% band
    fig.add_trace(
        go.Scatter(
            x=list(forecast_index) + list(forecast_index)[::-1],
            y=list(bands[95]) + list(bands[5])[::-1],
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.1)',
            line=dict(color='rgba(100, 100, 255, 0)'),
            name='90% CI'
        )
    )
    
    # 80% band
    fig.add_trace(
        go.Scatter(
            x=list(forecast_index) + list(forecast_index)[::-1],
            y=list(bands[90]) + list(bands[10])[::-1],
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.15)',
            line=dict(color='rgba(100, 100, 255, 0)'),
            name='80% CI'
        )
    )
    
    # 50% band
    fig.add_trace(
        go.Scatter(
            x=list(forecast_index) + list(forecast_index)[::-1],
            y=list(bands[75]) + list(bands[25])[::-1],
            fill='toself',
            fillcolor='rgba(100, 100, 255, 0.25)',
            line=dict(color='rgba(100, 100, 255, 0)'),
            name='50% CI'
        )
    )
    
    # Median line
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=bands[50],
            mode='lines',
            name='Median',
            line=dict(color='blue', width=2)
        )
    )
    
    # Mean forecast
    forecast_prices = forecast_from_origin(origin_price, forecast_returns)
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_prices,
            mode='lines+markers',
            name='Mean Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        )
    )
    
    fig.update_layout(
        title=title,
        height=plot_height,
        width=plot_width,
        template='plotly_white',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_cluster_analysis(
    cutoff: pd.Timestamp,
    df_original: pd.DataFrame,
    neighbor_horizons: pd.DataFrame,
    neighbor_labels: list,
    cluster_labels: np.ndarray,
    title: str = "Cluster Analysis",
    plot_width: int = 1400,
    plot_height: int = 800
) -> go.Figure:
    """
    Create cluster analysis visualization.
    
    Parameters
    ----------
    cutoff : pd.Timestamp
        Forecast cutoff time
    df_original : pd.DataFrame
        Original OHLC data
    neighbor_horizons : pd.DataFrame
        Individual neighbor horizon returns
    neighbor_labels : list
        Labels for neighbors
    cluster_labels : np.ndarray
        Cluster assignment for each neighbor
    title : str
        Plot title
    plot_width, plot_height : int
        Figure dimensions
        
    Returns
    -------
    go.Figure
        Plotly figure with clusters
    """
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=['Price Scenarios by Cluster', 'Cluster Distribution']
    )
    
    origin_price = df_original.loc[cutoff, 'close']
    horizon_length = neighbor_horizons.shape[1]
    
    # Create forecast index
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
    if freq:
        forecast_index = pd.date_range(start=cutoff, periods=horizon_length + 1, freq=freq)
    else:
        time_deltas = df_original.index.to_series().diff().dropna()
        median_delta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_length + 1)]
    
    # Define cluster colors
    cluster_colors = {
        0: '#2ecc71',  # Green - bullish
        1: '#e74c3c',  # Red - bearish
        2: '#3498db',  # Blue - neutral
        3: '#f39c12',  # Orange
        4: '#9b59b6'   # Purple
    }
    
    unique_clusters = np.unique(cluster_labels)
    cluster_counts = {c: np.sum(cluster_labels == c) for c in unique_clusters}
    
    # Plot scenarios colored by cluster
    for i, (idx, row) in enumerate(neighbor_horizons.iterrows()):
        cluster = cluster_labels[i]
        neighbor_prices = forecast_from_origin(origin_price, row.values)
        
        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=neighbor_prices,
                mode='lines',
                name=f"Cluster {cluster}",
                line=dict(color=cluster_colors.get(cluster, '#7f8c8d'), width=1.5),
                opacity=0.6,
                showlegend=(i == list(cluster_labels).index(cluster))  # Show legend only once per cluster
            ),
            row=1, col=1
        )
    
    # Cluster distribution bar chart
    fig.add_trace(
        go.Bar(
            x=[f"Cluster {c}" for c in unique_clusters],
            y=[cluster_counts[c] for c in unique_clusters],
            marker_color=[cluster_colors.get(c, '#7f8c8d') for c in unique_clusters],
            name='Count',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title=title,
        height=plot_height,
        width=plot_width,
        template='plotly_white',
        showlegend=True
    )
    
    return fig


# =============================================================================
# VOLATILITY VISUALIZATION (NEW)
# =============================================================================

def plot_with_volatility(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    forecast_returns: np.ndarray,
    window_size: int,
    actual_returns: Optional[np.ndarray] = None,
    regime: int = -1,
    vol_series: Optional[pd.Series] = None,
    title: str = "Forecast with Volatility",
    plot_width: int = 1400,
    plot_height: int = 800,
    hist_context_bars: int = 100
) -> go.Figure:
    """
    Plot forecast with volatility subplot and regime shading.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLC data with datetime index
    cutoff : pd.Timestamp
        Forecast cutoff point
    forecast_returns : np.ndarray
        Forecasted returns
    window_size : int
        Number of bars in historical window (used for vol calculation)
    actual_returns : np.ndarray, optional
        Actual returns (if available)
    regime : int
        Query regime: 0=LOW, 1=MED, 2=HIGH, -1=not computed
    vol_series : pd.Series, optional
        Pre-computed volatility series (if None, computed from OHLC)
    title : str
        Plot title
    plot_width, plot_height : int
        Figure dimensions
    hist_context_bars : int
        Number of historical bars to show before cutoff (default: 100)
        
    Returns
    -------
    go.Figure
        Plotly figure with price and volatility subplots
    """
    from .volatility import garman_klass_volatility, REGIME_NAMES, REGIME_COLORS
    
    # Get window data - only show hist_context_bars before cutoff
    cutoff_loc = df.index.get_loc(cutoff)
    hist_start = max(0, cutoff_loc - hist_context_bars)
    window_df = df.iloc[hist_start:cutoff_loc + 1]
    horizon_size = len(forecast_returns)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=['Price Chart with Regime', 'Realized Volatility (Garman-Klass)']
    )
    
    # ROW 1: PRICE CHART
    fig.add_trace(
        go.Candlestick(
            x=window_df.index,
            open=window_df['open'],
            high=window_df['high'],
            low=window_df['low'],
            close=window_df['close'],
            name='OHLC',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350'
        ),
        row=1, col=1
    )
    
    # Forecast line
    last_price = window_df['close'].iloc[-1]
    forecast_prices = forecast_from_origin(last_price, forecast_returns)
    
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df.index))
    if freq is None:
        time_deltas = df.index.to_series().diff().dropna()
        median_delta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_size + 1)]
    else:
        forecast_index = pd.date_range(start=cutoff, periods=horizon_size + 1, freq=freq)
    
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_prices,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Actual if provided
    if actual_returns is not None:
        actual_prices = forecast_from_origin(last_price, actual_returns)
        fig.add_trace(
            go.Scatter(
                x=forecast_index,
                y=actual_prices,
                mode='lines+markers',
                name='Actual',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # Add regime shading
    if regime >= 0 and regime in REGIME_COLORS:
        fig.add_vrect(
            x0=window_df.index[0],
            x1=forecast_index[-1] if len(forecast_index) > 0 else cutoff,
            fillcolor=REGIME_COLORS[regime],
            layer='below',
            line_width=0,
            row=1, col=1
        )
    
    # ROW 2: VOLATILITY SUBPLOT
    if vol_series is None:
        gk_var = garman_klass_volatility(
            window_df['open'], window_df['high'], 
            window_df['low'], window_df['close']
        )
        vol_series = np.sqrt(gk_var.rolling(20).mean()) * 100
    else:
        vol_series = vol_series.loc[window_df.index[0]:window_df.index[-1]]
    
    fig.add_trace(
        go.Scatter(
            x=vol_series.index,
            y=vol_series.values,
            fill='tozeroy',
            name='GK Volatility',
            line=dict(color='purple', width=1),
            fillcolor='rgba(128, 0, 128, 0.3)'
        ),
        row=2, col=1
    )
    
    # Add regime threshold lines
    if len(vol_series) > 0:
        vol_median = vol_series.median()
        vol_75 = vol_series.quantile(0.67)
        
        fig.add_hline(
            y=vol_median, line_dash="dash", line_color="gray",
            annotation_text="Median", row=2, col=1
        )
        fig.add_hline(
            y=vol_75, line_dash="dash", line_color="red",
            annotation_text="High Vol Threshold", row=2, col=1
        )
    
    # LAYOUT
    regime_text = f" | Regime: {REGIME_NAMES.get(regime, 'N/A')}" if regime >= 0 else ""
    
    fig.update_layout(
        title=f"{title}{regime_text}",
        height=plot_height,
        width=plot_width,
        template='plotly_white',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


# =============================================================================
# ENHANCED QUANT VISUALIZATION (NEW)
# =============================================================================

def plot_forecast_analysis(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    forecast_returns: np.ndarray,
    actual_returns: np.ndarray,
    neighbor_windows: List[Any],
    neighbor_distances: np.ndarray,
    score_dict: Dict[str, float],
    regime: int = -1,
    signal_quality: Optional[Dict[str, Any]] = None,
    regime_timeline: Optional[Dict[str, Any]] = None,
    title: str = "Forecast Analysis",
    plot_width: int = 1600,
    plot_height: int = 1000,
    hist_context_bars: int = 100
) -> go.Figure:
    """
    Create comprehensive forecast analysis visualization with 4 subplots:
    1. Price chart with forecast vs actual
    2. Neighbor distance vs return scatter (reveals selection quality)
    3. Cumulative return comparison (forecast vs actual over horizon)
    4. Residual analysis (forecast errors over horizon)
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLC data with datetime index
    cutoff : pd.Timestamp
        Forecast cutoff point
    forecast_returns : np.ndarray
        Forecasted returns per bar
    actual_returns : np.ndarray
        Actual returns per bar
    neighbor_windows : List[WindowData]
        List of neighbor WindowData objects
    neighbor_distances : np.ndarray
        Distance to each neighbor
    score_dict : dict
        Dictionary with rmse, mae, r2, mse
    regime : int
        Query regime: 0=LOW, 1=MED, 2=HIGH
    title : str
        Plot title
    plot_width, plot_height : int
        Figure dimensions
    hist_context_bars : int
        Number of historical bars to show
        
    Returns
    -------
    go.Figure
        Plotly figure with 4 analytical subplots
    """
    from .volatility import REGIME_NAMES, REGIME_COLORS
    
    # Create 2x2 subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Price Forecast vs Actual',
            'Regime Timeline (Recent History)',
            'Cumulative Return Over Horizon',
            'Forecast Residuals (Error Analysis)'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
    )
    
    # Get window data
    cutoff_loc = df.index.get_loc(cutoff)
    hist_start = max(0, cutoff_loc - hist_context_bars)
    window_df = df.iloc[hist_start:cutoff_loc + 1]
    horizon_size = len(forecast_returns)
    
    # Create forecast time index
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df.index))
    if freq is None:
        time_deltas = df.index.to_series().diff().dropna()
        median_delta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_size + 1)]
    else:
        forecast_index = pd.date_range(start=cutoff, periods=horizon_size + 1, freq=freq)
    
    last_price = window_df['close'].iloc[-1]
    
    # =========================================================================
    # SUBPLOT 1: Price Forecast vs Actual (top-left)
    # =========================================================================
    
    # Historical candlestick
    fig.add_trace(
        go.Candlestick(
            x=window_df.index,
            open=window_df['open'],
            high=window_df['high'],
            low=window_df['low'],
            close=window_df['close'],
            name='Historical',
            increasing_line_color='#26A69A',
            decreasing_line_color='#EF5350',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Forecast
    forecast_prices = forecast_from_origin(last_price, forecast_returns)
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=forecast_prices,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#E53935', width=2, dash='dash'),
            marker=dict(size=5)
        ),
        row=1, col=1
    )
    
    # Actual
    actual_prices = forecast_from_origin(last_price, actual_returns)
    fig.add_trace(
        go.Scatter(
            x=forecast_index,
            y=actual_prices,
            mode='lines+markers',
            name='Actual',
            line=dict(color='#43A047', width=2),
            marker=dict(size=5)
        ),
        row=1, col=1
    )
    
    # Regime shading
    if regime >= 0 and regime in REGIME_COLORS:
        fig.add_vrect(
            x0=window_df.index[0],
            x1=forecast_index[-1],
            fillcolor=REGIME_COLORS[regime],
            layer='below',
            line_width=0,
            row=1, col=1
        )
    
    # =========================================================================
    # SUBPLOT 2: Regime Timeline (top-right)
    # =========================================================================
    
    if regime_timeline is not None and 'cutoffs' in regime_timeline and 'regimes' in regime_timeline:
        # Use full regime timeline
        timeline_cutoffs = regime_timeline['cutoffs']
        timeline_regimes = regime_timeline['regimes']
        
        # Limit to last N windows for readability
        max_windows = 50
        if len(timeline_cutoffs) > max_windows:
            timeline_cutoffs = timeline_cutoffs[-max_windows:]
            timeline_regimes = timeline_regimes[-max_windows:]
        
        # Create bar chart showing regime over time
        regime_colors_solid = {0: '#43A047', 1: '#FFA726', 2: '#E53935'}
        bar_colors = [regime_colors_solid.get(r, 'gray') for r in timeline_regimes]
        
        fig.add_trace(
            go.Bar(
                x=timeline_cutoffs,
                y=[1] * len(timeline_cutoffs),  # All bars same height
                marker_color=bar_colors,
                name='Regime',
                showlegend=False,
                hovertemplate="%{x}<br>Regime: %{customdata}<extra></extra>",
                customdata=[REGIME_NAMES.get(r, '?') for r in timeline_regimes]
            ),
            row=1, col=2
        )
        
        # Add current window marker (use shape instead of vline for timestamp compatibility)
        fig.add_shape(
            type="line",
            x0=cutoff, x1=cutoff,
            y0=0, y1=1.2,
            line=dict(color="black", width=3),
            row=1, col=2
        )
        fig.add_annotation(
            x=cutoff, y=1.1,
            text="NOW",
            showarrow=False,
            font=dict(size=12, color="black"),
            row=1, col=2
        )
        
        # Add legend for regimes
        for regime_id, color in regime_colors_solid.items():
            fig.add_trace(
                go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'{REGIME_NAMES.get(regime_id, "?")} Vol',
                    showlegend=True
                ),
                row=1, col=2
            )
    else:
        # Fallback: Show neighbor distance vs return scatter
        neighbor_cum_returns = []
        for w in neighbor_windows:
            cum_ret = np.sum(w.y) * 100
            neighbor_cum_returns.append(cum_ret)
        
        neighbor_cum_returns = np.array(neighbor_cum_returns)
        colors = ['#43A047' if r > 0 else '#E53935' for r in neighbor_cum_returns]
        
        fig.add_trace(
            go.Scatter(
                x=neighbor_distances,
                y=neighbor_cum_returns,
                mode='markers',
                name='Neighbors',
                marker=dict(size=12, color=colors, line=dict(width=1, color='white')),
                hovertemplate="Distance: %{x:.2e}<br>Return: %{y:.2f}%<extra></extra>"
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, row=1, col=2)
    
    # =========================================================================
    # SUBPLOT 3: Cumulative Return Over Horizon (bottom-left)
    # =========================================================================
    
    cum_forecast = np.cumsum(forecast_returns) * 100
    cum_actual = np.cumsum(actual_returns) * 100
    bars = np.arange(1, horizon_size + 1)
    
    fig.add_trace(
        go.Scatter(
            x=bars,
            y=cum_forecast,
            mode='lines+markers',
            name='Forecast Cum',
            line=dict(color='#E53935', width=2, dash='dash'),
            marker=dict(size=4),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=bars,
            y=cum_actual,
            mode='lines+markers',
            name='Actual Cum',
            line=dict(color='#43A047', width=2),
            marker=dict(size=4),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Fill between
    fig.add_trace(
        go.Scatter(
            x=list(bars) + list(bars)[::-1],
            y=list(cum_forecast) + list(cum_actual)[::-1],
            fill='toself',
            fillcolor='rgba(128, 128, 128, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=2, col=1
    )
    
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.5, row=2, col=1)
    
    # =========================================================================
    # SUBPLOT 4: Residual Analysis (bottom-right)
    # =========================================================================
    
    residuals = (forecast_returns - actual_returns) * 100  # As percentage
    
    # Bar chart of residuals
    colors_resid = ['#E53935' if r > 0 else '#43A047' for r in residuals]
    
    fig.add_trace(
        go.Bar(
            x=bars,
            y=residuals,
            name='Residuals',
            marker_color=colors_resid,
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add +/- 1 std bands
    resid_std = np.std(residuals)
    fig.add_hline(y=resid_std, line_dash="dash", line_color="gray", row=2, col=2)
    fig.add_hline(y=-resid_std, line_dash="dash", line_color="gray", row=2, col=2)
    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1, row=2, col=2)
    
    # =========================================================================
    # LAYOUT AND ANNOTATIONS
    # =========================================================================
    
    regime_name = REGIME_NAMES.get(regime, 'N/A')
    direction_correct = (np.sum(forecast_returns) > 0) == (np.sum(actual_returns) > 0)
    
    # Add statistics annotation
    stats_text = (
        f"<b>Statistical Evaluation</b><br>"
        f"RMSE: {score_dict['rmse']:.6f}<br>"
        f"MAE: {score_dict['mae']:.6f}<br>"
        f"R2: {score_dict['r2']:.4f}<br>"
        f"Direction: {'CORRECT' if direction_correct else 'WRONG'}<br>"
        f"Regime: {regime_name}"
    )
    
    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=1.02, y=0.98,
        showarrow=False,
        font=dict(size=11, family="Courier"),
        align="left",
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Add Signal Quality annotation if provided
    if signal_quality is not None:
        # Color based on signal
        if signal_quality['signal'] == "TRADE":
            signal_color = "#43A047"  # Green
            signal_icon = "TRADE"
        elif signal_quality['signal'] == "CAUTION":
            signal_color = "#FFA726"  # Orange
            signal_icon = "CAUTION"
        else:
            signal_color = "#E53935"  # Red
            signal_icon = "NO TRADE"
        
        signal_text = (
            f"<b>SIGNAL QUALITY</b><br>"
            f"<span style='color:{signal_color};font-size:14px'><b>{signal_icon}</b></span><br>"
            f"Direction: <b>{signal_quality['direction']}</b><br>"
            f"Confidence: {signal_quality['confidence']*100:.0f}%<br>"
            f"Anomaly: {signal_quality['anomaly_score']*100:.0f}%<br>"
            f"<br>"
            f"<b>What This Means:</b><br>"
        )
        
        # Add interpretation lines
        for interp in signal_quality['interpretation'][:2]:  # First 2 lines
            signal_text += f"{interp}<br>"
        
        # Add regime stability info
        regime_stability = signal_quality.get('regime_stability', 'N/A')
        windows_in_regime = signal_quality.get('windows_in_regime', 'N/A')
        is_transitioning = signal_quality.get('is_transitioning', False)
        
        if regime_stability == "STABLE":
            stability_color = "#43A047"
        elif regime_stability == "UNSTABLE":
            stability_color = "#E53935"
        else:
            stability_color = "#FFA726"
        
        signal_text += (
            f"<br>"
            f"<b>Regime Status:</b><br>"
            f"Current: {regime_name}<br>"
            f"Stability: <span style='color:{stability_color}'>{regime_stability}</span><br>"
            f"Windows in regime: {windows_in_regime}<br>"
        )
        
        if is_transitioning:
            signal_text += f"<span style='color:#E53935'><b>TRANSITIONING!</b></span><br>"
        
        # Add regime history if available
        regime_history = signal_quality.get('regime_history', [])
        if regime_history:
            signal_text += f"<br><b>Recent History:</b><br>"
            for rname, rcount in regime_history[:3]:
                signal_text += f"{rname}: {rcount} windows<br>"
        
        fig.add_annotation(
            text=signal_text,
            xref="paper", yref="paper",
            x=1.02, y=0.65,
            showarrow=False,
            font=dict(size=10, family="Courier"),
            align="left",
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=signal_color,
            borderwidth=2
        )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_xaxes(title_text="Bar", row=2, col=1)
    fig.update_xaxes(title_text="Bar", row=2, col=2)
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Regime", row=1, col=2, showticklabels=False)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Residual (%)", row=2, col=2)
    
    fig.update_layout(
        title=dict(
            text=f"{title} | K={len(neighbor_windows)} | Regime={regime_name}",
            x=0.5
        ),
        height=plot_height,
        width=plot_width,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        xaxis_rangeslider_visible=False
    )
    
    return fig
