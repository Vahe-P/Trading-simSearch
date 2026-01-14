"""
Visualization utilities for forecast results using Plotly.
"""
from typing import cast, Optional, Tuple, Any, Dict

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots


def plot_forecast_bands(
        cutoff: pd.Timestamp,
        forecast_returns: np.ndarray,
        window_size: int = None,
        neighbor_horizons: Optional[pd.DataFrame] = None,
        title: str = "Market Forecast with Regression Bands",
        neighbor_subplots: bool = False,
        neighbor_labels: list = None,
        df_original: pd.DataFrame = None,
        actual_returns: Optional[pd.Series] = None,
        score_dict: dict = None,
        plot_width: int = 1200,
        plot_height: int = 600
):
    """
    Plot actual prices and forecast with regression bands using plotly.

    Parameters
    ----------
    cutoff : pd.Timestamp
        test cutoff
    window_size: int
        window size
    forecast_returns : array-like
        Forecasted returns
    neighbor_horizons : pd.DataFrame optional
        Individual neighbor horizons
    title : str
        Plot title
    neighbor_subplots : bool, optional
        If True, plot each neighbor forecast as a separate subplot below the main forecast
    neighbor_labels : list, optional
        Labels (timestamps) for each neighbor forecast, used to look up actual prices
    df_original : pd.DataFrame, optional
        Original dataframe to look up actual prices for neighbors
    actual_returns : array-like, optional
        Actual returns to plot alongside the forecast (if available)
    score_dict : dict, optional
        Dictionary containing score metrics (mse, rmse, mae, r2) to display in subtitle

    Returns
    -------
    plotly.graph_objects.Figure
        The plotly figure object
    """
    horizon_size = len(forecast_returns)
    test_window, test_horizon = get_window_and_horizon(df_original, cutoff, window_size, horizon_size)
    last_price = test_window.iloc[-1]
    target_forecast_px = forecast_from_origin(last_price, forecast_returns)

    # Create future index
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
    if freq is None:
        # Fallback to median time delta if frequency cannot be inferred
        time_deltas = df_original.index.to_series().diff().dropna()
        median_delta: pd.Timedelta = time_deltas.median()  # noqa
        forecast_index = [cutoff + median_delta * i for i in range(horizon_size + 1)]
    else:
        forecast_index = pd.date_range(
            start=cutoff,
            periods=horizon_size + 1,
            freq=freq
        )

    # Convert actual returns to prices if provided
    target_actual_px = None
    if actual_returns is not None:
        target_actual_px = forecast_from_origin(last_price, actual_returns)

    # Create subtitle with score if provided
    subtitle = ""
    if score_dict is not None:
        subtitle = f"RMSE: {score_dict['rmse']:.4f} | MAE: {score_dict['mae']:.4f} | R²: {score_dict['r2']:.4f}"

    # Determine if we should create subplots
    if neighbor_subplots and neighbor_horizons is not None and len(neighbor_horizons) > 0:
        # Create subplots: 1 main plot + 1 for each neighbor
        n_neighbors = len(neighbor_horizons)
        subplot_titles = [f"{title}<br><sub>{subtitle}</sub>" if subtitle else title]
        subplot_titles.extend([f"Neighbor {i + 1}" for i in range(n_neighbors)])

        fig = make_subplots(
            rows=1 + n_neighbors,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=0.03,  # Minimal space to prevent title/axis overlap
            row_heights=[1] * (1 + n_neighbors)  # All plots same height
        )

        # Main plot - historical prices (get actual prices from df_original)
        cutoff_loc = df_original.index.get_loc(cutoff)
        hist_start = max(0, cutoff_loc - window_size)
        historical_prices = df_original.iloc[hist_start:cutoff_loc + 1]['close']

        fig.add_trace(go.Scatter(
            x=historical_prices.index,
            y=historical_prices.values,
            mode='lines',
            name='Historical Close',
            line=dict(color='blue', width=2),
            showlegend=True
        ), row=1, col=1)

        # Main plot - forecast
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=target_forecast_px,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6),
            showlegend=True
        ), row=1, col=1)

        # Main plot - actual prices if available
        if target_actual_px is not None:
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=target_actual_px,
                mode='lines+markers',
                name='Actual',
                line=dict(color='green', width=2),
                marker=dict(size=6),
                showlegend=True
            ), row=1, col=1)

        # Neighbor subplots
        for i, neighbor_ret in neighbor_horizons.reset_index(drop=True).iterrows():
            # Get neighbor's last price
            i = cast(int, i)
            neighbor_label = neighbor_labels[i]
            neighbor_horizon_len = len(neighbor_ret)

            # Get historical prices for this neighbor
            neighbor_cutoff = df_original.index.get_loc(neighbor_label)
            hist_start = max(0, neighbor_cutoff - window_size)
            neighbor_hist_prices = df_original.iloc[hist_start:neighbor_cutoff + 1]['close']

            # Convert neighbor returns to prices
            neighbor_prices = [neighbor_hist_prices.iloc[-1]]
            for ret in neighbor_ret:
                neighbor_cutoff += 1
                original_px = df_original.iloc[neighbor_cutoff]['close']

                # Sanity check.
                neighbor_forecast_px = neighbor_prices[-1] * (1 + ret)
                if neighbor_cutoff > 5:
                    volatility = df_original.close.iloc[neighbor_cutoff - 5:neighbor_cutoff].diff().abs().mean()
                    forecast_err = abs(neighbor_forecast_px - original_px)
                    forecast_err_ratio = forecast_err / volatility
                    if forecast_err_ratio > 0.1:
                        raise ValueError(
                            f'Forecast was off by non-trivial amount: {forecast_err} ({forecast_err_ratio}).')
                neighbor_prices.append(original_px)
            neighbor_prices = neighbor_prices[1:]

            # Create forecast index for neighbor
            neighbor_cutoff = df_original.index.get_loc(neighbor_label)
            neighbor_horizon_index = df_original.index[neighbor_cutoff + 1:neighbor_cutoff + 1 + neighbor_horizon_len]

            # Plot neighbor historical prices
            fig.add_trace(go.Scatter(
                x=neighbor_hist_prices.index,
                y=neighbor_hist_prices.values,
                mode='lines',
                name=f'Historical {i + 1}',
                line=dict(color='blue', width=1),
                showlegend=False
            ), row=i + 2, col=1)

            # Plot neighbor horizon
            fig.add_trace(go.Scatter(
                x=neighbor_horizon_index,
                y=neighbor_prices,
                mode='lines+markers',
                name=f'Neighbor {i + 1} Forecast',
                line=dict(color='red', width=1, dash='dash'),
                marker=dict(size=4),
                showlegend=False
            ), row=i + 2, col=1)

        # Update layout - all subplots get equal height
        fig.update_layout(
            template='plotly_white',
            height=plot_height * (1 + n_neighbors),  # Each subplot gets plot_height
            width=plot_width,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update axes labels (only show 'Time' on bottom subplot)
        for i in range(1 + n_neighbors):
            if i == n_neighbors:  # Bottom subplot
                fig.update_xaxes(title_text='Time', row=i + 1, col=1)
            else:
                fig.update_xaxes(title_text='', row=i + 1, col=1)
            fig.update_yaxes(title_text='Price', row=i + 1, col=1)

    else:
        # Original single plot
        fig = go.Figure()

        # Plot historical prices (get actual prices from df_original)
        cutoff_loc = df_original.index.get_loc(cutoff)
        hist_start = max(0, cutoff_loc - window_size)
        historical_prices = df_original.iloc[hist_start:cutoff_loc + 1]['close']

        fig.add_trace(go.Scatter(
            x=historical_prices.index,
            y=historical_prices.values,
            mode='lines',
            name='Historical Close',
            line=dict(color='blue', width=2)
        ))

        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=target_forecast_px,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))

        # Plot actual forecast if available
        if target_actual_px is not None:
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=target_actual_px,
                mode='lines+markers',
                name='Actual',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))

        # Set x-axis range to include forecast period
        fig.update_xaxes(range=[historical_prices.index[0], forecast_index[-1]])

        # Update layout
        full_title = f"{title}<br><sub>{subtitle}</sub>" if subtitle else title
        fig.update_layout(
            title=full_title,
            xaxis_title='Time',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white',
            height=plot_height,
            width=plot_width,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

    return fig


def forecast_from_origin(last_price: float, forecast_returns: np.ndarray) -> list[float]:
    forecast_prices = [last_price]
    for ret in forecast_returns:
        forecast_prices.append(forecast_prices[-1] * (1 + ret))
    # Return full list including origin to connect lines
    return forecast_prices


def get_window_and_horizon(df: pd.DataFrame, cutoff: pd.Timestamp, window_size: int, horizon_len: int) \
        -> Tuple[pd.Series, pd.Series]:
    cutoff_loc = df.index.get_loc(cutoff)
    window_start = max(0, cutoff_loc - window_size)
    window = df.iloc[window_start:cutoff_loc + 1]['close']
    horizon = df.iloc[cutoff_loc + 1:cutoff_loc + 1 + horizon_len]['close']
    return window, horizon


def plot_forecast_with_percentile_bands(
        cutoff: pd.Timestamp,
        percentile_bands: dict,
        window_size: int,
        df_original: pd.DataFrame,
        actual_returns: Optional[np.ndarray] = None,
        score_dict: Optional[dict] = None,
        title: str = "Market Forecast with Probability Cone",
        plot_width: int = 1200,
        plot_height: int = 600,
        band_color: str = 'rgba(68, 68, 68, 0.3)',
        median_color: str = 'red',
        actual_color: str = 'green'
) -> go.Figure:
    """
    Plot forecast with percentile bands (probability cone) using plotly.

    Creates a visualization showing:
    - Historical prices leading up to cutoff
    - Median forecast (p50) as the main prediction line
    - Shaded area between p20 and p80 showing expected range
    - Actual prices if available for comparison

    Parameters
    ----------
    cutoff : pd.Timestamp
        Forecast origin (last known price point)
    percentile_bands : dict
        Dictionary with keys 'p20', 'p50', 'p80' containing return arrays.
        Typically from calculate_forecast_percentiles()
    window_size : int
        Number of historical bars to show before cutoff
    df_original : pd.DataFrame
        Original price data with 'close' column
    actual_returns : np.ndarray, optional
        Actual returns for comparison (if available)
    score_dict : dict, optional
        Score metrics to display in subtitle
    title : str
        Plot title
    plot_width : int
        Plot width in pixels
    plot_height : int
        Plot height in pixels
    band_color : str
        Fill color for percentile band (RGBA format)
    median_color : str
        Color for median forecast line
    actual_color : str
        Color for actual price line

    Returns
    -------
    go.Figure
        Plotly figure object
    """
    # Validate percentile bands
    required_keys = ['p20', 'p50', 'p80']
    for key in required_keys:
        if key not in percentile_bands:
            raise ValueError(f"percentile_bands must contain '{key}'. Got keys: {list(percentile_bands.keys())}")

    horizon_size = len(percentile_bands['p50'])
    
    # Get historical window and last price
    test_window, _ = get_window_and_horizon(df_original, cutoff, window_size, horizon_size)
    last_price = test_window.iloc[-1]

    # Convert returns to prices for each percentile
    p20_prices = forecast_from_origin(last_price, percentile_bands['p20'])
    p50_prices = forecast_from_origin(last_price, percentile_bands['p50'])
    p80_prices = forecast_from_origin(last_price, percentile_bands['p80'])

    # Create future index
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
    if freq is None:
        time_deltas = df_original.index.to_series().diff().dropna()
        median_delta: pd.Timedelta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_size + 1)]
    else:
        forecast_index = pd.date_range(
            start=cutoff,
            periods=horizon_size + 1,
            freq=freq
        )

    # Convert actual returns to prices if provided
    actual_prices = None
    if actual_returns is not None:
        actual_prices = forecast_from_origin(last_price, actual_returns)

    # Create subtitle with score if provided
    subtitle = ""
    if score_dict is not None:
        subtitle = f"RMSE: {score_dict['rmse']:.4f} | MAE: {score_dict['mae']:.4f} | R²: {score_dict['r2']:.4f}"

    # Create figure
    fig = go.Figure()

    # Get historical prices
    cutoff_loc = df_original.index.get_loc(cutoff)
    hist_start = max(0, cutoff_loc - window_size)
    historical_prices = df_original.iloc[hist_start:cutoff_loc + 1]['close']

    # Plot historical prices
    fig.add_trace(go.Scatter(
        x=historical_prices.index,
        y=historical_prices.values,
        mode='lines',
        name='Historical Close',
        line=dict(color='blue', width=2)
    ))

    # Plot p20 (lower bound) - invisible line as base for fill
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=p20_prices,
        mode='lines',
        name='20th Percentile',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Plot p80 (upper bound) with fill to p20
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=p80_prices,
        mode='lines',
        name='20-80 Percentile Range',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='tonexty',
        fillcolor=band_color,
        hovertemplate='Upper: %{y:.2f}<extra></extra>'
    ))

    # Plot p50 (median) as main forecast
    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=p50_prices,
        mode='lines+markers',
        name='Median Forecast (p50)',
        line=dict(color=median_color, width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Median: %{y:.2f}<extra></extra>'
    ))

    # Plot actual prices if available
    if actual_prices is not None:
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=actual_prices,
            mode='lines+markers',
            name='Actual',
            line=dict(color=actual_color, width=2),
            marker=dict(size=6),
            hovertemplate='Actual: %{y:.2f}<extra></extra>'
        ))

    # Extend x-axis to include forecast period
    fig.update_xaxes(range=[historical_prices.index[0], forecast_index[-1]])

    # Update layout
    full_title = f"{title}<br><sub>{subtitle}</sub>" if subtitle else title
    fig.update_layout(
        title=full_title,
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_white',
        height=plot_height,
        width=plot_width,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def plot_forecast_clusters(
        cutoff: pd.Timestamp,
        cluster_data: Dict[str, Any],
        window_size: int,
        df_original: pd.DataFrame,
        actual_returns: Optional[np.ndarray] = None,
        title: str = "Forecast Scenarios (Clustered Paths)",
        plot_width: int = 1200,
        plot_height: int = 600
) -> go.Figure:
    """
    Plot distinct forecast scenarios identified by clustering.
    
    Parameters
    ----------
    cutoff : pd.Timestamp
        Forecast origin
    cluster_data : dict
        Results from forecast_clusters() containing 'centers', 'probabilities', etc.
    window_size : int
        Historical window size
    df_original : pd.DataFrame
        Original price data
    actual_returns : array-like, optional
        Actual returns
    """
    centers = cluster_data['centers']  # Shape (k, horizon)
    probs = cluster_data['probabilities']
    n_clusters = len(probs)
    horizon_size = centers.shape[1]
    
    # Get historical window and last price
    test_window, _ = get_window_and_horizon(df_original, cutoff, window_size, horizon_size)
    last_price = test_window.iloc[-1]
    
    # Create future index
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df_original.index))
    if freq is None:
        time_deltas = df_original.index.to_series().diff().dropna()
        median_delta: pd.Timedelta = time_deltas.median()
        forecast_index = [cutoff + median_delta * i for i in range(horizon_size + 1)]
    else:
        forecast_index = pd.date_range(start=cutoff, periods=horizon_size+1, freq=freq)
        
    # Create Figure
    fig = go.Figure()
    
    # Plot history
    cutoff_loc = df_original.index.get_loc(cutoff)
    hist_start = max(0, cutoff_loc - window_size)
    historical_prices = df_original.iloc[hist_start:cutoff_loc + 1]['close']
    
    fig.add_trace(go.Scatter(
        x=historical_prices.index,
        y=historical_prices.values,
        mode='lines',
        name='Historical Close',
        line=dict(color='blue', width=2)
    ))
    
    # Color map for clusters (Red -> Green -> Blue ish)
    # Simple predefined colors
    colors = ['#FF4136', '#2ECC40', '#0074D9', '#FF851B', '#B10DC9']
    
    # Plot each cluster center
    for i in range(n_clusters):
        center_returns = centers[i]
        prob = probs[i]
        
        # Convert returns to prices
        scenario_prices = forecast_from_origin(last_price, center_returns)
        
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=scenario_prices,
            mode='lines+markers',
            name=f'Scenario {i+1} ({prob:.1%})',
            line=dict(color=color, width=2, dash='solid'),
            marker=dict(size=4),
            hovertemplate=f'Scenario {i+1}: %{{y:.2f}} (Prob: {prob:.1%})<extra></extra>'
        ))
        
    # Plot actual
    if actual_returns is not None:
         actual_prices = forecast_from_origin(last_price, actual_returns)
         fig.add_trace(go.Scatter(
            x=forecast_index,
            y=actual_prices,
            mode='lines+markers',
            name='Actual',
            line=dict(color='black', width=3, dash='dot'),
            marker=dict(size=6)
        ))
         
    fig.update_layout(
        title=f"{title} - {n_clusters} Distinct Paths Found",
        xaxis_title='Time',
        yaxis_title='Price',
        hovermode='x unified',
        template='plotly_white',
        height=plot_height,
        width=plot_width,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
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
    hist_context_bars: int = 100  # Only show this many bars before cutoff
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
        This controls how much "context" is visible, not the full pattern length.
        
    Returns
    -------
    go.Figure
        Plotly figure with price and volatility subplots
    """
    from .volatility import garman_klass_volatility, REGIME_NAMES, REGIME_COLORS
    
    # Get window data - only show hist_context_bars before cutoff, not the entire pattern
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
    
    # =========================================================================
    # ROW 1: PRICE CHART
    # =========================================================================
    
    # Historical prices (candlestick)
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
    
    # Forecast line - starts at cutoff and projects forward
    last_price = window_df['close'].iloc[-1]
    forecast_prices = forecast_from_origin(last_price, forecast_returns)  # Returns n+1 elements (origin + forecast)
    
    # Create forecast index - must include cutoff to match forecast_prices length
    freq = pd.infer_freq(cast(pd.DatetimeIndex, df.index))
    if freq is None:
        time_deltas = df.index.to_series().diff().dropna()
        median_delta = time_deltas.median()
        # Include cutoff as first point so forecast connects from last historical price
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
    
    # =========================================================================
    # ROW 2: VOLATILITY SUBPLOT
    # =========================================================================
    
    # Compute volatility if not provided
    if vol_series is None:
        gk_var = garman_klass_volatility(
            window_df['open'], window_df['high'], 
            window_df['low'], window_df['close']
        )
        vol_series = np.sqrt(gk_var.rolling(20).mean()) * 100  # As percentage
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
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    
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
    
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Volatility (%)', row=2, col=1)
    fig.update_xaxes(title_text='Time', row=2, col=1)
    
    return fig
