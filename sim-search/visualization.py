"""
Visualization utilities for forecast results using Plotly.
"""
from typing import cast, Optional, Tuple

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
        forecast_index = [cutoff + median_delta * (i + 1) for i in range(horizon_size)]
    else:
        forecast_index = pd.date_range(
            start=cutoff,
            periods=horizon_size + 1,
            freq=freq
        )[1:]

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

        # Set x-axis range for main plot to include forecast period
        # fig.update_xaxes(
        #     range=[historical_prices.index[0], forecast_index[-1]],
        #     row=1, col=1
        # )

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
                # name=neighbor_label,
                line=dict(color='blue', width=1),
                showlegend=False
            ), row=i + 2, col=1)

            # Plot neighbor horizon
            fig.add_trace(go.Scatter(
                x=neighbor_horizon_index,
                y=neighbor_prices,
                mode='lines+markers',
                name=f'Neighbor {i + 1} Forecast',
                # name=f'{neighbor_label} Horizon',
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

        # Plot regression bands from individual neighbors
        # if neighbor_horizons is not None and len(neighbor_horizons) > 0:
        #     for i, neighbor_ret in neighbor_horizons.reset_index(drop=True).iterrows():
        #         neighbor_prices = [last_price]
        #         for ret in neighbor_ret:
        #             neighbor_prices.append(neighbor_prices[-1] * (1 + ret))
        #         neighbor_prices = neighbor_prices[1:]
        #
        #         fig.add_trace(go.Scatter(
        #             x=forecast_index,
        #             y=neighbor_prices,
        #             mode='lines',
        #             name = f'Historical {i + 1}',
        #             # name=idx,
        #             line=dict(width=1, dash='dot'),
        #             opacity=0.3,
        #             showlegend=(i < 3)  # Only show first 3 in legend
        #         ))

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
    forecast_prices = forecast_prices[1:]  # Remove initial price
    return forecast_prices


def get_window_and_horizon(df: pd.DataFrame, cutoff: pd.Timestamp, window_size: int, horizon_len: int) \
        -> Tuple[pd.Series, pd.Series]:
    cutoff_loc = df.index.get_loc(cutoff)
    window_start = max(0, cutoff_loc - window_size)
    window = df.iloc[window_start:cutoff_loc + 1]['close']
    horizon = df.iloc[cutoff_loc + 1:cutoff_loc + 1 + horizon_len]['close']
    return window, horizon
