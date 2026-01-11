"""
Market data similarity search and forecasting using sktime.

This script uses KNeighborsTimeSeriesRegressor with DTW distance to find
similar historical patterns and forecast future returns.
"""
from datetime import time

import numpy as np
import pandas as pd
from pathlib import Path

from market_search.config import ForecastConfig
from market_search.forecaster import (prepare_panel_data, similarity_search,
                                      forecast_from_neighbors, score_forecast
                                      )
from market_search.times import set_default_tz, resample
from market_search.visualization import plot_forecast_bands
from market_search.windowing import partition_time_anchored


def main():
    """Main execution function."""

    # Load config from .env or use defaults (can override: ForecastConfig(n_neighbors=10))
    config = ForecastConfig(
        project_root=Path(__file__).parent,
        data_path=Path('data/test/NQ_2024-09-06_2025-09-13.parquet')
    )
    set_default_tz(config.timezone)

    print("Configuration:")
    print(f"  Window size: {config.window_size}")
    print(f"  Forecast horizon: {config.forecast_horizon}")
    print(f"  N neighbors: {config.n_neighbors}")
    print(f"  Distance metric: {config.distance_metric}")
    print(f"  Forecast impl: {config.forecast_impl}")
    print(f"  Data path: {config.data_path}")
    print(f"  Resample: {config.resample}")

    print(f"\nLoading market data from {config.data_path}")
    df = pd.read_parquet(config.data_path)
    if config.resample:
        df = resample(df, config.resample)
    print(f"Loaded {len(df)} bars of data")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Prepare panel data
    print("\nPreparing panel data for sktime PanelDfList format...")
    intervals = partition_time_anchored(df, time(8), time(9, 30), 1)
    x_panel, y_df, labels = prepare_panel_data(df, intervals, config.feature_col, config.forecast_horizon,
                                               norm_method=config.norm_method)
    print(f"x_panel length: {len(x_panel)} (list of DataFrames)")
    print(f"x_panel[0] shape: {x_panel[0].shape}")
    print(f"y_df shape: {y_df.shape}")

    # Split into train and test (use last window as test)
    x_train = x_panel[:-1]
    y_train = y_df.iloc[:-1]
    x_test = x_panel[-1]
    y_test = y_df.iloc[-1]

    print(f"\nTrain windows: {len(x_train)}")
    test_cutoff = labels[-1]
    test_interval = intervals[-1]
    if test_cutoff != test_interval.right:
        raise ValueError(f'test_cutoff ({test_cutoff}) != test_interval.right ({test_interval.right})')
    test_start = test_interval.left
    print(f"\nTest window {test_start}-{test_cutoff}")
    window_size = len(x_test)

    # Build DTW distance parameters if using DTW or DDTW
    distance_params = {}
    if config.distance_metric in ['dtw', 'ddtw']:
        if config.dtw_sc_band is not None:
            distance_params['window'] = config.dtw_sc_band
        if config.dtw_itakura_max_slope is not None:
            distance_params['itakura_max_slope'] = config.dtw_itakura_max_slope

        if distance_params:
            print(f"DTW parameters: {distance_params}")

    # Run similarity search to get neighbors
    print(f"\nRunning similarity search with {config.distance_metric} distance (k={config.n_neighbors})...")
    print(f"x_train length: {len(x_train)}, x_train[0] shape: {x_train[0].shape}")
    print(f"x_test type: {type(x_test)}, x_test shape: {x_test.shape}")
    print(f"distance_params: {distance_params}")

    neighbor_indices, neighbor_distances = similarity_search(
        x_train, np.zeros(len(x_train)), x_test,
        n_neighbors=config.n_neighbors,
        impl='knn',
        distance=config.distance_metric,
        distance_params=distance_params if distance_params else None
    )
    print("Similarity search complete!")

    # Make prediction using new forecast method
    print(f"\nGenerating forecast using {config.forecast_impl} aggregation...")
    neighbor_horizons = y_train.iloc[neighbor_indices, :]
    y_pred = forecast_from_neighbors(neighbor_horizons.to_numpy(), neighbor_distances, impl=config.forecast_impl)

    # Extract forecast returns
    forecast_returns = y_pred
    actual_returns = y_test

    print(f"\nForecast statistics:")
    print(f"  Mean return: {forecast_returns.mean():.4%}")
    print(f"  Std return: {forecast_returns.std():.4%}")
    print(f"  Min return: {forecast_returns.min():.4%}")
    print(f"  Max return: {forecast_returns.max():.4%}")

    # Calculate score
    print("\nCalculating forecast score...")
    score_dict = score_forecast(forecast_returns, actual_returns)
    print(f"  RMSE: {score_dict['rmse']:.4f}")
    print(f"  MAE: {score_dict['mae']:.4f}")
    print(f"  R²: {score_dict['r2']:.4f}")

    # Get individual neighbor forecasts for plotting
    print("\nExtracting individual neighbor forecasts...")

    # Get neighbor labels (cutoffs)
    neighbor_labels_for_plot = [labels[idx] for idx in neighbor_indices]

    # Plot results
    print("\nCreating visualization...")
    fig = plot_forecast_bands(
        cutoff=test_cutoff,
        forecast_returns=forecast_returns,
        window_size=window_size,
        neighbor_horizons=neighbor_horizons,
        title=f"Market Forecast using {config.distance_metric.upper()} Similarity (k={config.n_neighbors})",
        neighbor_subplots=config.neighbor_subplots,
        neighbor_labels=neighbor_labels_for_plot,
        df_original=df,
        actual_returns=actual_returns,
        score_dict=score_dict,
        plot_width=config.plot_width,
        plot_height=config.plot_height
    )
    fig.write_html(
        "report.html",
        full_html=True,  # make a complete HTML page
        auto_open=True  # opens it in your default browser
    )
    print("\nForecast complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n" + "="*80)
        print("FULL ERROR:")
        print("="*80)
        traceback.print_exc()
        print("\nException type:", type(e).__name__)
        print("Exception message:", str(e))
        print("="*80)
        raise
