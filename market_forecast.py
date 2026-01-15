"""
Market Similarity Search and Forecasting with Regime-Aware Analysis.

This script demonstrates the two-stage regime-aware pipeline:
1. Classify query window into volatility regime (LOW/MED/HIGH)
2. Find similar patterns using WDTW on same-regime windows only

Outputs multiple HTML reports comparing strategies.
"""
from datetime import time
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from sim_search.config import ForecastConfig
from sim_search.forecaster import (
    prepare_panel_data, similarity_search, regime_aware_similarity_search,
    forecast_from_neighbors, score_forecast,
    calculate_forecast_percentiles
)
from sim_search.times import set_default_tz, resample
from sim_search.visualization import (
    plot_forecast_bands, 
    plot_probability_cone,
    plot_scenarios,
    plot_with_volatility
)
from sim_search.windowing import partition_time_anchored
from sim_search.volatility import (
    compute_all_window_volatilities,
    compute_regime_thresholds,
    classify_regime,
    regime_summary,
    REGIME_NAMES,
    REGIME_COLORS
)


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def print_metrics(label: str, forecast: np.ndarray, actual: np.ndarray):
    """Print forecast metrics in a nice format."""
    score = score_forecast(forecast, actual)
    
    # Direction accuracy
    forecast_dir = np.sum(forecast) > 0
    actual_dir = np.sum(actual) > 0
    direction_correct = "✓" if forecast_dir == actual_dir else "✗"
    
    print(f"\n  {label}:")
    print(f"    RMSE: {score['rmse']:.6f}  |  MAE: {score['mae']:.6f}  |  R²: {score['r2']:.4f}")
    print(f"    Direction: {'LONG' if forecast_dir else 'SHORT'} → {'Correct ' + direction_correct if forecast_dir == actual_dir else 'Wrong ' + direction_correct}")
    
    return score


def run_strategy(
    name: str,
    x_train: list,
    y_train: pd.DataFrame,
    x_test,
    y_test: np.ndarray,
    df: pd.DataFrame,
    intervals: pd.IntervalIndex,
    test_idx: int,
    config: ForecastConfig,
    use_regime: bool = False,
    distance: str = 'dtw'
) -> dict:
    """
    Run a single strategy and return results.
    """
    if use_regime:
        # Regime-aware search
        neighbor_indices, neighbor_distances, query_regime, all_regimes = regime_aware_similarity_search(
            x_train,
            y_train.to_numpy(),
            x_test,
            df=df,
            intervals=intervals,
            query_idx=test_idx,
            n_neighbors=config.n_neighbors,
            distance=distance,
            vol_method='garman_klass'
        )
    else:
        # Standard search
        neighbor_indices, neighbor_distances = similarity_search(
            x_train, np.zeros(len(x_train)), x_test,
            n_neighbors=config.n_neighbors,
            impl='knn',
            distance=distance
        )
        query_regime = -1
        all_regimes = None
    
    # Generate forecast
    neighbor_horizons = y_train.iloc[neighbor_indices, :]
    forecast = forecast_from_neighbors(neighbor_horizons.to_numpy(), neighbor_distances, impl='avg')
    
    # Score
    score = score_forecast(forecast, y_test)
    
    # Percentile bands
    bands = calculate_forecast_percentiles(neighbor_horizons.to_numpy())
    
    return {
        'name': name,
        'forecast': forecast,
        'neighbor_indices': neighbor_indices,
        'neighbor_distances': neighbor_distances,
        'neighbor_horizons': neighbor_horizons,
        'query_regime': query_regime,
        'all_regimes': all_regimes,
        'score': score,
        'bands': bands,
        'distance': distance,
        'use_regime': use_regime
    }


def main():
    """Main execution function with strategy comparison."""
    
    print_header("MARKET SIMILARITY SEARCH - REGIME-AWARE ANALYSIS", "═")
    
    # Load config
    config = ForecastConfig(
        project_root=Path(__file__).parent,
        data_path=Path('data/test/NQ_2024-09-06_2025-09-13.parquet')
    )
    set_default_tz(config.timezone)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    print_header("Configuration")
    print(f"  Data path:        {config.data_path}")
    print(f"  Resample:         {config.resample or 'None (raw data)'}")
    print(f"  Forecast horizon: {config.forecast_horizon} bars")
    print(f"  N neighbors:      {config.n_neighbors}")
    print(f"  Norm method:      {config.norm_method}")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print_header("Loading Data")
    df = pd.read_parquet(config.data_path)
    if config.resample:
        df = resample(df, config.resample)
    
    print(f"  Bars loaded:  {len(df):,}")
    print(f"  Date range:   {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Columns:      {', '.join(df.columns.tolist())}")
    
    # =========================================================================
    # PREPARE WINDOWS
    # =========================================================================
    print_header("Preparing Windows")
    
    # Time-anchored windows: overnight + morning (8:00 PM - 9:30 AM next day)
    intervals = partition_time_anchored(df, time(8), time(9, 30), 1)
    x_panel, y_df, labels = prepare_panel_data(
        df, intervals, config.feature_col, config.forecast_horizon,
        norm_method=config.norm_method
    )
    
    print(f"  Total windows:    {len(x_panel)}")
    print(f"  Window size:      {x_panel[0].shape[1]} bars")
    print(f"  Horizon size:     {y_df.shape[1]} bars")
    
    # =========================================================================
    # VOLATILITY & REGIME ANALYSIS
    # =========================================================================
    print_header("Volatility & Regime Analysis")
    
    # Compute volatility for all windows
    all_vols = compute_all_window_volatilities(df, intervals, method='garman_klass')
    thresholds = compute_regime_thresholds(all_vols[:-1])  # Exclude test window
    
    print(f"\n  Garman-Klass Volatility Statistics:")
    print(f"    Mean:   {np.mean(all_vols):.6f}")
    print(f"    Std:    {np.std(all_vols):.6f}")
    print(f"    Min:    {np.min(all_vols):.6f}")
    print(f"    Max:    {np.max(all_vols):.6f}")
    
    print(f"\n  Regime Thresholds (from training data):")
    print(f"    LOW < {thresholds[0]:.6f}")
    print(f"    MED:   {thresholds[0]:.6f} - {thresholds[1]:.6f}")
    print(f"    HIGH > {thresholds[1]:.6f}")
    
    # Classify test window
    test_vol = all_vols[-1]
    test_regime = classify_regime(test_vol, thresholds)
    
    print(f"\n  Test Window Volatility:")
    print(f"    GK Vol:  {test_vol:.6f}")
    print(f"    Regime:  {REGIME_NAMES[test_regime]} {'🟢' if test_regime == 0 else '🟡' if test_regime == 1 else '🔴'}")
    
    # Regime distribution
    from sim_search.volatility import classify_all_regimes
    all_regimes = classify_all_regimes(all_vols[:-1], thresholds)
    summary = regime_summary(all_regimes)
    
    print(f"\n  Training Data Regime Distribution:")
    for name, stats in summary.items():
        bar = "█" * int(stats['percentage'] / 5)
        print(f"    {name:6s}: {stats['count']:4d} ({stats['percentage']:5.1f}%) {bar}")
    
    # Count same-regime windows
    same_regime_count = np.sum(all_regimes == test_regime)
    print(f"\n  Same-regime windows available: {same_regime_count} ({same_regime_count/len(all_regimes)*100:.1f}%)")
    
    # =========================================================================
    # SPLIT TRAIN/TEST
    # =========================================================================
    x_train = x_panel[:-1]
    y_train = y_df.iloc[:-1]
    x_test = x_panel[-1]
    y_test = y_df.iloc[-1].to_numpy()
    test_cutoff = labels[-1]
    test_idx = len(x_panel) - 1
    window_size = x_test.shape[1]
    
    print_header("Test Window")
    print(f"  Cutoff:     {test_cutoff}")
    print(f"  Train size: {len(x_train)} windows")
    
    # =========================================================================
    # RUN STRATEGIES
    # =========================================================================
    print_header("Running Strategies")
    
    strategies = {}
    
    # Strategy 1: Baseline DTW (current approach)
    print("\n  [1/3] Baseline: DTW on all windows...")
    strategies['baseline_dtw'] = run_strategy(
        name="Baseline DTW",
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        df=df, intervals=intervals, test_idx=test_idx, config=config,
        use_regime=False, distance='dtw'
    )
    
    # Strategy 2: WDTW without regime (test WDTW alone)
    print("  [2/3] WDTW on all windows...")
    strategies['wdtw_only'] = run_strategy(
        name="WDTW Only",
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        df=df, intervals=intervals, test_idx=test_idx, config=config,
        use_regime=False, distance='wdtw'
    )
    
    # Strategy 3: Regime-Aware WDTW (new approach)
    print("  [3/3] Regime-Aware: GK filter + WDTW...")
    strategies['regime_wdtw'] = run_strategy(
        name="Regime-Aware WDTW",
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        df=df, intervals=intervals, test_idx=test_idx, config=config,
        use_regime=True, distance='wdtw'
    )
    
    # =========================================================================
    # COMPARE RESULTS
    # =========================================================================
    print_header("Strategy Comparison")
    
    actual_dir = "LONG (↑)" if np.sum(y_test) > 0 else "SHORT (↓)"
    actual_cum = np.sum(y_test) * 100
    print(f"\n  Actual outcome: {actual_dir} ({actual_cum:+.2f}%)")
    
    print("\n  ┌─────────────────────────┬──────────┬──────────┬─────────┬───────────┐")
    print("  │ Strategy                │   RMSE   │   MAE    │   R²    │ Direction │")
    print("  ├─────────────────────────┼──────────┼──────────┼─────────┼───────────┤")
    
    for key, result in strategies.items():
        score = result['score']
        forecast_dir = np.sum(result['forecast']) > 0
        actual_dir_bool = np.sum(y_test) > 0
        dir_match = "✓" if forecast_dir == actual_dir_bool else "✗"
        dir_str = f"{'LONG' if forecast_dir else 'SHORT':5s} {dir_match}"
        
        print(f"  │ {result['name']:<23s} │ {score['rmse']:.6f} │ {score['mae']:.6f} │ {score['r2']:+.4f} │ {dir_str:9s} │")
    
    print("  └─────────────────────────┴──────────┴──────────┴─────────┴───────────┘")
    
    # Best strategy
    best_key = min(strategies.keys(), key=lambda k: strategies[k]['score']['rmse'])
    print(f"\n  🏆 Best by RMSE: {strategies[best_key]['name']}")
    
    # =========================================================================
    # REGIME-AWARE NEIGHBOR ANALYSIS
    # =========================================================================
    print_header("Regime-Aware Neighbor Analysis")
    
    regime_result = strategies['regime_wdtw']
    neighbor_regimes = regime_result['all_regimes'][regime_result['neighbor_indices']]
    
    print(f"\n  Query regime: {REGIME_NAMES[test_regime]}")
    print(f"\n  Neighbors found ({config.n_neighbors}):")
    print("  ┌─────┬──────────────────────┬──────────┬────────┐")
    print("  │  #  │ Cutoff               │ Distance │ Regime │")
    print("  ├─────┼──────────────────────┼──────────┼────────┤")
    
    for i, (idx, dist) in enumerate(zip(regime_result['neighbor_indices'], regime_result['neighbor_distances'])):
        neighbor_regime = regime_result['all_regimes'][idx]
        regime_str = REGIME_NAMES[neighbor_regime]
        regime_emoji = '🟢' if neighbor_regime == 0 else '🟡' if neighbor_regime == 1 else '🔴'
        cutoff_str = str(labels[idx])[:19]
        print(f"  │ {i+1:3d} │ {cutoff_str:20s} │ {dist:8.4f} │ {regime_str:3s} {regime_emoji} │")
    
    print("  └─────┴──────────────────────┴──────────┴────────┘")
    
    same_regime_neighbors = np.sum(neighbor_regimes == test_regime)
    print(f"\n  Same-regime neighbors: {same_regime_neighbors}/{config.n_neighbors} ({same_regime_neighbors/config.n_neighbors*100:.0f}%)")
    
    # =========================================================================
    # GENERATE HTML REPORTS
    # =========================================================================
    print_header("Generating HTML Reports")
    
    # Report 1: Regime-Aware with Volatility (NEW - main report)
    print("\n  📊 Creating regime-aware report with volatility subplot...")
    fig_vol = plot_with_volatility(
        df=df,
        cutoff=test_cutoff,
        forecast_returns=regime_result['forecast'],
        window_size=window_size,
        actual_returns=y_test,
        regime=test_regime,
        title=f"Regime-Aware Forecast ({REGIME_NAMES[test_regime]} Vol Regime)",
        hist_context_bars=100  # Show last 100 bars before cutoff
    )
    fig_vol.write_html("report_with_volatility.html", full_html=True, auto_open=True)
    print("     ✓ Saved: report_with_volatility.html (opened in browser)")
    
    # Report 2: Probability Cone (percentile bands)
    print("\n  📊 Creating probability cone report...")
    fig_bands = plot_probability_cone(
        cutoff=test_cutoff,
        forecast_returns=regime_result['forecast'],
        neighbor_horizons=regime_result['neighbor_horizons'],
        df_original=df,
        title=f"Probability Cone - Regime-Aware WDTW (k={config.n_neighbors})",
        plot_width=config.plot_width,
        plot_height=config.plot_height
    )
    fig_bands.write_html("report_probability_cone.html", full_html=True, auto_open=False)
    print("     ✓ Saved: report_probability_cone.html")
    
    # Report 3: Neighbor comparison view
    print("\n  📊 Creating neighbor comparison report...")
    neighbor_labels_for_plot = [labels[idx] for idx in regime_result['neighbor_indices']]
    fig_neighbors = plot_forecast_bands(
        cutoff=test_cutoff,
        forecast_returns=regime_result['forecast'],
        window_size=window_size,
        neighbor_horizons=regime_result['neighbor_horizons'],
        title=f"Neighbor Comparison - Regime-Aware WDTW",
        neighbor_subplots=config.neighbor_subplots,
        neighbor_labels=neighbor_labels_for_plot,
        df_original=df,
        actual_returns=y_test,
        score_dict=regime_result['score'],
        plot_width=config.plot_width,
        plot_height=config.plot_height
    )
    fig_neighbors.write_html("report_neighbors.html", full_html=True, auto_open=False)
    print("     ✓ Saved: report_neighbors.html")
    
    # Report 4: Scenario paths
    print("\n  📊 Creating scenario paths report...")
    fig_scenarios = plot_scenarios(
        cutoff=test_cutoff,
        forecast_returns=regime_result['forecast'],
        neighbor_horizons=regime_result['neighbor_horizons'],
        neighbor_labels=neighbor_labels_for_plot,
        df_original=df,
        title=f"Forecast Scenarios - Regime-Aware WDTW",
        plot_width=config.plot_width,
        plot_height=config.plot_height
    )
    fig_scenarios.write_html("report_scenarios.html", full_html=True, auto_open=False)
    print("     ✓ Saved: report_scenarios.html")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("Summary", "═")
    print(f"""
  Test Window:
    • Cutoff: {test_cutoff}
    • Volatility: {test_vol:.6f} ({REGIME_NAMES[test_regime]} regime)
    • Actual outcome: {'+' if np.sum(y_test) > 0 else ''}{np.sum(y_test)*100:.2f}%
    
  Best Strategy: {strategies[best_key]['name']}
    • RMSE: {strategies[best_key]['score']['rmse']:.6f}
    • Direction: {'Correct ✓' if (np.sum(strategies[best_key]['forecast']) > 0) == (np.sum(y_test) > 0) else 'Wrong ✗'}
    
  Reports Generated:
    • report_with_volatility.html   (main - with vol subplot)
    • report_probability_cone.html  (percentile bands)
    • report_neighbors.html         (neighbor comparison)
    • report_scenarios.html         (clustered scenarios)
    """)
    
    print_header("Forecast Complete! 🚀", "═")


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
