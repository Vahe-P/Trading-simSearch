#!/usr/bin/env python3
"""
GRID SEARCH - Parameter Optimization

Systematically search through parameter combinations to find optimal settings.
Results are saved to CSV for analysis.

Usage:
    # Quick search (reduced parameters, ~5 min)
    uv run python grid_search_params.py --quick
    
    # Full search (all parameters, ~30+ min)
    uv run python grid_search_params.py --full
    
    # Custom: edit PARAM_GRID below and run
    uv run python grid_search_params.py

Output:
    - grid_search_results.csv (all results)
    - best_config.json (best configuration)
    - Console summary of best configurations
"""

from datetime import time
from itertools import product
import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress loguru output during grid search
import os
os.environ["LOGURU_LEVEL"] = "WARNING"

# =============================================================================
# GRID SEARCH PARAMETERS
# =============================================================================

# Quick search (default) - fewer combinations, faster
PARAM_GRID_QUICK = {
    'n_neighbors': [5, 10, 15],
    'distance_metric': ['wdtw', 'euclidean'],
    'wdtw_g': [0.05],
    'regime_filter': [True, False],
    'calendar_filter': [True, False],
    'norm_method': ['log_returns'],
    'horizon_len': [20],
}

# Full search - comprehensive but slow
PARAM_GRID_FULL = {
    'n_neighbors': [5, 10, 15, 20],
    'distance_metric': ['wdtw', 'dtw', 'euclidean'],
    'wdtw_g': [0.01, 0.05, 0.1],
    'regime_filter': [True, False],
    'calendar_filter': [True, False],
    'norm_method': ['log_returns', 'pct_change'],
    'horizon_len': [10, 20, 30],
}

# Fixed parameters (not searched)
DATA_PATH = "data/test/NQ_2024-09-06_2025-09-13.parquet"
VOL_METHOD = "garman_klass"
WINDOW_START = time(8, 0)
WINDOW_END = time(9, 30)
EXTEND_SESSIONS = 1

# =============================================================================
# GRID SEARCH IMPLEMENTATION
# =============================================================================

def run_single_config(df, config, n_test_windows=15):
    """
    Run a single configuration and return metrics.
    Uses walk-forward validation over multiple test windows.
    """
    from sim_search import (
        WindowCollectionBuilder,
        RegimeFilter, CalendarFilter, FilterPipeline,
    )
    from sim_search.forecaster import similarity_search, forecast_from_neighbors, score_forecast
    from sim_search.datastructures import WindowCollection
    
    results = []
    
    # Build base collection
    builder = WindowCollectionBuilder(df)
    collection = (builder
        .with_time_anchored_windows(WINDOW_START, WINDOW_END, EXTEND_SESSIONS)
        .with_horizon(config['horizon_len'])
        .with_normalization(config['norm_method'])
        .with_volatility(VOL_METHOD)
        .with_calendar_events()
        .build())
    
    if len(collection) < n_test_windows + 30:
        return None
    
    # Walk-forward: test on last n_test_windows
    for test_idx in range(-n_test_windows, 0):
        try:
            train_windows = list(collection)[:test_idx]
            test_window = collection[test_idx]
            
            if len(train_windows) < 30:
                continue
            
            train_collection = WindowCollection(train_windows)
            train_collection._vol_method = VOL_METHOD
            
            if hasattr(train_collection, '_classify_regimes_no_leakage'):
                train_collection._classify_regimes_no_leakage(test_window)
            
            # Build filter pipeline
            filters = []
            if config['regime_filter']:
                filters.append(RegimeFilter(enabled=True, vol_method=VOL_METHOD))
            if config['calendar_filter']:
                filters.append(CalendarFilter(enabled=True, match_fomc_context=True))
            
            if filters:
                pipeline = FilterPipeline(filters)
                pipeline.fit(train_collection)
                filtered_indices = pipeline.transform(train_collection, query=test_window)
                filtered_train = train_collection.filter_by_indices(filtered_indices.tolist())
            else:
                filtered_train = train_collection
            
            if len(filtered_train) < config['n_neighbors']:
                continue
            
            # Run KNN
            x_train_list = [pd.DataFrame(w.x) for w in filtered_train]
            x_test_df = pd.DataFrame(test_window.x)
            
            distance_params = None
            if config['distance_metric'] == 'wdtw':
                distance_params = {"g": config['wdtw_g']}
            
            neighbor_idx, neighbor_dists = similarity_search(
                x_train_list,
                np.zeros(len(x_train_list)),
                x_test_df,
                n_neighbors=min(config['n_neighbors'], len(filtered_train)),
                impl='knn',
                distance=config['distance_metric'],
                distance_params=distance_params
            )
            
            # Forecast
            neighbor_horizons = np.stack([filtered_train[i].y for i in neighbor_idx])
            forecast = forecast_from_neighbors(neighbor_horizons, neighbor_dists, impl='avg')
            
            # Score
            score = score_forecast(forecast, test_window.y)
            
            forecast_ret = np.sum(forecast)
            actual_ret = np.sum(test_window.y)
            direction_correct = (forecast_ret > 0) == (actual_ret > 0)
            
            results.append({
                'rmse': score['rmse'],
                'mae': score['mae'],
                'r2': score['r2'],
                'direction_correct': direction_correct,
            })
            
        except Exception:
            continue
    
    if not results:
        return None
    
    return {
        'n_tests': len(results),
        'direction_accuracy': np.mean([r['direction_correct'] for r in results]),
        'avg_rmse': np.mean([r['rmse'] for r in results]),
        'avg_mae': np.mean([r['mae'] for r in results]),
        'avg_r2': np.mean([r['r2'] for r in results]),
    }


def generate_param_combinations(param_grid):
    """Generate all valid parameter combinations."""
    combinations = []
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    
    for combo in product(*values):
        config = dict(zip(keys, combo))
        # Skip invalid: wdtw_g only matters for wdtw
        if config['distance_metric'] != 'wdtw' and config.get('wdtw_g', 0.05) != 0.05:
            continue
        combinations.append(config)
    
    return combinations


def main():
    # Parse arguments
    mode = 'quick'
    if '--full' in sys.argv:
        mode = 'full'
    elif '--quick' in sys.argv:
        mode = 'quick'
    
    param_grid = PARAM_GRID_QUICK if mode == 'quick' else PARAM_GRID_FULL
    n_test_windows = 15 if mode == 'quick' else 30
    
    print("\n" + "="*80)
    print(f" GRID SEARCH - Parameter Optimization ({mode.upper()} mode)")
    print("="*80)
    
    # Load data
    print(f"\n[DATA] Loading from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Total bars: {len(df):,}")
    
    # Generate parameter combinations
    combinations = generate_param_combinations(param_grid)
    print(f"\n[GRID] Testing {len(combinations)} parameter combinations")
    print(f"[GRID] Walk-forward validation with {n_test_windows} test windows each")
    
    # Print search space
    print("\n   Search Space:")
    for key, values in param_grid.items():
        print(f"      {key}: {values}")
    
    # Estimate time
    est_time = len(combinations) * n_test_windows * 0.3  # ~0.3s per test
    print(f"\n   Estimated time: ~{est_time/60:.1f} minutes")
    
    # Run grid search
    print("\n" + "-"*80)
    print(" RUNNING GRID SEARCH")
    print("-"*80)
    
    all_results = []
    
    for i, config in enumerate(combinations):
        pct = (i + 1) / len(combinations) * 100
        config_str = f"k={config['n_neighbors']}, dist={config['distance_metric']}, reg={'Y' if config['regime_filter'] else 'N'}, cal={'Y' if config['calendar_filter'] else 'N'}"
        print(f"\n   [{i+1:3d}/{len(combinations)}] ({pct:5.1f}%) {config_str}", end="", flush=True)
        
        metrics = run_single_config(df, config, n_test_windows=n_test_windows)
        
        if metrics is None:
            print(f" -> SKIP")
            continue
        
        result = {**config, **metrics}
        all_results.append(result)
        
        print(f" -> DirAcc: {metrics['direction_accuracy']:.0%}, RMSE: {metrics['avg_rmse']:.6f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\n   ERROR: No valid results.")
        return
    
    # Save to CSV
    output_file = "grid_search_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n\n[SAVE] Results saved to {output_file}")
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print(" GRID SEARCH RESULTS")
    print("="*80)
    
    # Best by direction accuracy
    best_dir = results_df.loc[results_df['direction_accuracy'].idxmax()]
    print(f"""
   BEST BY DIRECTION ACCURACY:
   +--------------------------------------------------------------------------+
   | Direction Accuracy: {best_dir['direction_accuracy']:.1%}                                           |
   | RMSE:               {best_dir['avg_rmse']:.6f}                                        |
   | R2:                 {best_dir['avg_r2']:.4f}                                           |
   +--------------------------------------------------------------------------+
   | Configuration:                                                           |
   |   n_neighbors:      {int(best_dir['n_neighbors']):4d}                                               |
   |   distance_metric:  {best_dir['distance_metric']:12s}                                       |
   |   regime_filter:    {str(best_dir['regime_filter']):5s}                                              |
   |   calendar_filter:  {str(best_dir['calendar_filter']):5s}                                              |
   |   norm_method:      {best_dir['norm_method']:12s}                                       |
   |   horizon_len:      {int(best_dir['horizon_len']):4d}                                               |
   +--------------------------------------------------------------------------+
""")
    
    # Parameter impact analysis
    print("\n   PARAMETER IMPACT ANALYSIS:")
    print("   " + "-"*74)
    
    # By distance metric
    print("\n   By Distance Metric:")
    for metric in results_df['distance_metric'].unique():
        subset = results_df[results_df['distance_metric'] == metric]
        print(f"      {metric:12s}: Dir Acc = {subset['direction_accuracy'].mean():.1%}, RMSE = {subset['avg_rmse'].mean():.6f}")
    
    # By regime filter
    print("\n   By Regime Filter:")
    for enabled in [True, False]:
        subset = results_df[results_df['regime_filter'] == enabled]
        label = "ENABLED" if enabled else "DISABLED"
        print(f"      {label:12s}: Dir Acc = {subset['direction_accuracy'].mean():.1%}, RMSE = {subset['avg_rmse'].mean():.6f}")
    
    # By calendar filter
    print("\n   By Calendar Filter:")
    for enabled in [True, False]:
        subset = results_df[results_df['calendar_filter'] == enabled]
        label = "ENABLED" if enabled else "DISABLED"
        print(f"      {label:12s}: Dir Acc = {subset['direction_accuracy'].mean():.1%}, RMSE = {subset['avg_rmse'].mean():.6f}")
    
    # By K neighbors
    print("\n   By K Neighbors:")
    for k in sorted(results_df['n_neighbors'].unique()):
        subset = results_df[results_df['n_neighbors'] == k]
        print(f"      k={int(k):2d}:         Dir Acc = {subset['direction_accuracy'].mean():.1%}, RMSE = {subset['avg_rmse'].mean():.6f}")
    
    # Top 5 configurations
    print("\n" + "="*80)
    print(" TOP 5 CONFIGURATIONS (by Direction Accuracy)")
    print("="*80)
    
    top5 = results_df.nlargest(5, 'direction_accuracy')
    print("\n   +-----+-------+----------+-------+-------+--------+--------+")
    print("   |  #  |   K   | Distance | Reg   | Cal   | DirAcc | RMSE   |")
    print("   +-----+-------+----------+-------+-------+--------+--------+")
    
    for i, (_, row) in enumerate(top5.iterrows()):
        reg_str = "Y" if row['regime_filter'] else "N"
        cal_str = "Y" if row['calendar_filter'] else "N"
        print(f"   | {i+1:3d} | {int(row['n_neighbors']):5d} | {row['distance_metric']:8s} | {reg_str:5s} | {cal_str:5s} | {row['direction_accuracy']:.0%}    | {row['avg_rmse']:.4f} |")
    
    print("   +-----+-------+----------+-------+-------+--------+--------+")
    
    # Save best config
    best_config = {
        'n_neighbors': int(best_dir['n_neighbors']),
        'distance_metric': best_dir['distance_metric'],
        'wdtw_g': float(best_dir.get('wdtw_g', 0.05)),
        'regime_filter': bool(best_dir['regime_filter']),
        'calendar_filter': bool(best_dir['calendar_filter']),
        'norm_method': best_dir['norm_method'],
        'horizon_len': int(best_dir['horizon_len']),
        'metrics': {
            'direction_accuracy': float(best_dir['direction_accuracy']),
            'avg_rmse': float(best_dir['avg_rmse']),
            'avg_r2': float(best_dir['avg_r2']),
        }
    }
    
    import json
    with open('best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\n[SAVE] Best configuration saved to best_config.json")
    
    print("\n" + "="*80)
    print(" HOW TO USE BEST CONFIG")
    print("="*80)
    print("""
   1. Copy parameters to config_playground.py:
   
      N_NEIGHBORS = """ + str(int(best_dir['n_neighbors'])) + """
      DISTANCE_METRIC = \"""" + best_dir['distance_metric'] + """\"
      REGIME_FILTER_ENABLED = """ + str(best_dir['regime_filter']) + """
      CALENDAR_FILTER_ENABLED = """ + str(best_dir['calendar_filter']) + """
   
   2. Or load from best_config.json programmatically
   
   3. Run playground:
      uv run python config_playground.py
""")
    print("="*80)


if __name__ == "__main__":
    main()
