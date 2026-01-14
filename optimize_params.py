"""
Parameter Optimization Test - Compare different configurations.
"""
from datetime import time
import pandas as pd
from sim_search.backtester import BacktestConfig, run_backtest_from_file
from sim_search.times import set_default_tz


def run_comparison():
    """Run backtests with different parameter combinations."""
    
    # Base config
    base = {
        'data_path': 'data/test/NQ_2024-09-06_2025-09-13.parquet',
        'window_start_time': time(8, 0),
        'window_end_time': time(9, 30),
        'extend_sessions': 1,
        'min_train_windows': 50,  # More training data
        'tp_threshold': 0.005,
        'sl_threshold': 0.005,
        'max_test_days': 100,  # More trades for significance
        'timezone': 'America/New_York'
    }
    
    # Parameter variations to test
    configs = [
        # Original baseline
        {**base, 'n_neighbors': 7, 'distance_metric': 'euclidean', 
         'norm_method': 'log_returns', 'forecast_horizon': 20,
         'name': 'Baseline (k=7, euclidean, log_ret)'},
        
        # More neighbors
        {**base, 'n_neighbors': 15, 'distance_metric': 'euclidean', 
         'norm_method': 'log_returns', 'forecast_horizon': 20,
         'name': 'More neighbors (k=15)'},
        
        # DTW distance
        {**base, 'n_neighbors': 10, 'distance_metric': 'dtw', 
         'norm_method': 'log_returns', 'forecast_horizon': 20,
         'name': 'DTW distance (k=10)'},
        
        # Rolling z-score normalization
        {**base, 'n_neighbors': 10, 'distance_metric': 'euclidean', 
         'norm_method': 'rolling_zscore', 'forecast_horizon': 20,
         'name': 'Rolling Z-Score norm'},
        
        # Shorter horizon (less noise)
        {**base, 'n_neighbors': 10, 'distance_metric': 'euclidean', 
         'norm_method': 'log_returns', 'forecast_horizon': 10,
         'name': 'Shorter horizon (10 bars)'},
        
        # Longer horizon
        {**base, 'n_neighbors': 10, 'distance_metric': 'euclidean', 
         'norm_method': 'log_returns', 'forecast_horizon': 30,
         'name': 'Longer horizon (30 bars)'},
    ]
    
    results = []
    
    print("=" * 80)
    print("PARAMETER OPTIMIZATION - BACKTEST COMPARISON")
    print("=" * 80)
    
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\n>>> Testing: {name}")
        print("-" * 60)
        
        try:
            config = BacktestConfig(**cfg)
            result = run_backtest_from_file(config.data_path, config=config, verbose=False)
            
            print(f"  Trades: {result.total_trades}")
            print(f"  Direction: {result.direction_accuracy:.1%}")
            print(f"  Hit Ratio: {result.hit_ratio:.1%}")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
            
            results.append({
                'Configuration': name,
                'Trades': result.total_trades,
                'Direction %': f"{result.direction_accuracy:.1%}",
                'Hit Ratio %': f"{result.hit_ratio:.1%}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Avg RMSE': f"{result.avg_rmse:.6f}"
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'Configuration': name,
                'Trades': 0,
                'Direction %': 'ERROR',
                'Hit Ratio %': 'ERROR',
                'Profit Factor': 'ERROR',
                'Avg RMSE': str(e)[:30]
            })
    
    # Summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('backtest_comparison.csv', index=False)
    print("\nResults saved to backtest_comparison.csv")
    
    return results


if __name__ == "__main__":
    run_comparison()
