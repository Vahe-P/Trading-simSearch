"""
Grid Search with CSV Export - Milestone 3.3.

Runs multiple backtest configurations in PARALLEL and exports results to CSV.
"""
from datetime import time
from pathlib import Path

from sim_search.optimizer import GridSearch
from sim_search.reporting import BacktestReport
from sim_search.times import set_default_tz

def run_grid_search():
    """Run parameter grid search and export results to CSV."""
    
    # Base configuration
    base_config = {
        'data_path': 'data/test/NQ_2024-09-06_2025-09-13.parquet',
        'window_start_time': time(8, 0),
        'window_end_time': time(9, 30),
        'extend_sessions': 1,
        'min_train_windows': 30,
        'tp_threshold': 0.005,
        'sl_threshold': 0.005,
        'max_test_days': 40,  # Limit for faster testing
        'timezone': 'America/New_York',
        'resample': '5min',
    }
    
    # Parameter grid
    # Trying more combinations now since we have parallel execution!
    param_grid = {
        'n_neighbors': [5, 10, 15, 20],
        'forecast_horizon': [10, 20, 30],
        'distance_metric': ['euclidean'],  # DTW still slow, keep limited
        'norm_method': ['log_returns', 'pct_change'],  # Compare normalization methods
    }
    
    optimizer = GridSearch(
        base_config=base_config,
        param_grid=param_grid,
        max_workers=None  # Use all available cores
    )
    
    # Run optimization
    report = optimizer.run()
    
    # Print summary
    report.print_summary()
    
    # Export to CSV
    output_dir = Path("reports")
    paths = report.export_all(str(output_dir))
    
    print("\n" + "=" * 70)
    print("EXPORTED FILES:")
    print("=" * 70)
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    # Ensure safe multiprocessing on Windows
    # (Though ProcessPoolExecutor handles this, good practice to keep main guard)
    report = run_grid_search()
