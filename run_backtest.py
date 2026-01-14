"""
Run Grid Search Backtest on Market Data.

Executes backtests across a grid of parameters and logs results to CSV.
"""
from datetime import time
from pathlib import Path
import pandas as pd
import itertools
from loguru import logger
import csv
import os

from sim_search.backtester import BacktestConfig, Backtester
from sim_search.times import set_default_tz

RESULTS_FILE = "backtest_results.csv"

def run_grid_search():
    # Load Data Once
    data_path = 'data/test/NQ_2024-09-06_2025-09-13.parquet'
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Define Parameter Grid
    grid = {
        'window_len': [30, 60],
        'n_neighbors': [5, 10],
        'distance_metric': ['euclidean', 'dtw'],
        'step_size': [5]  # Step size for sliding window (speed up vs density)
    }
    
    # Generate combinations
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    logger.info(f"Generated {len(combinations)} combinations to test.")
    
    # Initialize CSV if not exists key
    file_exists = os.path.isfile(RESULTS_FILE)
    fieldnames = [
        'Market', 'WindowSize', 'StepSize', 'Metric', 'K', 
        'Avg_RMSE', 'Avg_MAE', 'Hit_Ratio', 'Dir_Accuracy', 
        'Profit_Factor', 'Sharpe', 'Coverage_80'
    ]
    
    if not file_exists:
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    # Run Loop
    for i, params in enumerate(combinations):
        logger.info(f"Running config {i+1}/{len(combinations)}: {params}")
        
        config = BacktestConfig(
            data_path=data_path,
            forecast_horizon=20,
            min_train_windows=30,
            tp_threshold=0.005,
            sl_threshold=0.005,
            **params
        )
        
        backtester = Backtester(config)
        result = backtester.run(df)
        
        # Dictionary for CSV
        row = result.to_csv_dict()
        
        # Log to Console
        logger.success(
            f"Result: HitRatio={row['Hit_Ratio']:.1%} | PF={row['Profit_Factor']:.2f} | "
            f"Acc={row['Dir_Accuracy']:.1%}"
        )
        
        # Append to CSV
        with open(RESULTS_FILE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            
    logger.info(f"Grid Search Complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_grid_search()
