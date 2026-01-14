"""
Best Model Selection - Milestone 3.4.

Automates the selection of the optimal model configuration based on
grid search results. Saves the "Golden Config" for production use.
"""
from pathlib import Path
import pandas as pd
from datetime import time

from sim_search.optimizer import GridSearch, ModelSelector
from sim_search.reporting import BacktestReport
from sim_search.backtester import BacktestResult

def load_results_and_select():
    """Load results from CSV or run a quick grid search if none found."""
    
    # Check for existing reports
    report_dir = Path("reports")
    csv_files = sorted(report_dir.glob("grid_search_optimization_comparison_*.csv"))
    
    results = []
    
    if csv_files:
        # Load the latest report
        latest_csv = csv_files[-1]
        print(f"Loading results from {latest_csv}...")
        df = pd.read_csv(latest_csv)
        
        # Convert DataFrame back to BacktestResult objects (approximate reconstruction)
        # Note: In a real system, we might pickle results or store richer data.
        # For selection, we mainly need the dicts, but ModelSelector expects BacktestResult objects.
        # We'll create dummy objects populated with the dict data.
        
        for _, row in df.iterrows():
            # Extract config
            config_dict = row.to_dict()
            
            # Create dummy result with populated metrics
            res = BacktestResult(config=config_dict)
            res.total_trades = int(row['total_trades'])
            res.profit_factor = float(row['profit_factor'])
            res.hit_ratio = float(row['hit_ratio'])
            res.sharpe_ratio = float(row['sharpe_ratio']) if not pd.isna(row['sharpe_ratio']) else 0.0
            results.append(res)
            
    else:
        print("No existing reports found. Running a quick grid search...")
        # Run a small grid search just to have something
        base_config = {
            'data_path': 'data/test/NQ_2024-09-06_2025-09-13.parquet',
            'window_start_time': time(8, 0),
            'window_end_time': time(9, 30),
            'extend_sessions': 1,
            'min_train_windows': 30,
            'timezone': 'America/New_York',
            'resample': '5min',
        }
        param_grid = {
            'n_neighbors': [5, 10], 
            'forecast_horizon': [10, 20]
        }
        optimizer = GridSearch(base_config, param_grid)
        report = optimizer.run()
        results = report.results

    # Select Best Model
    selector = ModelSelector(results)
    
    print("\n" + "=" * 60)
    print("MODEL SELECTION RANKING")
    print("=" * 60)
    
    # Rank models
    ranked = selector.rank_models(min_trades=10, min_profit_factor=0.5, metric='weighted_score')
    
    # Display top 5
    cols_to_show = ['profit_factor', 'hit_ratio', 'sharpe_ratio', 'n_neighbors', 'forecast_horizon', 'weighted_score']
    # Filter only columns that exist
    cols_to_show = [c for c in cols_to_show if c in ranked.columns]
    
    print(ranked[cols_to_show].head(5).to_string())
    
    # Save Golden Config
    print("\n" + "=" * 60)
    best_config = selector.save_best_config("golden_config.json", min_trades=10, metric='weighted_score')
    print("=" * 60)
    print("Golden Config Saved:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    return best_config

if __name__ == "__main__":
    load_results_and_select()
