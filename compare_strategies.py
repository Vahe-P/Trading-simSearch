#!/usr/bin/env python3
"""
A/B Test: Baseline vs Regime-Aware Similarity Search

Compares:
1. BASELINE: KNN with DTW on all windows (current approach)
2. REGIME-AWARE: Regime filter (GK vol) + KNN with WDTW on same-regime windows

Metrics compared:
- Direction Accuracy: % of forecasts with correct direction
- Sharpe Ratio: Risk-adjusted returns
- Profit Factor: Gross wins / Gross losses
- Hit Ratio: % that hit TP before SL
- Coverage 80: % of actuals within p20-p80 band
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

from sim_search.backtester import BacktestConfig, Backtester
from sim_search.times import set_default_tz

# Configuration
DATA_PATH = 'data/test/NQ_2024-09-06_2025-09-13.parquet'
OUTPUT_DIR = Path('comparison_results')


def get_strategies():
    """
    Define strategies to compare.
    
    Focused comparison: Baseline DTW vs Regime-Aware WDTW
    """
    return {
        # BASELINE: Current approach - DTW on all windows
        'baseline_dtw': {
            'distance_metric': 'dtw',
            'use_regime_filter': False,
            'description': 'Baseline: DTW on all windows'
        },
        
        # NEW: Regime-aware with WDTW (the main improvement)
        'regime_wdtw': {
            'distance_metric': 'wdtw',
            'use_regime_filter': True,
            'vol_method': 'garman_klass',
            'description': 'Regime-Aware: GK vol filter + WDTW'
        },
        
        # Euclidean baseline (fast reference)
        'baseline_euclidean': {
            'distance_metric': 'euclidean',
            'use_regime_filter': False,
            'description': 'Baseline: Euclidean (fast)'
        },
    }


def run_comparison(data_path: str = DATA_PATH, verbose: bool = True):
    """
    Run A/B comparison of all strategies.
    
    Parameters
    ----------
    data_path : str
        Path to parquet file with market data
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    pd.DataFrame
        Comparison results
    """
    set_default_tz('America/New_York')
    
    # Load data once
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} bars")
    
    strategies = get_strategies()
    results = []
    
    # Common config
    # Note: Using larger step_size (60) to make backtest feasible
    # This creates ~6000 windows instead of 72000
    base_config = {
        'data_path': data_path,
        'window_len': 60,
        'step_size': 60,  # Non-overlapping windows for faster backtest
        'forecast_horizon': 20,
        'n_neighbors': 10,
        'min_train_windows': 50,
        'tp_threshold': 0.005,
        'sl_threshold': 0.005,
        'norm_method': 'log_returns',
        'max_test_days': 500,  # Limit test windows for faster comparison
    }
    
    for name, strategy_params in strategies.items():
        description = strategy_params.pop('description', name)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {description}")
        logger.info(f"{'='*60}")
        
        try:
            # Merge configs
            config_dict = {**base_config, **strategy_params}
            config = BacktestConfig(**config_dict)
            
            # Run backtest
            backtester = Backtester(config)
            result = backtester.run(df)
            
            # Collect metrics
            metrics = {
                'strategy': name,
                'description': description,
                **strategy_params,
                'total_trades': result.total_trades,
                'direction_accuracy': result.direction_accuracy,
                'hit_ratio': result.hit_ratio,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'avg_rmse': result.avg_rmse,
                'avg_mae': result.avg_mae,
                'coverage_80': result.avg_coverage_80,
                'max_drawdown': result.max_drawdown,
                'expectancy': result.expectancy,
            }
            results.append(metrics)
            
            logger.success(
                f"✓ {name}: Dir Acc={metrics['direction_accuracy']:.1%} | "
                f"Sharpe={metrics['sharpe_ratio']:.2f} | "
                f"PF={metrics['profit_factor']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({'strategy': name, 'error': str(e)})
        
        # Restore description for next iteration display
        strategy_params['description'] = description
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def print_comparison(results_df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80 + "\n")
    
    # Key metrics to display
    display_cols = [
        'strategy', 'direction_accuracy', 'sharpe_ratio', 
        'profit_factor', 'hit_ratio', 'avg_rmse'
    ]
    
    available_cols = [c for c in display_cols if c in results_df.columns]
    display_df = results_df[available_cols].copy()
    
    # Format percentages
    if 'direction_accuracy' in display_df.columns:
        display_df['direction_accuracy'] = display_df['direction_accuracy'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    if 'hit_ratio' in display_df.columns:
        display_df['hit_ratio'] = display_df['hit_ratio'].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    
    print(display_df.to_string(index=False))
    
    # Find best strategy
    if 'error' not in results_df.columns or results_df['error'].isna().all():
        valid_results = results_df[results_df.get('error', pd.Series([None]*len(results_df))).isna()]
        if len(valid_results) > 0 and 'direction_accuracy' in valid_results.columns:
            best_idx = valid_results['direction_accuracy'].idxmax()
            best = valid_results.loc[best_idx]
            print(f"\n🏆 BEST STRATEGY: {best['strategy']}")
            print(f"   Direction Accuracy: {best['direction_accuracy']:.1%}")
            print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"   Profit Factor: {best['profit_factor']:.2f}")


def save_results(results_df: pd.DataFrame, output_dir: Path = OUTPUT_DIR):
    """Save results to CSV."""
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"comparison_{timestamp}.csv"
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nResults saved to {output_path}")
    
    return output_path


def main():
    """Main entry point."""
    # Check if data exists
    if not Path(DATA_PATH).exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        logger.info("Please ensure you have the NQ data file in data/test/")
        sys.exit(1)
    
    # Run comparison
    results_df = run_comparison()
    
    # Print results
    print_comparison(results_df)
    
    # Save results
    save_results(results_df)
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)


if __name__ == "__main__":
    main()
