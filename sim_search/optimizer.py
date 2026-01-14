"""
Parameter Grid Search Optimization.

Provides parallel execution of backtests across a grid of parameters
to identify optimal settings for market similarity search.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
from loguru import logger
import pandas as pd
import json
import numpy as np

from .backtester import BacktestConfig, BacktestResult, run_backtest_from_file
from .reporting import BacktestReport


def _run_single_config(config: BacktestConfig) -> BacktestResult:
    """
    Worker function for parallel execution.
    Must be top-level for pickling.
    """
    try:
        # Disable logging in workers to avoid clutter/conflicts
        # (Optional, depending on logging setup)
        return run_backtest_from_file(config.data_path, config=config, verbose=False)
    except Exception as e:
        logger.error(f"Backtest failed for {config}: {e}")
        # Return empty/failed result? Or raise?
        # Better to return a result object indicating failure or let it fail
        raise e


@dataclass
class GridSearch:
    """
    Executor for parameter grid search optimization.
    
    Attributes
    ----------
    base_config : Dict[str, Any]
        Base configuration parameters common to all runs
    param_grid : Dict[str, List[Any]]
        Dictionary where keys are parameter names and values are lists of settings to try
    max_workers : int
        Number of parallel processes (default: None = CPU count)
    """
    base_config: Dict[str, Any]
    param_grid: Dict[str, List[Any]]
    max_workers: Optional[int] = None
    
    def run(self) -> BacktestReport:
        """
        Execute the grid search.
        
        Returns
        -------
        BacktestReport
            Aggregated results
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))
        
        logger.info(f"Starting Grid Search: {len(combinations)} configurations")
        logger.info(f"Parameters: {keys}")
        
        configs = []
        for combo in combinations:
            # Create config dictionary
            params = dict(zip(keys, combo))
            full_params = {**self.base_config, **params}
            
            # Create BacktestConfig object
            config = BacktestConfig(**full_params)
            configs.append(config)
        
        report = BacktestReport(name="grid_search_optimization")
        start_time = time.time()
        
        # Parallel execution
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_config = {
                executor.submit(_run_single_config, cfg): cfg 
                for cfg in configs
            }
            
            completed_count = 0
            total_count = len(configs)
            
            for future in as_completed(future_to_config):
                cfg = future_to_config[future]
                try:
                    result = future.result()
                    report.add_result(result)
                    
                    completed_count += 1
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_count
                    remaining = avg_time * (total_count - completed_count)
                    
                    logger.info(
                        f"[{completed_count}/{total_count}] "
                        f"PF: {result.profit_factor:.2f} | "
                        f"Acc: {result.direction_accuracy:.1%} | "
                        f"Params: {self._get_varying_params(cfg)}"
                    )
                    
                except Exception as e:
                    logger.error(f"Job failed for config {cfg}: {e}")
        
        total_time = time.time() - start_time
        logger.success(f"Grid Search complete in {total_time:.1f}s")
        
        return report

    def _get_varying_params(self, config: BacktestConfig) -> str:
        """Helper to format string of only the parameters that vary."""
        varying = []
        for key in self.param_grid.keys():
            val = getattr(config, key)
            varying.append(f"{key}={val}")
        return ", ".join(varying)


@dataclass
class ModelSelector:
    """
    Selects the best model from backtest results based on performance metrics.
    """
    results: List[BacktestResult]
    
    def rank_models(self, 
                   min_trades: int = 20,
                   min_profit_factor: float = 1.0,
                   metric: str = 'weighted_score') -> pd.DataFrame:
        """
        Rank models based on criteria.
        
        Parameters
        ----------
        min_trades : int
            Minimum number of trades required to be considered
        min_profit_factor : float
            Minimum profit factor requirement
        metric : str
            Metric to sort by ('profit_factor', 'sharpe_ratio', 'weighted_score')
            
        Returns
        -------
        pd.DataFrame
            Ranked results dataframe
        """
        # Convert to DataFrame for easier handling
        data = [r.to_dict() for r in self.results]
        df = pd.DataFrame(data)
        
        if df.empty:
            return df
            
        # Filter
        df_filtered = df[
            (df['total_trades'] >= min_trades) & 
            (df['profit_factor'] >= min_profit_factor)
        ].copy()
        
        if df_filtered.empty:
            logger.warning("No models met the filtering criteria! Returning all models sorted by PF.")
            df_filtered = df.copy()
        
        # Calculate Weighted Score if requested
        # Score = (Profit Factor * 0.4) + (Hit Ratio * 3 * 0.3) + (Sharpe * 0.3)
        # Note: Scaled hit ratio * 3 to be comparable (0.6 * 3 = 1.8)
        if metric == 'weighted_score':
            df_filtered['weighted_score'] = (
                (df_filtered['profit_factor'] * 0.4) + 
                (df_filtered['hit_ratio'] * 2.0 * 0.3) + 
                (df_filtered['sharpe_ratio'].fillna(0) * 0.3)
            )
        
        # Sort
        return df_filtered.sort_values(metric, ascending=False)

    def get_best_config(self, **kwargs) -> Dict[str, Any]:
        """Get the configuration of the best ranked model."""
        ranked = self.rank_models(**kwargs)
        if ranked.empty:
            raise ValueError("No results available to select from.")
        
        best = ranked.iloc[0].to_dict()
        
        # Remove metric columns to strip down to config
        metric_cols = [
            'total_trades', 'hit_ratio', 'direction_accuracy', 'avg_rmse', 'avg_mae',
            'avg_coverage_80', 'profit_factor', 'avg_mfe', 'avg_mae_excursion', 
            'mfe_mae_ratio', 'sharpe_ratio', 'max_drawdown', 'long_win_rate', 
            'short_win_rate', 'avg_win', 'avg_loss', 'expectancy', 'weighted_score'
        ]
        config = {k: v for k, v in best.items() if k not in metric_cols}
        return config
    
    def save_best_config(self, output_path: str = "best_model_config.json", **kwargs):
        """Save best config to JSON file."""
        config = self.get_best_config(**kwargs)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=4, default=str)
        logger.info(f"Saved best model configuration to {output_path}")
        return config
