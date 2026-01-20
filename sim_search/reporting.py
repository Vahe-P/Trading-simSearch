"""
CSV Reporting System for Market Similarity Search Backtests.

Generates structured CSV reports for:
- Single backtest results with all trades
- Multi-backtest comparison (grid search results)
- Summary statistics
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .backtester import BacktestResult, TradeResult


def export_backtest_summary(
    result: BacktestResult,
    output_path: str = "backtest_summary.csv",
    include_config: bool = True
) -> str:
    """
    Export a single backtest summary to CSV.
    
    Parameters
    ----------
    result : BacktestResult
        Backtest result to export
    output_path : str
        Output file path
    include_config : bool
        Whether to include config parameters in output
    
    Returns
    -------
    str
        Path to the generated CSV file
    """
    data = result.to_dict()
    df = pd.DataFrame([data])
    df.to_csv(output_path, index=False)
    return output_path


def export_trades_detail(
    result: BacktestResult,
    output_path: str = "trades_detail.csv"
) -> str:
    """
    Export all individual trades to CSV with detailed metrics.
    
    Parameters
    ----------
    result : BacktestResult
        Backtest result containing trades
    output_path : str
        Output file path
    
    Returns
    -------
    str
        Path to the generated CSV file
    """
    trades_data = []
    for i, trade in enumerate(result.trades):
        trades_data.append({
            'symbol': result.symbol,  # Include symbol for multi-asset support
            'trade_id': i + 1,
            'cutoff': trade.cutoff,
            'direction': 'Long' if np.sum(trade.forecast_returns) > 0 else 'Short',
            'direction_correct': trade.direction_correct,
            'hit_tp': trade.hit_tp if trade.hit_tp is not None else 'N/A',
            'rmse': trade.rmse,
            'mae': trade.mae,
            'mfe': trade.mfe,
            'mfe_bar': trade.mfe_bar,
            'mae_excursion': trade.mae_excursion,
            'mae_bar': trade.mae_bar,
            'avg_neighbor_mfe': trade.avg_neighbor_mfe,
            'avg_neighbor_mae': trade.avg_neighbor_mae,
            'e_ratio': trade.e_ratio,
            'cum_return': np.sum(trade.actual_returns),
            'avg_neighbor_distance': np.mean(trade.neighbor_distances),
        })
    
    df = pd.DataFrame(trades_data)
    df.to_csv(output_path, index=False)
    return output_path


def export_comparison_report(
    results: list[BacktestResult],
    output_path: str = "backtest_comparison.csv",
    sort_by: str = "profit_factor",
    ascending: bool = False
) -> str:
    """
    Export multiple backtest results for comparison (grid search).
    
    Parameters
    ----------
    results : list[BacktestResult]
        List of backtest results to compare
    output_path : str
        Output file path
    sort_by : str
        Column to sort by (default: profit_factor)
    ascending : bool
        Sort order (default: False = descending)
    
    Returns
    -------
    str
        Path to the generated CSV file
    """
    data = [r.to_dict() for r in results]
    df = pd.DataFrame(data)
    
    # Reorder columns for readability - symbol first for multi-asset support
    priority_cols = [
        'symbol', 'total_trades', 'direction_accuracy', 'hit_ratio', 'profit_factor',
        'sharpe_ratio', 'max_drawdown', 'expectancy', 
        'avg_mfe', 'avg_mae_excursion', 'mfe_mae_ratio',
        'avg_neighbor_mfe', 'avg_neighbor_mae', 'avg_e_ratio',
        'long_win_rate', 'short_win_rate', 'avg_win', 'avg_loss'
    ]
    
    # Get config columns
    config_cols = [c for c in df.columns if c not in priority_cols]
    
    # Reorder
    ordered_cols = [c for c in priority_cols if c in df.columns] + config_cols
    df = df[ordered_cols]
    
    # Sort
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)
    
    df.to_csv(output_path, index=False)
    return output_path


@dataclass
class BacktestReport:
    """
    Manages backtest reporting with multiple export options.
    
    Example
    -------
    >>> report = BacktestReport()
    >>> report.add_result(result1)
    >>> report.add_result(result2)
    >>> report.export_all("./reports/")
    """
    results: list[BacktestResult] = field(default_factory=list)
    name: str = "backtest_report"
    
    def add_result(self, result: BacktestResult):
        """Add a backtest result to the report."""
        self.results.append(result)
    
    def export_all(self, output_dir: str = ".") -> dict:
        """
        Export all reports to the specified directory.
        
        Returns dict with paths to generated files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        paths = {}
        
        if len(self.results) == 1:
            # Single backtest - export summary and trades
            paths['summary'] = export_backtest_summary(
                self.results[0],
                str(output_dir / f"{self.name}_summary_{timestamp}.csv")
            )
            paths['trades'] = export_trades_detail(
                self.results[0],
                str(output_dir / f"{self.name}_trades_{timestamp}.csv")
            )
        else:
            # Multiple backtests - export comparison
            paths['comparison'] = export_comparison_report(
                self.results,
                str(output_dir / f"{self.name}_comparison_{timestamp}.csv")
            )
            
            # Also export trades for best result
            best = max(self.results, key=lambda r: r.profit_factor)
            paths['best_trades'] = export_trades_detail(
                best,
                str(output_dir / f"{self.name}_best_trades_{timestamp}.csv")
            )
        
        return paths
    
    def print_summary(self):
        """Print a summary of all results."""
        if not self.results:
            print("No results to display.")
            return
        
        # Get unique symbols tested
        symbols = list(set(r.symbol for r in self.results))
        
        print(f"\n{'='*70}")
        print(f"BACKTEST REPORT: {self.name}")
        print(f"{'='*70}")
        print(f"Total configurations tested: {len(self.results)}")
        print(f"Symbols: {', '.join(symbols)}")
        
        if len(self.results) > 1:
            # Show best by different metrics
            best_pf = max(self.results, key=lambda r: r.profit_factor)
            best_dir = max(self.results, key=lambda r: r.direction_accuracy)
            best_sharpe = max(self.results, key=lambda r: r.sharpe_ratio)
            
            print(f"\nBest by Profit Factor: {best_pf.profit_factor:.2f} ({best_pf.symbol})")
            print(f"Best by Direction Accuracy: {best_dir.direction_accuracy:.1%} ({best_dir.symbol})")
            print(f"Best by Sharpe Ratio: {best_sharpe.sharpe_ratio:.2f} ({best_sharpe.symbol})")
        
        print(f"{'='*70}\n")
