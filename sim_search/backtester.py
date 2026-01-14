"""
Walk-Forward Backtester for Market Similarity Search.

OPTIMIZED VERSION - O(n) complexity instead of O(n²).

Simulates trading over historical data by:
1. Pre-compute ALL normalized panels ONCE before the loop
2. At each test point, SLICE pre-computed arrays (no re-normalization)
3. Generate forecast using nearest neighbors
4. Compare forecast to actual outcomes
5. Track performance metrics (hit ratio, accuracy, etc.)
"""

from dataclasses import dataclass, field, asdict
from datetime import time, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .config import ForecastConfig
from .forecaster import (
    prepare_panel_data, similarity_search, 
    forecast_from_neighbors, score_forecast, calculate_forecast_percentiles
)
from .windowing import partition_sliding
from .times import set_default_tz, resample


@dataclass
class TradeResult:
    """Result of a single trade/forecast."""
    cutoff: pd.Timestamp  # Forecast origin
    forecast_returns: np.ndarray  # Predicted returns
    actual_returns: np.ndarray  # Actual returns
    percentile_bands: dict  # p20, p50, p80
    neighbor_distances: np.ndarray  # Distances to neighbors
    
    # Calculated metrics
    rmse: float = 0.0
    mae: float = 0.0
    direction_correct: bool = False
    hit_tp: Optional[bool] = None
    coverage_80: float = 0.0  # % of actual points within p20-p80 band
    
    # Excursion Analysis
    mfe: float = 0.0  
    mfe_bar: int = 0  
    mae_excursion: float = 0.0  
    mae_bar: int = 0  
    
    def __post_init__(self):
        """Calculate metrics after initialization."""
        if len(self.forecast_returns) > 0 and len(self.actual_returns) > 0:
            self.rmse = np.sqrt(np.mean((self.forecast_returns - self.actual_returns) ** 2))
            self.mae = np.mean(np.abs(self.forecast_returns - self.actual_returns))
            
            # Direction accuracy
            forecast_cum = np.sum(self.forecast_returns)
            actual_cum = np.sum(self.actual_returns)
            self.direction_correct = (forecast_cum > 0) == (actual_cum > 0)
            
            # Coverage 80 (percentage of points inside p20-p80)
            p20 = self.percentile_bands['p20']
            p80 = self.percentile_bands['p80']
            inside = (self.actual_returns >= p20) & (self.actual_returns <= p80)
            self.coverage_80 = np.mean(inside)
            
            # Calculate MFE/MAE
            self._calculate_excursions()
    
    def _calculate_excursions(self):
        cum_returns = np.cumsum(self.actual_returns)
        forecast_direction = np.sum(self.forecast_returns) > 0
        
        if forecast_direction:  # Bullish trade
            self.mfe = np.max(cum_returns)
            self.mfe_bar = int(np.argmax(cum_returns))
            self.mae_excursion = np.min(cum_returns)
            self.mae_bar = int(np.argmin(cum_returns))
        else:  # Bearish trade
            self.mfe = -np.min(cum_returns)
            self.mfe_bar = int(np.argmin(cum_returns))
            self.mae_excursion = -np.max(cum_returns)
            self.mae_bar = int(np.argmax(cum_returns))


@dataclass
class BacktestResult:
    """Aggregated results from walk-forward backtest."""
    config: dict
    trades: List[TradeResult] = field(default_factory=list)
    symbol: str = "UNKNOWN"  # Added symbol support
    
    # Aggregate metrics
    total_trades: int = 0
    hit_ratio: float = 0.0
    direction_accuracy: float = 0.0
    avg_rmse: float = 0.0
    avg_mae: float = 0.0
    avg_coverage_80: float = 0.0
    profit_factor: float = 0.0
    
    # Excursion Analysis
    avg_mfe: float = 0.0
    avg_mae_excursion: float = 0.0
    mfe_mae_ratio: float = 0.0
    
    # Performance
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    
    def calculate_metrics(self):
        if not self.trades:
            return
            
        self.total_trades = len(self.trades)
        
        # Simple aggregates
        self.direction_accuracy = np.mean([t.direction_correct for t in self.trades])
        self.avg_rmse = np.mean([t.rmse for t in self.trades])
        self.avg_mae = np.mean([t.mae for t in self.trades])
        self.avg_coverage_80 = np.mean([t.coverage_80 for t in self.trades])
        
        # Hit Ratio
        trades_with_tp = [t for t in self.trades if t.hit_tp is not None]
        if trades_with_tp:
            self.hit_ratio = sum(1 for t in trades_with_tp if t.hit_tp) / len(trades_with_tp)
            
        # PnL Analysis
        wins = []
        losses = []
        trade_returns = []
        
        for t in self.trades:
            acc_ret = np.sum(t.actual_returns)
            # Adjust for direction
            is_long = np.sum(t.forecast_returns) > 0
            pnl = acc_ret if is_long else -acc_ret
            trade_returns.append(pnl)
            
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(abs(pnl))
                
        total_wins = sum(wins)
        total_losses = sum(losses) if losses else 1e-10
        self.profit_factor = total_wins / total_losses
        
        self.avg_win = np.mean(wins) if wins else 0.0
        self.avg_loss = np.mean(losses) if losses else 0.0
        
        win_rate = len(wins) / self.total_trades
        loss_rate = len(losses) / self.total_trades
        self.expectancy = (win_rate * self.avg_win) - (loss_rate * self.avg_loss)
        
        # Excursion
        self.avg_mfe = np.mean([t.mfe for t in self.trades])
        self.avg_mae_excursion = np.mean([t.mae_excursion for t in self.trades])
        self.mfe_mae_ratio = abs(self.avg_mfe / self.avg_mae_excursion) if self.avg_mae_excursion != 0 else 0.0
        
        # Sharpe (Annualized assuming daily)
        if len(trade_returns) > 1:
            ret_std = np.std(trade_returns, ddof=1)
            if ret_std > 0:
                self.sharpe_ratio = (np.mean(trade_returns) / ret_std) * np.sqrt(252)
                
        # Drawdown
        equity = np.cumsum(trade_returns)
        running_max = np.maximum.accumulate(equity)
        dd = running_max - equity
        self.max_drawdown = np.max(dd) if len(dd) > 0 else 0.0
        
        # Split win rates
        longs = [t for t in self.trades if np.sum(t.forecast_returns) > 0]
        shorts = [t for t in self.trades if np.sum(t.forecast_returns) <= 0]
        
        if longs:
            self.long_win_rate = np.mean([t.direction_correct for t in longs])
        if shorts:
            self.short_win_rate = np.mean([t.direction_correct for t in shorts])

    def to_csv_dict(self) -> dict:
        """Flattened dictionary for CSV logging."""
        d = {
            # Config - now uses symbol from instance
            'Symbol': self.symbol,
            'WindowSize': self.config.get('window_len'),
            'StepSize': self.config.get('step_size'),
            'Metric': self.config.get('distance_metric'),
            'K': self.config.get('n_neighbors'),
            
            # Key Metrics
            'Avg_RMSE': self.avg_rmse,
            'Avg_MAE': self.avg_mae,
            'Hit_Ratio': self.hit_ratio,
            'Dir_Accuracy': self.direction_accuracy,
            'Profit_Factor': self.profit_factor,
            'Sharpe': self.sharpe_ratio,
            'Coverage_80': self.avg_coverage_80
        }
        return d
    
    def to_dict(self) -> dict:
        """Full dictionary representation for DataFrame export."""
        return {
            # Config
            'symbol': self.symbol,
            'window_len': self.config.get('window_len'),
            'step_size': self.config.get('step_size'),
            'distance_metric': self.config.get('distance_metric'),
            'n_neighbors': self.config.get('n_neighbors'),
            'forecast_horizon': self.config.get('forecast_horizon'),
            'norm_method': self.config.get('norm_method'),
            
            # Metrics
            'total_trades': self.total_trades,
            'hit_ratio': self.hit_ratio,
            'direction_accuracy': self.direction_accuracy,
            'avg_rmse': self.avg_rmse,
            'avg_mae': self.avg_mae,
            'avg_coverage_80': self.avg_coverage_80,
            'profit_factor': self.profit_factor,
            'avg_mfe': self.avg_mfe,
            'avg_mae_excursion': self.avg_mae_excursion,
            'mfe_mae_ratio': self.mfe_mae_ratio,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'long_win_rate': self.long_win_rate,
            'short_win_rate': self.short_win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'expectancy': self.expectancy,
        }


@dataclass
class BacktestConfig:
    """Configuration for Backtester."""
    data_path: str
    
    # Symbol identification
    symbol: str = "UNKNOWN"
    
    # Window settings
    window_len: int = 60
    step_size: int = 1
    
    # Forecast settings
    forecast_horizon: int = 20
    n_neighbors: int = 5
    distance_metric: str = 'euclidean'
    norm_method: str = 'log_returns'
    
    # Test settings
    min_train_windows: int = 30
    tp_threshold: float = 0.01
    sl_threshold: float = 0.01
    max_test_days: Optional[int] = None  # Limit test windows for faster runs
    
    # Time-anchored window settings (optional)
    window_start_time: Optional[time] = None
    window_end_time: Optional[time] = None
    extend_sessions: int = 0
    
    resample: str = ''
    timezone: str = 'America/New_York'


class Backtester:
    """
    Optimized Walk-Forward Backtester.
    
    Key optimization: Pre-computes ALL normalized panels ONCE before
    the backtest loop, then slices arrays inside the loop.
    
    Complexity: O(n) instead of O(n²)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def run(self, df: pd.DataFrame) -> BacktestResult:
        """
        Run walk-forward backtest with pre-computed panels.
        
        Parameters
        ----------
        df : pd.DataFrame
            Market data with DatetimeIndex and 'close' column
            
        Returns
        -------
        BacktestResult
            Aggregated backtest results
        """
        set_default_tz(self.config.timezone)
        
        # =====================================================================
        # STEP 1: Create all sliding windows ONCE
        # =====================================================================
        logger.info(f"Creating sliding windows (len={self.config.window_len}, step={self.config.step_size})...")
        
        windows = partition_sliding(
            df, 
            window_len=self.config.window_len, 
            step_size=self.config.step_size,
            horizon_len=self.config.forecast_horizon
        )
        
        n_windows = len(windows)
        if n_windows < self.config.min_train_windows + 1:
            logger.error(f"Not enough windows for backtest: {n_windows} < {self.config.min_train_windows + 1}")
            return BacktestResult(config=asdict(self.config), symbol=self.config.symbol)
        
        logger.info(f"Created {n_windows} windows")
        
        # =====================================================================
        # STEP 2: PRE-COMPUTE ALL PANELS ONCE (THE KEY OPTIMIZATION)
        # =====================================================================
        logger.info("Pre-computing normalized panels for ALL windows (one-time cost)...")
        
        x_all, y_all, labels_all = prepare_panel_data(
            df, 
            windows,
            feature_col='close',
            horizon_len=self.config.forecast_horizon,
            norm_method=self.config.norm_method
        )
        
        # Convert x_all list to numpy array for efficient slicing
        # x_all is List[DataFrame], each shape (1, window_len)
        # Stack into (n_windows, 1, window_len) for efficient indexing
        x_array = np.stack([x.to_numpy() for x in x_all], axis=0)  # Shape: (n_windows, 1, window_len)
        y_array = y_all.to_numpy()  # Shape: (n_windows, horizon_len)
        
        logger.info(f"Pre-computed arrays: X={x_array.shape}, Y={y_array.shape}")
        
        # =====================================================================
        # STEP 3: Initialize result object
        # =====================================================================
        result = BacktestResult(config=asdict(self.config), symbol=self.config.symbol)
        
        # =====================================================================
        # STEP 4: Determine test range
        # =====================================================================
        test_start = self.config.min_train_windows
        test_end = n_windows
        
        # Optionally limit test windows for faster runs
        if self.config.max_test_days is not None:
            test_end = min(test_end, test_start + self.config.max_test_days)
        
        test_indices = range(test_start, test_end)
        logger.info(f"Running backtest on {len(test_indices)} test windows...")
        
        # =====================================================================
        # STEP 5: Walk-forward loop (NOW O(n) - just slicing!)
        # =====================================================================
        for i in tqdm(test_indices, desc="Backtesting"):
            try:
                # SLICE pre-computed arrays (O(1) operation)
                x_train = x_array[:i]  # All windows before current
                y_train = y_array[:i]
                x_test = x_array[i:i+1]  # Current window (keep dim for kneighbors)
                y_test = y_array[i]
                test_cutoff = labels_all[i]
                
                # Skip if not enough training data
                if len(x_train) < self.config.n_neighbors:
                    continue
                
                # Similarity Search
                # Note: KNN expects list of DataFrames, so convert back
                x_train_list = [pd.DataFrame(x) for x in x_train]
                x_test_df = pd.DataFrame(x_test[0])
                
                idx, dists = similarity_search(
                    x_train_list, 
                    np.zeros(len(x_train_list)),  # Dummy y for fitting
                    x_test_df,
                    n_neighbors=self.config.n_neighbors,
                    impl='knn',
                    distance=self.config.distance_metric
                )
                
                # Forecast from neighbors
                neighbor_horizons = y_train[idx]
                forecast = forecast_from_neighbors(neighbor_horizons, dists, impl='avg')
                bands = calculate_forecast_percentiles(neighbor_horizons)
                
                # Check TP/SL outcome
                hit_tp = _check_tp_sl(y_test, forecast, self.config.tp_threshold, self.config.sl_threshold)
                
                # Record trade
                trade = TradeResult(
                    cutoff=test_cutoff,
                    forecast_returns=forecast,
                    actual_returns=y_test,
                    percentile_bands=bands,
                    neighbor_distances=dists,
                    hit_tp=hit_tp
                )
                result.trades.append(trade)
                
            except Exception as e:
                logger.debug(f"Error at window {i}: {e}")
                continue
        
        # =====================================================================
        # STEP 6: Calculate aggregate metrics
        # =====================================================================
        result.calculate_metrics()
        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Dir Acc: {result.direction_accuracy:.1%}, "
                   f"PF: {result.profit_factor:.2f}")
        
        return result


def run_backtest_from_file(
    data_path: str,
    config: Optional[BacktestConfig] = None,
    verbose: bool = True
) -> BacktestResult:
    """
    Convenience function to run backtest from a data file.
    
    Parameters
    ----------
    data_path : str
        Path to parquet file with market data
    config : BacktestConfig, optional
        Backtest configuration (will use data_path if not provided)
    verbose : bool
        Whether to show progress bars and logs
        
    Returns
    -------
    BacktestResult
        Backtest results
    """
    if config is None:
        config = BacktestConfig(data_path=data_path)
    
    # Auto-detect symbol from filename if not set
    if config.symbol == "UNKNOWN":
        path = Path(data_path)
        # Try to extract symbol from filename like "NQ_2024-09-06_2025-09-13.parquet"
        name_parts = path.stem.split('_')
        if name_parts:
            config.symbol = name_parts[0]
    
    # Suppress logging if not verbose
    if not verbose:
        logger.disable("sim_search")
    
    try:
        # Load data
        df = pd.read_parquet(config.data_path)
        
        # Resample if configured
        if config.resample:
            df = resample(df, config.resample)
        
        # Run backtest
        backtester = Backtester(config)
        result = backtester.run(df)
        
    finally:
        if not verbose:
            logger.enable("sim_search")
    
    return result


def _check_tp_sl(actual, forecast, tp, sl):
    """
    Check if take-profit or stop-loss was hit.
    
    Parameters
    ----------
    actual : np.ndarray
        Actual returns
    forecast : np.ndarray
        Forecasted returns (used to determine direction)
    tp : float
        Take-profit threshold (cumulative return)
    sl : float
        Stop-loss threshold (cumulative return)
        
    Returns
    -------
    bool or None
        True if TP hit, False if SL hit, None if neither
    """
    direction = np.sum(forecast) > 0
    cum = np.cumsum(actual)
    
    for ret in cum:
        if direction:  # Long trade
            if ret >= tp:
                return True
            if ret <= -sl:
                return False
        else:  # Short trade
            if ret <= -tp:
                return True
            if ret >= sl:
                return False
    
    return None
