"""
Walk-Forward Backtester for Market Similarity Search.

OPTIMIZED VERSION - O(n) complexity instead of O(n²).

Simulates trading over historical data by:
1. Pre-compute ALL normalized panels ONCE before the loop
2. At each test point, SLICE pre-computed arrays (no re-normalization)
3. Generate forecast using nearest neighbors
4. Compare forecast to actual outcomes
5. Track performance metrics (hit ratio, accuracy, etc.)

Transaction costs can be applied via the TransactionCosts config.
"""

from dataclasses import dataclass, field, asdict
from datetime import time, datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from .config import ForecastConfig
from .costs import TransactionCosts, NO_COSTS, get_preset
from .forecaster import (
    prepare_panel_data, similarity_search, regime_aware_similarity_search,
    forecast_from_neighbors, score_forecast, calculate_forecast_percentiles,
    calculate_excursion_metrics_per_neighbor
)
from .volatility import REGIME_NAMES, log_regime_distribution
from .windowing import partition_sliding
from .times import set_default_tz, resample


@dataclass
class TradeResult:
    """Result of a single trade/forecast."""
    cutoff: pd.Timestamp  # Forecast origin
    forecast_returns: np.ndarray  # Predicted returns
    actual_returns: np.ndarray  # Actual returns (gross, before costs)
    percentile_bands: dict  # p20, p50, p80
    neighbor_distances: np.ndarray  # Distances to neighbors
    
    # Regime information
    regime: int = -1  # Query regime: 0=LOW, 1=MED, 2=HIGH, -1=not computed
    same_regime_count: int = 0  # How many neighbors from same regime
    
    # Transaction costs (NEW)
    entry_price: float = 0.0  # Price at entry for cost calculation
    transaction_cost: float = 0.0  # Round-trip cost in dollars
    cost_as_return: float = 0.0  # Cost expressed as return decimal
    net_returns: Optional[np.ndarray] = None  # Actual returns AFTER costs
    
    # Calculated metrics
    rmse: float = 0.0
    mae: float = 0.0
    direction_correct: bool = False
    hit_tp: Optional[bool] = None
    coverage_80: float = 0.0  # % of actual points within p20-p80 band
    
    # P&L metrics (gross and net)
    gross_pnl: float = 0.0  # P&L before costs (in return terms)
    net_pnl: float = 0.0  # P&L after costs (in return terms)
    
    # Excursion Analysis (from actual returns - legacy)
    mfe: float = 0.0  
    mfe_bar: int = 0  
    mae_excursion: float = 0.0  
    mae_bar: int = 0
    
    # Excursion Analysis (per-neighbor from OHLC - NEW)
    neighbor_mfe_values: Optional[np.ndarray] = None
    neighbor_mae_values: Optional[np.ndarray] = None
    avg_neighbor_mfe: float = 0.0
    avg_neighbor_mae: float = 0.0
    e_ratio: float = 0.0  
    
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
            
            # Calculate P&L (gross)
            is_long = forecast_cum > 0
            self.gross_pnl = actual_cum if is_long else -actual_cum
            
            # Calculate net P&L (after costs)
            if self.net_returns is not None:
                net_cum = np.sum(self.net_returns)
                self.net_pnl = net_cum if is_long else -net_cum
            else:
                # No costs applied, net = gross
                self.net_pnl = self.gross_pnl - self.cost_as_return
            
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
    
    # Excursion Analysis (from actual returns - legacy)
    avg_mfe: float = 0.0
    avg_mae_excursion: float = 0.0
    mfe_mae_ratio: float = 0.0
    
    # Excursion Analysis (per-neighbor from OHLC - NEW)
    avg_neighbor_mfe: float = 0.0
    avg_neighbor_mae: float = 0.0
    avg_e_ratio: float = 0.0
    
    # Performance (GROSS - before costs)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    expectancy: float = 0.0
    
    # Performance (NET - after costs)
    total_costs: float = 0.0  # Total transaction costs in dollars
    net_profit_factor: float = 0.0
    net_sharpe_ratio: float = 0.0
    net_max_drawdown: float = 0.0
    net_avg_win: float = 0.0
    net_avg_loss: float = 0.0
    net_expectancy: float = 0.0
    
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
        
        # Total transaction costs
        self.total_costs = sum(t.transaction_cost for t in self.trades)
            
        # PnL Analysis (GROSS - before costs)
        wins = []
        losses = []
        trade_returns = []
        
        for t in self.trades:
            pnl = t.gross_pnl
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
        
        # PnL Analysis (NET - after costs)
        net_wins = []
        net_losses = []
        net_trade_returns = []
        
        for t in self.trades:
            pnl = t.net_pnl
            net_trade_returns.append(pnl)
            
            if pnl > 0:
                net_wins.append(pnl)
            else:
                net_losses.append(abs(pnl))
        
        net_total_wins = sum(net_wins)
        net_total_losses = sum(net_losses) if net_losses else 1e-10
        self.net_profit_factor = net_total_wins / net_total_losses
        
        self.net_avg_win = np.mean(net_wins) if net_wins else 0.0
        self.net_avg_loss = np.mean(net_losses) if net_losses else 0.0
        
        net_win_rate = len(net_wins) / self.total_trades
        net_loss_rate = len(net_losses) / self.total_trades
        self.net_expectancy = (net_win_rate * self.net_avg_win) - (net_loss_rate * self.net_avg_loss)
        
        # Net Sharpe
        if len(net_trade_returns) > 1:
            net_ret_std = np.std(net_trade_returns, ddof=1)
            if net_ret_std > 0:
                self.net_sharpe_ratio = (np.mean(net_trade_returns) / net_ret_std) * np.sqrt(252)
        
        # Net Drawdown
        net_equity = np.cumsum(net_trade_returns)
        net_running_max = np.maximum.accumulate(net_equity)
        net_dd = net_running_max - net_equity
        self.net_max_drawdown = np.max(net_dd) if len(net_dd) > 0 else 0.0
        
        # Excursion (from actual returns - legacy)
        self.avg_mfe = np.mean([t.mfe for t in self.trades])
        self.avg_mae_excursion = np.mean([t.mae_excursion for t in self.trades])
        self.mfe_mae_ratio = abs(self.avg_mfe / self.avg_mae_excursion) if self.avg_mae_excursion != 0 else 0.0
        
        # Excursion Analysis (per-neighbor from OHLC - NEW)
        self.avg_neighbor_mfe = np.mean([t.avg_neighbor_mfe for t in self.trades if t.avg_neighbor_mfe > 0])
        self.avg_neighbor_mae = np.mean([t.avg_neighbor_mae for t in self.trades if t.avg_neighbor_mae > 0])
        # Calculate average E-Ratio (weighted by trade count)
        e_ratios = [t.e_ratio for t in self.trades if t.e_ratio > 0]
        self.avg_e_ratio = np.mean(e_ratios) if e_ratios else 0.0
        
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
            'RegimeFilter': self.config.get('use_regime_filter', False),
            
            # Key Metrics (GROSS)
            'Avg_RMSE': self.avg_rmse,
            'Avg_MAE': self.avg_mae,
            'Hit_Ratio': self.hit_ratio,
            'Dir_Accuracy': self.direction_accuracy,
            'Profit_Factor': self.profit_factor,
            'Sharpe': self.sharpe_ratio,
            'Coverage_80': self.avg_coverage_80,
            
            # Key Metrics (NET - after costs)
            'Total_Costs': self.total_costs,
            'Net_Profit_Factor': self.net_profit_factor,
            'Net_Sharpe': self.net_sharpe_ratio,
            'Net_Expectancy': self.net_expectancy,
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
            'use_regime_filter': self.config.get('use_regime_filter', False),
            'vol_method': self.config.get('vol_method', 'garman_klass'),
            
            # Metrics (GROSS - before costs)
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
            'avg_neighbor_mfe': self.avg_neighbor_mfe,
            'avg_neighbor_mae': self.avg_neighbor_mae,
            'avg_e_ratio': self.avg_e_ratio,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'long_win_rate': self.long_win_rate,
            'short_win_rate': self.short_win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'expectancy': self.expectancy,
            
            # Metrics (NET - after costs)
            'total_costs': self.total_costs,
            'net_profit_factor': self.net_profit_factor,
            'net_sharpe_ratio': self.net_sharpe_ratio,
            'net_max_drawdown': self.net_max_drawdown,
            'net_avg_win': self.net_avg_win,
            'net_avg_loss': self.net_avg_loss,
            'net_expectancy': self.net_expectancy,
        }


@dataclass
class BacktestConfig:
    """
    Configuration for Backtester.
    
    Transaction Costs
    -----------------
    Set transaction_costs to apply realistic trading costs:
    
        from sim_search.costs import FUTURES_NQ, FUTURES_ES, NO_COSTS
        
        config = BacktestConfig(
            data_path='data/NQ.parquet',
            transaction_costs=FUTURES_NQ,  # Standard NQ costs (~$22/RT)
        )
    
    Or use a string preset: 'NQ', 'ES', 'MNQ', 'MES', 'CL', 'GC', 'STOCK', 'NONE'
    """
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
    
    # Regime-aware settings
    use_regime_filter: bool = False  # Enable two-stage regime filtering
    vol_method: str = 'garman_klass'  # Volatility estimator: 'garman_klass' or 'parkinson'
    min_same_regime: int = 5  # Minimum windows in same regime before fallback
    
    # Transaction costs (NEW)
    # Can be TransactionCosts instance or string preset ('NQ', 'ES', etc.)
    transaction_costs: Union[TransactionCosts, str, None] = None
    
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
    
    def __post_init__(self):
        """Resolve transaction_costs if string preset."""
        if isinstance(self.transaction_costs, str):
            self.transaction_costs = get_preset(self.transaction_costs)
        elif self.transaction_costs is None:
            self.transaction_costs = NO_COSTS


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
                
                query_regime = -1
                same_regime_count = 0
                
                if self.config.use_regime_filter:
                    # NEW: Two-stage regime-aware search
                    idx, dists, query_regime, all_regimes = regime_aware_similarity_search(
                        x_train_list,
                        y_train,
                        x_test_df,
                        df=df,
                        intervals=windows,
                        query_idx=i,
                        n_neighbors=self.config.n_neighbors,
                        distance=self.config.distance_metric,
                        vol_method=self.config.vol_method,
                        min_same_regime=self.config.min_same_regime
                    )
                    # Count how many neighbors are from same regime
                    same_regime_count = np.sum(all_regimes[idx] == query_regime)
                else:
                    # Original: Standard similarity search
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
                
                # Calculate transaction costs
                costs = self.config.transaction_costs
                entry_price = df.loc[test_cutoff, 'close']
                
                # Get cost metrics
                _, _, transaction_cost = costs.calculate_trade_cost(entry_price)
                cost_as_return = costs.cost_as_return(entry_price)
                net_returns = costs.adjust_returns(y_test, entry_price)
                
                # Calculate Excursion Analysis (MFE/MAE per neighbor from OHLC)
                forecast_direction = np.sum(forecast) > 0  # True = LONG, False = SHORT
                excursion_metrics = calculate_excursion_metrics_per_neighbor(
                    neighbor_indices=idx,
                    neighbor_horizons=neighbor_horizons,
                    df=df,
                    intervals=windows,
                    entry_price=entry_price,
                    forecast_direction=forecast_direction
                )
                
                # Record trade
                trade = TradeResult(
                    cutoff=test_cutoff,
                    forecast_returns=forecast,
                    actual_returns=y_test,
                    percentile_bands=bands,
                    neighbor_distances=dists,
                    hit_tp=hit_tp,
                    regime=query_regime,
                    same_regime_count=same_regime_count,
                    # Transaction costs
                    entry_price=entry_price,
                    transaction_cost=transaction_cost,
                    cost_as_return=cost_as_return,
                    net_returns=net_returns,
                    # Excursion Analysis (per-neighbor)
                    neighbor_mfe_values=excursion_metrics['mfe_per_neighbor'],
                    neighbor_mae_values=excursion_metrics['mae_per_neighbor'],
                    avg_neighbor_mfe=excursion_metrics['avg_mfe'],
                    avg_neighbor_mae=excursion_metrics['avg_mae'],
                    e_ratio=excursion_metrics['e_ratio'],
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
                   f"PF: {result.profit_factor:.2f}, "
                   f"E-Ratio: {result.avg_e_ratio:.2f}")
        
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


def run_backtest(
    signals: pd.Series,
    returns: pd.Series,
    prices: Optional[pd.Series] = None,
    cost_model: Optional[TransactionCosts] = None
) -> Dict[str, Any]:
    """
    Vectorized backtest function for signal-based strategies.
    
    Calculates strategy returns using fully vectorized operations for O(n) performance.
    Applies lookahead protection and transaction costs.
    
    Parameters
    ----------
    signals : pd.Series
        Trading signals (-1, 0, +1) with DatetimeIndex. 
        Values > 0 = long, < 0 = short, 0 = flat.
    returns : pd.Series
        Asset returns (log or simple returns) with same index as signals.
    prices : pd.Series, optional
        Asset prices for cost calculation. If None, uses returns to approximate.
        Should have same index as signals/returns.
    cost_model : TransactionCosts, optional
        Transaction cost model. If None, uses NO_COSTS (no costs applied).
        
    Returns
    -------
    dict
        Dictionary with keys:
        - total_return: float - Total cumulative return
        - sharpe: float - Annualized Sharpe ratio
        - equity_curve: pd.Series - Cumulative equity curve
        - strategy_returns: pd.Series - Bar-by-bar strategy returns
        - trade_count: int - Number of trades executed
        - turnover: float - Portfolio turnover (avg trades per period)
        - total_costs: float - Total transaction costs in return terms
        
    Notes
    -----
    - Applies signals.shift(1) to prevent lookahead bias
    - Detects trades using signals.diff().abs() > 0
    - Subtracts transaction costs (commission + slippage) from returns at trade points
    - Cost is applied as: cost_as_return = cost_model.cost_as_return(price)
    
    Example
    -------
    >>> from sim_search.costs import FUTURES_NQ
    >>> signals = pd.Series([0, 1, 1, -1, -1, 0], index=dates)
    >>> returns = pd.Series([0.001, 0.002, -0.001, -0.002, 0.001], index=dates)
    >>> prices = pd.Series([100, 101, 103, 102, 100, 101], index=dates)
    >>> result = run_backtest(signals, returns, prices, cost_model=FUTURES_NQ)
    >>> print(f"Total return: {result['total_return']:.2%}")
    >>> print(f"Sharpe: {result['sharpe']:.2f}")
    """
    if cost_model is None:
        cost_model = NO_COSTS
    
    # Ensure signals and returns are aligned and have same length
    if not signals.index.equals(returns.index):
        # Align indices if possible
        common_idx = signals.index.intersection(returns.index)
        if len(common_idx) == 0:
            raise ValueError("signals and returns must have overlapping indices")
        signals = signals.loc[common_idx]
        returns = returns.loc[common_idx]
    
    n = len(signals)
    if n != len(returns):
        raise ValueError(f"signals and returns must have same length: {len(signals)} vs {len(returns)}")
    
    # Apply lookahead protection: signal at time T affects return at T+1
    signals_shifted = signals.shift(1)
    
    # Handle NaN at first row (no signal to use)
    signals_shifted = signals_shifted.fillna(0)
    
    # Calculate strategy returns: position * market return
    # signals > 0 = long, < 0 = short, 0 = flat
    strategy_returns = signals_shifted * returns
    
    # Detect trades: changes in signal position
    # signals.diff().abs() > 0 indicates a trade occurred
    signal_changes = signals.diff().abs()
    trade_mask = signal_changes > 0
    
    # Initialize cost series (will remain zeros if costs disabled or no trades)
    cost_as_return = pd.Series(0.0, index=signals.index)
    
    # Calculate transaction costs
    if cost_model.enabled and trade_mask.any():
        # Get prices for cost calculation
        if prices is None:
            # Approximate prices from returns (for cost calculation only)
            # Start with a base price of 100, then compound returns
            base_price = 100.0
            prices_approx = base_price * (1 + returns).cumprod()
            prices_for_cost = prices_approx
        else:
            # Align prices with signals/returns
            if not prices.index.equals(signals.index):
                prices_for_cost = prices.reindex(signals.index).ffill().bfill()
            else:
                prices_for_cost = prices
        
        # Fully vectorized cost calculation
        # Formula: cost_as_return = round_trip_cost / (price * point_value)
        # Where point_value = tick_value / tick_size
        # Rearranged: cost_as_return = (round_trip_cost / point_value) / price
        point_value = cost_model.tick_value / cost_model.tick_size
        cost_constant = cost_model.round_trip_cost / point_value
        
        # Calculate cost as return for ALL prices (fully vectorized)
        cost_as_return_full = cost_constant / prices_for_cost
        
        # Only apply costs at trade points (mask non-trade points to zero)
        cost_as_return = cost_as_return_full * trade_mask.astype(float)
        
        # Subtract costs from strategy returns at trade points
        # Costs are applied when entering a position (or changing position)
        strategy_returns = strategy_returns - cost_as_return
    
    # Calculate equity curve (cumulative returns)
    equity_curve = (1 + strategy_returns).cumprod()
    
    # Calculate total return
    total_return = equity_curve.iloc[-1] - 1.0
    
    # Calculate Sharpe ratio (annualized, assuming daily returns)
    if len(strategy_returns) > 1:
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std(ddof=1)
        if std_return > 0:
            # Annualized Sharpe = (mean / std) * sqrt(252) for daily
            sharpe = (mean_return / std_return) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    # Calculate trade statistics
    trade_count = int(trade_mask.sum())
    turnover = trade_count / n if n > 0 else 0.0
    
    # Total costs in return terms
    total_costs = float(cost_as_return.sum())
    
    return {
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'equity_curve': equity_curve,
        'strategy_returns': strategy_returns,
        'trade_count': trade_count,
        'turnover': float(turnover),
        'total_costs': float(total_costs),
    }