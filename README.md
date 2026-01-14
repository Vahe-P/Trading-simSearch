# Market Similarity Search

A quantitative finance tool for market pattern matching and forecasting using time series similarity search.

## Overview

This project finds historically similar market patterns to forecast future price movements. Given a current market window (e.g., overnight + morning session), it finds the K most similar historical windows and uses their subsequent movements to generate a forecast.

## Key Features

### Core Functionality
- **KNN-based Similarity Search**: Uses `aeon` library's `KNeighborsTimeSeriesRegressor`
- **Multiple Distance Metrics**: DTW, WDTW (weighted DTW), Euclidean, MSM, and more
- **Walk-Forward Backtesting**: Proper out-of-sample testing with no look-ahead bias
- **Parallel Grid Search**: Optimize parameters across multiple configurations

### Regime-Aware Similarity Search (NEW)

The latest enhancement adds **volatility regime filtering** to ensure we only compare patterns from similar market conditions.

**The Problem**: A 2% drop looks identical in shape whether it happens in a calm market or a volatile one (e.g., FOMC day). But the implications are completely different - in low volatility, this is a significant move; in high volatility, it's just noise.

**The Solution**: Two-stage pipeline:
1. **Stage 1 - Regime Filter**: Compute realized volatility using Garman-Klass (uses all OHLC data), classify windows into LOW/MED/HIGH volatility buckets, filter to same regime
2. **Stage 2 - WDTW Similarity**: Run KNN with Weighted DTW on filtered windows only. WDTW automatically applies more weight to recent bars (tail weighting)

**Why Garman-Klass instead of VXN?**
- VXN is daily implied volatility (forward-looking, options-derived)
- We need intraday realized volatility at the same granularity as our patterns
- GK uses all OHLC data and is ~5x more statistically efficient than close-to-close

## Installation

```bash
# Ensure you have uv installed
# https://docs.astral.sh/uv/guides/install-python/

# Sync dependencies
uv sync

# Run the comparison
uv run python compare_strategies.py
```

## Quick Start

### Run A/B Comparison

Compare baseline (DTW on all windows) vs regime-aware (GK filter + WDTW):

```bash
uv run python compare_strategies.py
```

### Run Single Backtest

```python
from sim_search.backtester import BacktestConfig, Backtester
import pandas as pd

# Load data
df = pd.read_parquet('data/test/NQ_2024-09-06_2025-09-13.parquet')

# Configure backtest with regime filtering
config = BacktestConfig(
    data_path='data/test/NQ_2024-09-06_2025-09-13.parquet',
    window_len=60,
    step_size=5,
    forecast_horizon=20,
    n_neighbors=10,
    distance_metric='wdtw',        # Weighted DTW (built-in tail weighting)
    use_regime_filter=True,         # Enable regime filtering
    vol_method='garman_klass',      # Use GK volatility
)

# Run backtest
backtester = Backtester(config)
result = backtester.run(df)

print(f"Direction Accuracy: {result.direction_accuracy:.1%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Profit Factor: {result.profit_factor:.2f}")
```

### Visualize with Volatility

```python
from sim_search.visualization import plot_with_volatility

fig = plot_with_volatility(
    df=df,
    cutoff=cutoff_timestamp,
    forecast_returns=forecast,
    window_size=60,
    actual_returns=actual,
    regime=1,  # 0=LOW, 1=MED, 2=HIGH
)
fig.write_html("report_with_vol.html", auto_open=True)
```

## Configuration Options

### BacktestConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_len` | int | 60 | Historical window size (bars) |
| `step_size` | int | 1 | Bars to advance between windows |
| `forecast_horizon` | int | 20 | Bars to forecast ahead |
| `n_neighbors` | int | 5 | K in KNN |
| `distance_metric` | str | 'euclidean' | Distance: 'dtw', 'wdtw', 'euclidean', etc. |
| `use_regime_filter` | bool | False | Enable regime-aware filtering |
| `vol_method` | str | 'garman_klass' | Volatility estimator |
| `min_same_regime` | int | 5 | Min windows before fallback |

### Distance Metrics

| Metric | Speed | Description |
|--------|-------|-------------|
| `euclidean` | ⚡ Fast | Point-by-point L2 distance |
| `dtw` | 🐢 Slow | Dynamic Time Warping |
| `wdtw` | 🐢 Slow | **Weighted DTW** - more weight on recent bars |
| `ddtw` | 🐢 Slow | Derivative DTW |
| `msm` | 🐢 Slow | Move-Split-Merge |

## Project Structure

```
sim_search/
├── datastructures.py  # WindowData, WindowCollection (NEW)
├── filters.py         # RegimeFilter, CalendarFilter, FilterPipeline (NEW)
├── builder.py         # WindowCollectionBuilder - fluent API (NEW)
├── calendar_events.py # FOMC, CPI, NFP dates (NEW)
├── backtester.py      # Walk-forward backtesting engine
├── forecaster.py      # KNN similarity search + regime-aware search
├── volatility.py      # GK/Parkinson volatility + regime classification
├── windowing.py       # Window partitioning utilities
├── visualization.py   # Plotly charts with volatility subplot
├── optimizer.py       # Grid search optimization
├── clustering.py      # Future path clustering
└── config.py          # Configuration management

compare_strategies.py  # A/B test baseline vs regime-aware
market_forecast.py     # Main forecast script with strategy comparison
```

## New Abstractions (Pluggable Filter System)

### Data Structures

Instead of parallel arrays that must stay aligned, we now bundle everything in `WindowData`:

```python
from sim_search import WindowData, WindowCollection

# Each window has all data together
window = WindowData(
    idx=0,
    x=pattern_array,           # Normalized pattern
    y=future_returns,          # Forecast horizon
    cutoff=timestamp,          # Label
    volatility=0.0003,         # GK volatility
    regime=1,                  # LOW=0, MED=1, HIGH=2
    is_fomc_day=False,         # Calendar events
    days_since_fomc=5,
)
```

### Pluggable Filters (sklearn-compatible)

Filters follow sklearn's transformer API and can be chained:

```python
from sim_search import RegimeFilter, CalendarFilter, FilterPipeline

# Create filter pipeline
pipeline = FilterPipeline([
    RegimeFilter(enabled=True, vol_method='garman_klass'),
    CalendarFilter(match_fomc_context=True, exclude_red_folder=False),
])

# Fit on training data
pipeline.fit(train_collection)

# Get filtered indices for a query
indices = pipeline.transform(train_collection, query=test_window)
```

### Builder Pattern

Fluent builder for creating enriched collections:

```python
from sim_search import WindowCollectionBuilder
from datetime import time

builder = WindowCollectionBuilder(df)
collection = (builder
    .with_time_anchored_windows(time(8), time(9, 30), extend_sessions=1)
    .with_horizon(20)
    .with_normalization('log_returns')
    .with_volatility('garman_klass')
    .with_calendar_events()  # FOMC, CPI, NFP
    .build())

# Easy train/test split
train, test = collection.split_train_test()

# Filter by regime
same_regime = train.filter_by_regime(test.regime)
```

### Calendar Events

Built-in FOMC, CPI, NFP dates for filtering:

```python
from sim_search import is_fomc_day, days_since_fomc

# Check if date is FOMC day
is_fomc_day(timestamp)  # True/False

# Days since last FOMC
days_since_fomc(timestamp)  # e.g., 5
```

## Improvements Over Baseline

### What Was Missing
1. **No volatility awareness**: DTW compared all patterns regardless of market conditions
2. **Only used close price**: OHLC data available but unused for volatility
3. **No tail weighting**: All bars weighted equally despite recent bars being more predictive

### What's New
1. **Regime filtering**: Classify windows by GK volatility, only compare same regime
2. **WDTW distance**: Built-in tail weighting (recent bars matter more)
3. **Volatility visualization**: See vol subplot and regime shading in HTML reports

## Future Enhancements

- [ ] **Euclidean pre-filter**: Add fast coarse filter before WDTW (for 10k+ windows)
- [ ] **Path signatures**: State-of-the-art feature extraction for fast filter
- [ ] **Statistical significance testing**: Bootstrap CI, p-values
- [ ] **Transaction cost modeling**: Commissions, slippage, spread

## References

- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67-78.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE TASSP*, 26(1), 43-49.

## License

MIT
