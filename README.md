# Market Similarity Search

A quantitative finance tool for market pattern matching and forecasting using time series similarity search.

## Overview

This project finds historically similar market patterns to forecast future price movements. Given a current market window (e.g., overnight + morning session), it finds the K most similar historical windows and uses their subsequent movements to generate a forecast.

## 📊 How It Works

### Process Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SIMILARITY SEARCH PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────┘

  1. LOAD DATA
     df = pd.read_parquet("NQ_data.parquet")
                              │
                              ▼
  2. BUILD WINDOWS
     ┌──────────────────────────────────────────────────────────────┐
     │  WindowCollectionBuilder(df)                                  │
     │    .with_time_anchored_windows(8:00 PM → 9:30 AM)            │
     │    .with_normalization('log_returns')                         │
     │    .with_volatility('garman_klass')  ← Compute GK Vol        │
     │    .with_calendar_events()           ← Tag FOMC/CPI/NFP      │
     │    .build()                                                   │
     └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
     WindowCollection: [WindowData₁, WindowData₂, ..., WindowDataₙ]
     Each window bundles: pattern (x), horizon (y), cutoff, volatility,
                          regime, is_fomc_day, is_cpi_day, is_nfp_day
                              │
                              ▼
  3. SPLIT TRAIN/TEST
     train, test = collection.split_train_test()
                              │
                              ▼
  4. FILTER (Pre-KNN)
     ┌──────────────────────────────────────────────────────────────┐
     │  FilterPipeline([                                             │
     │      RegimeFilter()     ← Only same volatility regime        │
     │      CalendarFilter()   ← Match FOMC context                 │
     │  ])                                                           │
     └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
     193 windows → 62 windows (same regime + calendar context)
                              │
                              ▼
  5. KNN SIMILARITY SEARCH
     ┌──────────────────────────────────────────────────────────────┐
     │  KNeighborsTimeSeriesRegressor(                              │
     │      n_neighbors=10,                                          │
     │      distance='wdtw'   ← Weighted DTW (tail emphasis)        │
     │  )                                                            │
     │                                                               │
     │  Library: aeon.regression.distance_based                      │
     └──────────────────────────────────────────────────────────────┘
                              │
                              ▼
     10 most similar historical patterns found
                              │
                              ▼
  6. FORECAST
     Average the 10 neighbors' subsequent returns
     → Predicted return: +0.15% over next 20 bars
                              │
                              ▼
  7. VISUALIZE
     plot_with_volatility() → report.html
     Shows: candlestick + forecast + volatility subplot + regime shading
```

### Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `config_playground.py` | **PM Playground** - Easy parameter tuning | Quick experiments |
| `market_forecast.py` | Full forecast with strategy comparison | Daily analysis |
| `compare_strategies.py` | A/B backtest comparison | Validate improvements |

### Quick Commands

```bash
# PM wants to experiment with parameters:
uv run python config_playground.py

# Run full analysis with multiple strategies:
uv run python market_forecast.py

# Backtest comparison:
uv run python compare_strategies.py
```

---

## Key Features

### Core Functionality
- **KNN-based Similarity Search**: Uses `aeon` library's `KNeighborsTimeSeriesRegressor`
- **Multiple Distance Metrics**: DTW, WDTW (weighted DTW), Euclidean, MSM, and more
- **Walk-Forward Backtesting**: Proper out-of-sample testing with no look-ahead bias
- **Parallel Grid Search**: Optimize parameters across multiple configurations

### Regime-Aware Similarity Search

The latest enhancement adds **volatility regime filtering** to ensure we only compare patterns from similar market conditions.

**The Problem**: A 2% drop looks identical in shape whether it happens in a calm market or a volatile one (e.g., FOMC day). But the implications are completely different - in low volatility, this is a significant move; in high volatility, it's just noise.

**The Solution**: Two-stage pipeline:
1. **Stage 1 - Regime Filter**: Compute realized volatility using Garman-Klass (uses all OHLC data), classify windows into LOW/MED/HIGH volatility buckets, filter to same regime
2. **Stage 2 - WDTW Similarity**: Run KNN with Weighted DTW on filtered windows only. WDTW automatically applies more weight to recent bars (tail weighting)

**Why Garman-Klass instead of VXN?**
- VXN is daily implied volatility (forward-looking, options-derived)
- We need intraday realized volatility at the same granularity as our patterns
- GK uses all OHLC data and is ~5x more statistically efficient than close-to-close

---

## Installation

```bash
# Ensure you have uv installed
# https://docs.astral.sh/uv/guides/install-python/

# Sync dependencies
uv sync

# Run the playground
uv run python config_playground.py
```

---

## Usage Examples

### PM Parameter Playground

Edit variables at the top of `config_playground.py`:

```python
# Toggle filters ON/OFF
REGIME_FILTER_ENABLED = True
CALENDAR_FILTER_ENABLED = True
MATCH_FOMC_CONTEXT = True

# KNN settings
N_NEIGHBORS = 10
DISTANCE_METRIC = "wdtw"  # Options: "wdtw", "dtw", "euclidean"

# Then run:
# uv run python config_playground.py
```

### Run Single Backtest

```python
from sim_search.backtester import BacktestConfig, Backtester
import pandas as pd

df = pd.read_parquet('data/test/NQ_2024-09-06_2025-09-13.parquet')

config = BacktestConfig(
    data_path='data/test/NQ_2024-09-06_2025-09-13.parquet',
    window_len=60,
    forecast_horizon=20,
    n_neighbors=10,
    distance_metric='wdtw',
    use_regime_filter=True,
    vol_method='garman_klass',
)

backtester = Backtester(config)
result = backtester.run(df)
print(f"Direction Accuracy: {result.direction_accuracy:.1%}")
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
fig.write_html("report.html", auto_open=True)
```

---

## Project Structure

```
sim_search/
├── __init__.py           # Public API exports
│
│   # Data Structures (NEW)
├── datastructures.py     # WindowData, WindowCollection
├── builder.py            # WindowCollectionBuilder (fluent API)
├── filters.py            # RegimeFilter, CalendarFilter, FilterPipeline
├── calendar_events.py    # FOMC, CPI, NFP dates
│
│   # Core Algorithms
├── forecaster.py         # KNN similarity search
├── volatility.py         # GK/Parkinson vol, regime classification
├── windowing.py          # Window partitioning
├── backtester.py         # Walk-forward backtesting
│
│   # Visualization & Utils
├── visualization.py      # Plotly charts
├── times.py              # Exchange calendar utils
├── config.py             # Configuration
├── optimizer.py          # Grid search
└── reporting.py          # Backtest reports

# Root Scripts
config_playground.py      # 🎮 PM parameter playground
market_forecast.py        # Full forecast analysis
compare_strategies.py     # A/B strategy comparison
```

---

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

### Distance Metrics

| Metric | Speed | Description |
|--------|-------|-------------|
| `euclidean` | ⚡ Fast | Point-by-point L2 distance |
| `dtw` | 🐢 Slow | Dynamic Time Warping |
| `wdtw` | 🐢 Slow | **Weighted DTW** - more weight on recent bars |
| `ddtw` | 🐢 Slow | Derivative DTW |
| `msm` | 🐢 Slow | Move-Split-Merge |

---

## New Abstractions

### WindowData - Bundle Everything Together

Instead of parallel arrays that must stay aligned:

```python
from sim_search import WindowData, WindowCollection

window = WindowData(
    idx=0,
    x=pattern_array,           # Normalized pattern
    y=future_returns,          # Forecast horizon
    cutoff=timestamp,
    volatility=0.0003,         # GK volatility
    regime=1,                  # LOW=0, MED=1, HIGH=2
    is_fomc_day=False,
    days_since_fomc=5,
)
```

### Pluggable Filters (sklearn-compatible)

```python
from sim_search import RegimeFilter, CalendarFilter, FilterPipeline

pipeline = FilterPipeline([
    RegimeFilter(enabled=True),
    CalendarFilter(match_fomc_context=True),
])

pipeline.fit(train_collection)
indices = pipeline.transform(train_collection, query=test_window)
```

### Builder Pattern

```python
from sim_search import WindowCollectionBuilder
from datetime import time

collection = (WindowCollectionBuilder(df)
    .with_time_anchored_windows(time(8), time(9, 30), extend_sessions=1)
    .with_horizon(20)
    .with_normalization('log_returns')
    .with_volatility('garman_klass')
    .with_calendar_events()
    .build())

train, test = collection.split_train_test()
```

---

## Future Enhancements

- [ ] **Euclidean pre-filter**: Fast coarse filter before WDTW (for 10k+ windows)
- [ ] **Path signatures**: State-of-the-art feature extraction
- [ ] **Statistical significance testing**: Bootstrap CI, p-values
- [ ] **Transaction cost modeling**: Commissions, slippage, spread

---

## References

- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67-78.
- Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE TASSP*, 26(1), 43-49.

## License

MIT
