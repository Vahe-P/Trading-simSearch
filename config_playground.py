#!/usr/bin/env python3
"""
🎮 PARAMETER PLAYGROUND

Edit the parameters below and run this script to test different configurations.
This makes it easy to experiment with different settings without touching the main code.

Usage:
    uv run python config_playground.py

Output:
    - Console output with strategy comparison
    - report_with_volatility.html (open in browser)
"""

# =============================================================================
# 📊 DATA SETTINGS
# =============================================================================
DATA_PATH = "data/test/NQ_2024-09-06_2025-09-13.parquet"

# =============================================================================
# 🔧 FILTER SETTINGS (toggle on/off)
# =============================================================================

# Regime Filter - match patterns from same volatility regime
REGIME_FILTER_ENABLED = True
VOL_METHOD = "garman_klass"  # Options: "garman_klass", "parkinson"

# Calendar Filter - match FOMC/event context
CALENDAR_FILTER_ENABLED = True
MATCH_FOMC_CONTEXT = True      # FOMC days only match other FOMC days
EXCLUDE_RED_FOLDER = False     # Exclude all high-impact event days (FOMC, CPI, NFP)

# =============================================================================
# 🎯 KNN SETTINGS
# =============================================================================
N_NEIGHBORS = 10               # K in KNN (more = smoother, less = more reactive)
DISTANCE_METRIC = "wdtw"       # Options: "wdtw", "dtw", "euclidean", "msm"

# =============================================================================
# 📈 WINDOW SETTINGS
# =============================================================================
from datetime import time

WINDOW_START = time(8, 0)      # Window start time (8:00 PM = overnight start)
WINDOW_END = time(9, 30)       # Window end time (9:30 AM = market open)
EXTEND_SESSIONS = 1            # 1 = overnight (spans to next day)
HORIZON_LEN = 20               # Bars to forecast ahead
NORM_METHOD = "log_returns"    # Options: "log_returns", "pct_change", "rolling_zscore"

# =============================================================================
# 🚀 RUN FORECAST
# =============================================================================

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from loguru import logger
    
    from sim_search import (
        WindowCollectionBuilder,
        RegimeFilter, CalendarFilter, FilterPipeline,
        REGIME_NAMES
    )
    from sim_search.forecaster import similarity_search, forecast_from_neighbors
    from sim_search.visualization import plot_with_volatility
    
    print("="*70)
    print(" 🎮 PARAMETER PLAYGROUND")
    print("="*70)
    
    # Show current config
    print("\n📋 Current Configuration:")
    print(f"   Regime Filter:   {'✓ ON' if REGIME_FILTER_ENABLED else '✗ OFF'} ({VOL_METHOD})")
    print(f"   Calendar Filter: {'✓ ON' if CALENDAR_FILTER_ENABLED else '✗ OFF'}")
    print(f"   Match FOMC:      {'✓ YES' if MATCH_FOMC_CONTEXT else '✗ NO'}")
    print(f"   Exclude Events:  {'✓ YES' if EXCLUDE_RED_FOLDER else '✗ NO'}")
    print(f"   Distance Metric: {DISTANCE_METRIC}")
    print(f"   K Neighbors:     {N_NEIGHBORS}")
    print(f"   Horizon:         {HORIZON_LEN} bars")
    
    # Load data
    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Loaded {len(df):,} bars")
    
    # Build collection
    print("\n🔨 Building window collection...")
    builder = WindowCollectionBuilder(df)
    collection = (builder
        .with_time_anchored_windows(WINDOW_START, WINDOW_END, EXTEND_SESSIONS)
        .with_horizon(HORIZON_LEN)
        .with_normalization(NORM_METHOD)
        .with_volatility(VOL_METHOD)
        .with_calendar_events()
        .build())
    
    print(f"   Built {len(collection)} windows")
    
    # Split train/test
    train, test = collection.split_train_test()
    print(f"   Train: {len(train)}, Test regime: {test.regime_name}")
    
    # Build filter pipeline based on settings
    print("\n🔍 Applying filters...")
    filters = []
    
    if REGIME_FILTER_ENABLED:
        filters.append(RegimeFilter(enabled=True, vol_method=VOL_METHOD))
    
    if CALENDAR_FILTER_ENABLED:
        filters.append(CalendarFilter(
            enabled=True,
            match_fomc_context=MATCH_FOMC_CONTEXT,
            exclude_red_folder=EXCLUDE_RED_FOLDER
        ))
    
    pipeline = FilterPipeline(filters) if filters else None
    
    if pipeline:
        pipeline.fit(train)
        filtered_indices = pipeline.transform(train, query=test)
        filtered_train = train.filter_by_indices(filtered_indices.tolist())
        print(f"   Filtered: {len(filtered_train)}/{len(train)} windows ({len(filtered_train)/len(train)*100:.1f}%)")
    else:
        filtered_train = train
        print(f"   No filters applied, using all {len(train)} windows")
    
    # Run KNN
    print(f"\n🎯 Running KNN (k={N_NEIGHBORS}, distance={DISTANCE_METRIC})...")
    
    # Convert to format expected by similarity_search
    x_train_list = [pd.DataFrame(w.x) for w in filtered_train]
    x_test_df = pd.DataFrame(test.x)
    
    neighbor_idx, neighbor_dists = similarity_search(
        x_train_list,
        np.zeros(len(x_train_list)),
        x_test_df,
        n_neighbors=min(N_NEIGHBORS, len(filtered_train)),
        impl='knn',
        distance=DISTANCE_METRIC
    )
    
    # Get neighbor horizons and forecast
    neighbor_horizons = np.stack([filtered_train[i].y for i in neighbor_idx])
    forecast = forecast_from_neighbors(neighbor_horizons, neighbor_dists, impl='avg')
    
    print(f"   Found {len(neighbor_idx)} neighbors")
    print(f"   Forecast direction: {'LONG ↑' if np.sum(forecast) > 0 else 'SHORT ↓'}")
    print(f"   Forecast magnitude: {np.sum(forecast)*100:+.3f}%")
    
    # Actual
    actual_dir = "LONG ↑" if np.sum(test.y) > 0 else "SHORT ↓"
    actual_mag = np.sum(test.y) * 100
    print(f"   Actual direction:  {actual_dir}")
    print(f"   Actual magnitude:  {actual_mag:+.3f}%")
    
    # Direction match?
    match = "✓ CORRECT" if (np.sum(forecast) > 0) == (np.sum(test.y) > 0) else "✗ WRONG"
    print(f"\n   Direction: {match}")
    
    # Show neighbors
    print(f"\n📊 Neighbors Found:")
    print("   ┌─────┬──────────────────────┬──────────┬────────┐")
    print("   │  #  │ Cutoff               │ Distance │ Regime │")
    print("   ├─────┼──────────────────────┼──────────┼────────┤")
    for i, (idx, dist) in enumerate(zip(neighbor_idx, neighbor_dists)):
        w = filtered_train[idx]
        regime_emoji = '🟢' if w.regime == 0 else '🟡' if w.regime == 1 else '🔴'
        print(f"   │ {i+1:3d} │ {str(w.cutoff)[:20]:20s} │ {dist:8.4f} │ {w.regime_name:3s} {regime_emoji} │")
    print("   └─────┴──────────────────────┴──────────┴────────┘")
    
    # Generate visualization
    print("\n📈 Generating visualization...")
    fig = plot_with_volatility(
        df=df,
        cutoff=test.cutoff,
        forecast_returns=forecast,
        window_size=test.x.shape[-1],
        actual_returns=test.y,
        regime=test.regime,
        title=f"Forecast (k={N_NEIGHBORS}, {DISTANCE_METRIC}, regime={'ON' if REGIME_FILTER_ENABLED else 'OFF'})",
        hist_context_bars=100  # Only show last 100 bars before cutoff, not entire pattern
    )
    
    output_file = "report_playground.html"
    fig.write_html(output_file, full_html=True, auto_open=True)
    print(f"   ✓ Saved to {output_file} (opened in browser)")
    
    print("\n" + "="*70)
    print(" 🎮 Done! Edit parameters above and run again to experiment.")
    print("="*70)
