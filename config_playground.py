#!/usr/bin/env python3
"""
🎮 PARAMETER PLAYGROUND

Edit the parameters below and run this script to test different configurations.
This makes it easy to experiment with different settings without touching the main code.

Usage:
    uv run python config_playground.py

Output:
    - Detailed console output with regime/calendar stats
    - report_playground.html (opens in browser)
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

# WDTW weight parameter (only used if DISTANCE_METRIC = "wdtw")
# Higher = more weight on recent bars (tail of pattern)
WDTW_G = 0.05                  # Default: 0.05

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
    from collections import Counter
    
    from sim_search import (
        WindowCollectionBuilder,
        RegimeFilter, CalendarFilter, FilterPipeline,
        REGIME_NAMES
    )
    from sim_search.forecaster import similarity_search, forecast_from_neighbors
    from sim_search.visualization import plot_with_volatility
    # Calendar events are embedded in WindowData via enrich_window_with_calendar
    
    print("\n" + "="*70)
    print(" 🎮 PARAMETER PLAYGROUND - Regime-Aware Market Similarity Search")
    print("="*70)
    
    # =========================================================================
    # CONFIG SUMMARY
    # =========================================================================
    print("\n┌" + "─"*68 + "┐")
    print("│ 📋 CURRENT CONFIGURATION" + " "*43 + "│")
    print("├" + "─"*68 + "┤")
    print(f"│  Regime Filter:   {'✅ ENABLED':20s} │ Vol Method:  {VOL_METHOD:15s}  │")
    print(f"│  Calendar Filter: {'✅ ENABLED' if CALENDAR_FILTER_ENABLED else '❌ DISABLED':20s} │ Match FOMC:  {'YES' if MATCH_FOMC_CONTEXT else 'NO':15s}  │")
    print("├" + "─"*68 + "┤")
    print(f"│  KNN Algorithm:   K-Nearest Neighbors Time Series Regressor        │")
    print(f"│  Distance Metric: {DISTANCE_METRIC.upper():15s} │ K Neighbors: {N_NEIGHBORS:<15d}  │")
    if DISTANCE_METRIC == "wdtw":
        print(f"│  WDTW Weight (g): {WDTW_G:<15.3f} │ (higher = more recent weight) │")
    print("├" + "─"*68 + "┤")
    print(f"│  Window:          {WINDOW_START.strftime('%H:%M')} → {WINDOW_END.strftime('%H:%M')} (overnight session)              │")
    print(f"│  Horizon:         {HORIZON_LEN} bars ahead" + " "*40 + "│")
    print(f"│  Normalization:   {NORM_METHOD:20s}" + " "*28 + "│")
    print("└" + "─"*68 + "┘")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Total bars: {len(df):,}")
    print(f"   Date range: {df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # BUILD WINDOWS
    # =========================================================================
    print("\n🔨 Building window collection...")
    builder = WindowCollectionBuilder(df)
    collection = (builder
        .with_time_anchored_windows(WINDOW_START, WINDOW_END, EXTEND_SESSIONS)
        .with_horizon(HORIZON_LEN)
        .with_normalization(NORM_METHOD)
        .with_volatility(VOL_METHOD)
        .with_calendar_events()
        .build())
    
    print(f"   Total windows: {len(collection)}")
    
    # Split train/test
    train, test = collection.split_train_test()
    
    # =========================================================================
    # REGIME ANALYSIS
    # =========================================================================
    print("\n" + "─"*70)
    print(" 📊 MARKET REGIME ANALYSIS (Garman-Klass Volatility)")
    print("─"*70)
    
    # Count regimes in training set
    regime_counts = Counter(w.regime for w in train)
    total_train = len(train)
    
    print(f"\n   Training Data Regime Distribution:")
    print("   ┌─────────────┬─────────┬─────────┬──────────────────────────────┐")
    print("   │ Regime      │ Count   │ Percent │ Bar                          │")
    print("   ├─────────────┼─────────┼─────────┼──────────────────────────────┤")
    
    for regime in [0, 1, 2]:
        count = regime_counts.get(regime, 0)
        pct = count / total_train * 100 if total_train > 0 else 0
        name = REGIME_NAMES[regime]
        emoji = '🟢' if regime == 0 else '🟡' if regime == 1 else '🔴'
        bar = '█' * int(pct / 3)
        print(f"   │ {emoji} {name:8s} │ {count:7d} │ {pct:6.1f}% │ {bar:28s} │")
    
    print("   └─────────────┴─────────┴─────────┴──────────────────────────────┘")
    
    # Test window regime
    test_emoji = '🟢' if test.regime == 0 else '🟡' if test.regime == 1 else '🔴'
    print(f"\n   🎯 Test Window Regime: {test_emoji} {test.regime_name}")
    print(f"      Cutoff: {test.cutoff}")
    print(f"      Volatility: {test.volatility:.6f}")
    
    # =========================================================================
    # CALENDAR EVENT ANALYSIS
    # =========================================================================
    print("\n" + "─"*70)
    print(" 📅 CALENDAR EVENT ANALYSIS (Red Folder Days)")
    print("─"*70)
    
    # Count events in training
    fomc_count = sum(1 for w in train if w.is_fomc_day)
    cpi_count = sum(1 for w in train if w.is_cpi_day)
    nfp_count = sum(1 for w in train if w.is_nfp_day)
    event_days = sum(1 for w in train if w.is_fomc_day or w.is_cpi_day or w.is_nfp_day)
    
    print(f"\n   Training Set Event Days:")
    print("   ┌──────────────────────────────────────────────────────────────┐")
    print(f"   │ 🏛️  FOMC Days:           {fomc_count:4d} windows                        │")
    print(f"   │ 📊 CPI Release Days:    {cpi_count:4d} windows                        │")
    print(f"   │ 💼 NFP Release Days:    {nfp_count:4d} windows                        │")
    print("   ├──────────────────────────────────────────────────────────────┤")
    print(f"   │ 📛 Total Red Folder:    {event_days:4d} windows ({event_days/total_train*100:.1f}%)             │")
    print("   └──────────────────────────────────────────────────────────────┘")
    
    # Test window calendar info
    test_events = []
    if test.is_fomc_day: test_events.append("FOMC")
    if test.is_cpi_day: test_events.append("CPI")
    if test.is_nfp_day: test_events.append("NFP")
    
    print(f"\n   🎯 Test Window Calendar:")
    print(f"      FOMC day: {'✅ YES' if test.is_fomc_day else '❌ NO'} (Days since FOMC: {test.days_since_fomc})")
    print(f"      CPI day:  {'✅ YES' if test.is_cpi_day else '❌ NO'}")
    print(f"      NFP day:  {'✅ YES' if test.is_nfp_day else '❌ NO'}")
    if test_events:
        print(f"      ⚠️  RED FOLDER DAY: {', '.join(test_events)}")
    
    # =========================================================================
    # APPLY FILTERS
    # =========================================================================
    print("\n" + "─"*70)
    print(" 🔍 FILTERING PIPELINE")
    print("─"*70)
    
    filters = []
    
    if REGIME_FILTER_ENABLED:
        filters.append(RegimeFilter(enabled=True, vol_method=VOL_METHOD))
        print(f"\n   1️⃣  RegimeFilter: Only match {test.regime_name} volatility windows")
    
    if CALENDAR_FILTER_ENABLED:
        filters.append(CalendarFilter(
            enabled=True,
            match_fomc_context=MATCH_FOMC_CONTEXT,
            exclude_red_folder=EXCLUDE_RED_FOLDER
        ))
        if MATCH_FOMC_CONTEXT:
            print(f"   2️⃣  CalendarFilter: FOMC context matching {'enabled' if test.is_fomc_day else '(test is non-FOMC)'}")
        if EXCLUDE_RED_FOLDER:
            print(f"   2️⃣  CalendarFilter: Excluding red folder days")
    
    pipeline = FilterPipeline(filters) if filters else None
    
    if pipeline:
        pipeline.fit(train)
        filtered_indices = pipeline.transform(train, query=test)
        filtered_train = train.filter_by_indices(filtered_indices.tolist())
        
        # Show filtering breakdown
        print(f"\n   📉 Filter Results:")
        print(f"      Before filtering: {len(train)} windows")
        print(f"      After filtering:  {len(filtered_train)} windows")
        print(f"      Reduction:        {len(train) - len(filtered_train)} windows removed ({(1 - len(filtered_train)/len(train))*100:.1f}%)")
    else:
        filtered_train = train
        print(f"\n   ⚠️  No filters applied, using all {len(train)} windows")
    
    # Show filtered regime distribution
    filtered_regime_counts = Counter(w.regime for w in filtered_train)
    print(f"\n   Filtered Set Regime Distribution:")
    for regime in [0, 1, 2]:
        count = filtered_regime_counts.get(regime, 0)
        emoji = '🟢' if regime == 0 else '🟡' if regime == 1 else '🔴'
        print(f"      {emoji} {REGIME_NAMES[regime]}: {count} windows")
    
    # =========================================================================
    # KNN SIMILARITY SEARCH
    # =========================================================================
    print("\n" + "─"*70)
    print(f" 🎯 KNN SIMILARITY SEARCH ({DISTANCE_METRIC.upper()})")
    print("─"*70)
    
    print(f"\n   Algorithm: K-Nearest Neighbors Time Series Regressor")
    print(f"   Library:   aeon.regression.distance_based.KNeighborsTimeSeriesRegressor")
    print(f"   Distance:  {DISTANCE_METRIC.upper()}", end="")
    if DISTANCE_METRIC == "wdtw":
        print(f" (Weighted DTW - emphasizes recent bars)")
        print(f"              Weight parameter g={WDTW_G}")
    elif DISTANCE_METRIC == "dtw":
        print(f" (Dynamic Time Warping - shape matching)")
    elif DISTANCE_METRIC == "euclidean":
        print(f" (Euclidean - point-to-point distance)")
    elif DISTANCE_METRIC == "msm":
        print(f" (Move-Split-Merge - edit distance)")
    else:
        print()
    
    print(f"   K:         {N_NEIGHBORS} neighbors")
    print(f"   Search pool: {len(filtered_train)} windows")
    
    # Convert to format expected by similarity_search
    x_train_list = [pd.DataFrame(w.x) for w in filtered_train]
    x_test_df = pd.DataFrame(test.x)
    
    # Set distance params
    distance_params = None
    if DISTANCE_METRIC == "wdtw":
        distance_params = {"g": WDTW_G}
    
    print(f"\n   🔄 Computing {DISTANCE_METRIC.upper()} distances against {len(filtered_train)} patterns...")
    
    neighbor_idx, neighbor_dists = similarity_search(
        x_train_list,
        np.zeros(len(x_train_list)),
        x_test_df,
        n_neighbors=min(N_NEIGHBORS, len(filtered_train)),
        impl='knn',
        distance=DISTANCE_METRIC,
        distance_params=distance_params
    )
    
    # =========================================================================
    # NEIGHBORS FOUND
    # =========================================================================
    print(f"\n   ✅ Found {len(neighbor_idx)} similar patterns:")
    print("\n   ┌─────┬──────────────────────────┬──────────┬────────┬─────────┬───────┐")
    print("   │  #  │ Cutoff                   │ Distance │ Regime │ FOMC?   │ Δ Ret │")
    print("   ├─────┼──────────────────────────┼──────────┼────────┼─────────┼───────┤")
    
    for i, (idx, dist) in enumerate(zip(neighbor_idx, neighbor_dists)):
        w = filtered_train[idx]
        regime_emoji = '🟢' if w.regime == 0 else '🟡' if w.regime == 1 else '🔴'
        fomc_str = "🏛️ YES" if w.is_fomc_day else "   NO"
        horizon_ret = np.sum(w.y) * 100  # Convert to percent
        # Format distance: use scientific notation for very small values
        dist_str = f"{dist:.2e}" if dist < 0.0001 and dist > 0 else f"{dist:.4f}"
        print(f"   │ {i+1:3d} │ {str(w.cutoff)[:24]:24s} │ {dist_str:>8s} │ {w.regime_name:3s} {regime_emoji} │ {fomc_str} │ {horizon_ret:+5.2f}% │")
    
    print("   └─────┴──────────────────────────┴──────────┴────────┴─────────┴───────┘")
    
    # =========================================================================
    # FORECAST
    # =========================================================================
    print("\n" + "─"*70)
    print(" 📈 FORECAST RESULT")
    print("─"*70)
    
    # Get neighbor horizons and forecast
    neighbor_horizons = np.stack([filtered_train[i].y for i in neighbor_idx])
    forecast = forecast_from_neighbors(neighbor_horizons, neighbor_dists, impl='avg')
    
    forecast_ret = np.sum(forecast) * 100
    actual_ret = np.sum(test.y) * 100
    
    forecast_dir = "LONG ↑" if forecast_ret > 0 else "SHORT ↓"
    actual_dir = "LONG ↑" if actual_ret > 0 else "SHORT ↓"
    
    print(f"\n   Forecast:  {forecast_dir}  ({forecast_ret:+.3f}%)")
    print(f"   Actual:    {actual_dir}  ({actual_ret:+.3f}%)")
    
    # Direction accuracy
    correct = (forecast_ret > 0) == (actual_ret > 0)
    if correct:
        print(f"\n   ✅ DIRECTION CORRECT!")
    else:
        print(f"\n   ❌ Direction wrong")
    
    # Error
    mae = np.mean(np.abs(forecast - test.y))
    print(f"   MAE: {mae:.6f}")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print("\n" + "─"*70)
    print(" 📊 GENERATING VISUALIZATION")
    print("─"*70)
    
    fig = plot_with_volatility(
        df=df,
        cutoff=test.cutoff,
        forecast_returns=forecast,
        window_size=test.x.shape[-1],
        actual_returns=test.y,
        regime=test.regime,
        title=f"Forecast (k={N_NEIGHBORS}, {DISTANCE_METRIC.upper()}, Regime={test.regime_name})",
        hist_context_bars=100
    )
    
    output_file = "report_playground.html"
    fig.write_html(output_file, full_html=True, auto_open=True)
    print(f"\n   ✅ Saved to {output_file} (opened in browser)")
    
    # =========================================================================
    # HOW TO MODIFY
    # =========================================================================
    print("\n" + "="*70)
    print(" 🎮 HOW TO EXPERIMENT")
    print("="*70)
    print("""
   Edit parameters at the top of this file:
   
   1. Toggle filters ON/OFF:
      REGIME_FILTER_ENABLED = True/False
      CALENDAR_FILTER_ENABLED = True/False
   
   2. Change KNN settings:
      N_NEIGHBORS = 5, 10, 20, ...
      DISTANCE_METRIC = "wdtw", "dtw", "euclidean"
   
   3. Adjust WDTW weight:
      WDTW_G = 0.01 (less tail weight) to 0.1 (more tail weight)
   
   Then run again:
      uv run python config_playground.py
""")
    print("="*70)
