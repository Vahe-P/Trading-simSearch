#!/usr/bin/env python3
"""
PARAMETER PLAYGROUND

Edit the parameters below and run this script to test different configurations.
This makes it easy to experiment with different settings without touching the main code.

Usage:
    uv run python config_playground.py

Output:
    - Detailed console output with regime/calendar stats
    - report_playground.html (opens in browser)
"""

# =============================================================================
# DATA SETTINGS
# =============================================================================
DATA_PATH = "data/test/NQ_2024-09-06_2025-09-13.parquet"

# =============================================================================
# FILTER SETTINGS (toggle on/off)
# =============================================================================

# Regime Filter - match patterns from same volatility regime
REGIME_FILTER_ENABLED = True
VOL_METHOD = "garman_klass"  # Options: "garman_klass", "parkinson"

# Calendar Filter - match FOMC/event context
CALENDAR_FILTER_ENABLED = True
MATCH_FOMC_CONTEXT = True      # FOMC days only match other FOMC days
EXCLUDE_RED_FOLDER = False     # Exclude all high-impact event days (FOMC, CPI, NFP)

# =============================================================================
# KNN SETTINGS
# =============================================================================
N_NEIGHBORS = 10               # K in KNN (more = smoother, less = more reactive)
DISTANCE_METRIC = "wdtw"       # Options: "wdtw", "dtw", "euclidean", "msm"

# WDTW weight parameter (only used if DISTANCE_METRIC = "wdtw")
# Higher = more weight on recent bars (tail of pattern)
WDTW_G = 0.05                  # Default: 0.05

# =============================================================================
# WINDOW SETTINGS
# =============================================================================
from datetime import time

WINDOW_START = time(8, 0)      # Window start time (8:00 PM = overnight start)
WINDOW_END = time(9, 30)       # Window end time (9:30 AM = market open)
EXTEND_SESSIONS = 1            # 1 = overnight (spans to next day)
HORIZON_LEN = 20               # Bars to forecast ahead
NORM_METHOD = "rolling_zscore"  # Options: "log_returns", "pct_change", "rolling_zscore"

# =============================================================================
# RUN FORECAST
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
    from sim_search.forecaster import similarity_search, forecast_from_neighbors, score_forecast, compute_signal_quality
    from sim_search.visualization import plot_forecast_analysis
    from sim_search.volatility import analyze_regime_transitions
    
    print("\n" + "="*80)
    print(" PARAMETER PLAYGROUND - Regime-Aware Market Similarity Search")
    print("="*80)
    
    # =========================================================================
    # METHODOLOGY OVERVIEW
    # =========================================================================
    print("\n" + "-"*80)
    print(" METHODOLOGY OVERVIEW")
    print("-"*80)
    print("""
   This pipeline finds similar historical patterns and forecasts future returns.
   
   STEP 1: WINDOWING
   |-- Extract fixed time windows (e.g., 8:00 PM -> 9:30 AM overnight sessions)
   |-- Each window = one trading pattern with known future outcome (horizon)
   +-- Normalize patterns using log returns for scale invariance
   
   STEP 2: REGIME CLASSIFICATION (Volatility-Based)
   |-- Compute Garman-Klass volatility for each window (uses OHLC, 5x more efficient)
   |-- Classify into LOW / MEDIUM / HIGH volatility regimes
   +-- Why? Similar patterns in different vol regimes have different follow-through
   
   STEP 3: PRE-FILTERING
   |-- Regime Filter: Only compare against same-regime patterns
   |-- Calendar Filter: Match FOMC days with FOMC days (event context matters)
   +-- Reduces noise, improves signal
   
   STEP 4: KNN SIMILARITY SEARCH
   |-- Algorithm: K-Nearest Neighbors Time Series Regressor (aeon library)
   |-- Distance: WDTW (Weighted DTW) - emphasizes recent bars (tail of pattern)
   +-- Find K most similar historical patterns from filtered pool
   
   STEP 5: FORECAST
   |-- Aggregate K neighbors' future returns (weighted average by distance)
   +-- Direction + magnitude prediction
""")
    
    # =========================================================================
    # CONFIG SUMMARY
    # =========================================================================
    print("-"*80)
    print(" CURRENT CONFIGURATION")
    print("-"*80)
    print("\n   +" + "-"*76 + "+")
    print(f"   |  FILTERS                                                                   |")
    print(f"   |    Regime Filter:   {'ENABLED':12s}  |  Vol Method:    {VOL_METHOD:20s}   |")
    print(f"   |    Calendar Filter: {'ENABLED' if CALENDAR_FILTER_ENABLED else 'DISABLED':12s}  |  Match FOMC:    {'YES' if MATCH_FOMC_CONTEXT else 'NO':20s}   |")
    print("   +" + "-"*76 + "+")
    print(f"   |  KNN SETTINGS                                                              |")
    print(f"   |    Distance Metric: {DISTANCE_METRIC.upper():12s}  |  K Neighbors:   {N_NEIGHBORS:<20d}   |")
    if DISTANCE_METRIC == "wdtw":
        print(f"   |    WDTW Weight (g): {WDTW_G:<12.3f}  |  (higher g = more weight on recent bars)   |")
    print("   +" + "-"*76 + "+")
    print(f"   |  WINDOW SETTINGS                                                           |")
    print(f"   |    Time Range:      {WINDOW_START.strftime('%H:%M')} -> {WINDOW_END.strftime('%H:%M')}   |  Extend Sessions: {EXTEND_SESSIONS} (overnight)          |")
    print(f"   |    Horizon:         {HORIZON_LEN} bars       |  Normalization:   {NORM_METHOD:20s}   |")
    print("   +" + "-"*76 + "+")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    print(f"\n[DATA] Loading from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"   Total bars: {len(df):,}")
    print(f"   Date range: {df.index.min().strftime('%Y-%m-%d')} -> {df.index.max().strftime('%Y-%m-%d')}")
    
    # =========================================================================
    # BUILD WINDOWS
    # =========================================================================
    print("\n[BUILD] Creating window collection...")
    builder = WindowCollectionBuilder(df)
    collection = (builder
        .with_time_anchored_windows(WINDOW_START, WINDOW_END, EXTEND_SESSIONS)
        .with_horizon(HORIZON_LEN)
        .with_normalization(NORM_METHOD)
        .with_volatility(VOL_METHOD)
        .with_calendar_events()
        .build())
    
    print(f"   Total windows: {len(collection)}")
    
    # Split train/test - classifies regimes using ONLY training data (no data leakage!)
    train, test = collection.split_train_test()
    total_train = len(train)
    
    # =========================================================================
    # REGIME ANALYSIS - FULL BREAKDOWN
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 2: REGIME CLASSIFICATION RESULTS")
    print("="*80)
    
    # Count regimes in training set
    regime_counts = Counter(w.regime for w in train)
    
    print(f"""
   Volatility Method: {VOL_METHOD.upper()}
   
   Garman-Klass volatility uses all OHLC data (not just close) and is ~5x more
   statistically efficient than close-to-close volatility.
   
   Formula: var = 0.5*(ln(H/L))^2 - (2*ln(2)-1)*(ln(C/O))^2
   
   Regime thresholds computed from TRAINING DATA ONLY (no look-ahead bias):
   - LOW:    volatility < 33rd percentile
   - MEDIUM: volatility between 33rd and 67th percentile  
   - HIGH:   volatility > 67th percentile
""")
    
    print("   +---------------------------------------------------------------------------+")
    print("   |                    TRAINING SET: ALL {:4d} WINDOWS                        |".format(total_train))
    print("   +-----------+-------+---------+-------------------------------------------+")
    print("   | Regime    | Count | Percent | Distribution                              |")
    print("   +-----------+-------+---------+-------------------------------------------+")
    
    for regime in [0, 1, 2]:
        count = regime_counts.get(regime, 0)
        pct = count / total_train * 100 if total_train > 0 else 0
        name = REGIME_NAMES[regime]
        bar = '#' * int(pct / 2.5)
        regime_marker = '[L]' if regime == 0 else '[M]' if regime == 1 else '[H]'
        print(f"   | {regime_marker} {name:6s} | {count:5d} | {pct:6.1f}% | {bar:41s} |")
    
    print("   +-----------+-------+---------+-------------------------------------------+")
    
    # Test window regime
    test_marker = '[L]' if test.regime == 0 else '[M]' if test.regime == 1 else '[H]'
    print(f"""
   TEST WINDOW (what we're forecasting):
      |-- Cutoff:     {test.cutoff}
      |-- Volatility: {test.volatility:.6f}
      +-- Regime:     {test_marker} {test.regime_name}
""")
    
    # =========================================================================
    # REGIME TRANSITION ANALYSIS - Is the market changing?
    # =========================================================================
    print("-"*80)
    print(" REGIME TRANSITION ANALYSIS")
    print("-"*80)
    
    # Get regime history and transitions
    train_regimes = np.array([w.regime for w in train])
    train_cutoffs = [w.cutoff for w in train]
    
    regime_transitions = analyze_regime_transitions(
        regimes=train_regimes,
        cutoffs=train_cutoffs,
        current_regime=test.regime
    )
    
    # Stability indicator
    if regime_transitions['regime_stability'] == "STABLE":
        stability_icon = "[STABLE]"
        stability_bar = "##########"
    elif regime_transitions['regime_stability'] == "RECENT_CHANGE":
        stability_icon = "[CHANGE]"
        stability_bar = "#####-----"
    else:
        stability_icon = "[UNSTBL]"
        stability_bar = "##--------"
    
    print(f"""
   The system detects REGIME CHANGES to warn when:
   "The market is about to stop behaving normally."
   
   +----------------------------------------------------------------------------+
   |  REGIME STABILITY:  {regime_transitions['regime_stability']:12s}  {stability_icon}  {stability_bar}              |
   +----------------------------------------------------------------------------+
   
   CURRENT STATE:
      |-- Current regime:         {regime_transitions['current_regime_name']}
      |-- Windows in this regime: {regime_transitions['windows_in_current_regime']} consecutive
      |-- Transitions (last 10):  {regime_transitions['transitions_last_10']}
      +-- Transitions (last 20):  {regime_transitions['transitions_last_20']}
""")
    
    # Regime timeline (last 5 periods)
    if regime_transitions['regime_history']:
        print("   RECENT REGIME HISTORY (newest first):")
        print("   +----------+----------+")
        print("   | Regime   | Windows  |")
        print("   +----------+----------+")
        for regime_name, count in regime_transitions['regime_history']:
            print(f"   | {regime_name:8s} | {count:8d} |")
        print("   +----------+----------+")
    
    # Warning if transitioning
    if regime_transitions['is_transitioning']:
        print(f"""
   *** WARNING: REGIME TRANSITION DETECTED ***
   The test window is in {regime_transitions['current_regime_name']} regime,
   but the most recent training window was in a DIFFERENT regime.
   
   This suggests the market structure may be changing.
   -> Higher uncertainty, neighbors may be less predictive.
""")
    
    if regime_transitions['regime_stability'] == "UNSTABLE":
        print("""
   *** WARNING: UNSTABLE REGIME ***
   Multiple regime changes in recent history.
   The market is NOT behaving normally.
   -> Consider staying flat or reducing position size.
""")
    
    # =========================================================================
    # CALENDAR EVENT ANALYSIS
    # =========================================================================
    print("-"*80)
    print(" CALENDAR EVENT ANALYSIS (Red Folder Days)")
    print("-"*80)
    
    # Count events in training
    fomc_count = sum(1 for w in train if w.is_fomc_day)
    cpi_count = sum(1 for w in train if w.is_cpi_day)
    nfp_count = sum(1 for w in train if w.is_nfp_day)
    event_days = sum(1 for w in train if w.is_fomc_day or w.is_cpi_day or w.is_nfp_day)
    
    print(f"""
   "Red folder" events = high-impact macro releases that change market dynamics.
   Patterns on event days should only match other event days.
   
   Training Set Event Days:
   +--------------------------------------------+
   | FOMC Decision Days:       {fomc_count:4d} windows     |
   | CPI Release Days:         {cpi_count:4d} windows     |
   | NFP Release Days:         {nfp_count:4d} windows     |
   +--------------------------------------------+
   | Total Red Folder:         {event_days:4d} windows     |
   | ({event_days/total_train*100:.1f}% of training set)                |
   +--------------------------------------------+
""")
    
    # Test window calendar info
    test_events = []
    if test.is_fomc_day: test_events.append("FOMC")
    if test.is_cpi_day: test_events.append("CPI")
    if test.is_nfp_day: test_events.append("NFP")
    
    print(f"   TEST WINDOW CALENDAR:")
    print(f"      |-- FOMC day:       {'YES' if test.is_fomc_day else 'NO'}")
    print(f"      |-- CPI day:        {'YES' if test.is_cpi_day else 'NO'}")
    print(f"      |-- NFP day:        {'YES' if test.is_nfp_day else 'NO'}")
    print(f"      +-- Days since FOMC: {test.days_since_fomc}")
    if test_events:
        print(f"\n      ** THIS IS A RED FOLDER DAY: {', '.join(test_events)} **")
    
    # =========================================================================
    # FILTERING - CLEAR BREAKDOWN
    # =========================================================================
    print("\n" + "="*80)
    print(" STEP 3: PRE-FILTERING (What's KEPT vs IGNORED)")
    print("="*80)
    
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
        
        # Detailed filtering breakdown
        print(f"""
   Why filter? To avoid comparing apples to oranges:
   - A pattern in a quiet market (LOW vol) behaves differently than in chaos (HIGH vol)
   - FOMC announcement days have unique dynamics
   
   TEST WINDOW is: {test_marker} {test.regime_name} regime{', FOMC day' if test.is_fomc_day else ''}
""")
        
        print("   +---------------------------------------------------------------------------+")
        print("   |                         FILTERING BREAKDOWN                               |")
        print("   +-----------+-------+--------------+--------------------------------------+")
        print("   | Regime    | Count | Status       | Reason                               |")
        print("   +-----------+-------+--------------+--------------------------------------+")
        
        for regime in [0, 1, 2]:
            count = regime_counts.get(regime, 0)
            name = REGIME_NAMES[regime]
            regime_marker = '[L]' if regime == 0 else '[M]' if regime == 1 else '[H]'
            
            if regime == test.regime:
                status = "KEPT"
                reason = f"Matches test regime ({test.regime_name})"
            else:
                status = "IGNORED"
                reason = f"Different regime than test"
            
            print(f"   | {regime_marker} {name:6s} | {count:5d} | {status:12s} | {reason:36s} |")
        
        print("   +-----------+-------+--------------+--------------------------------------+")
        
        # Calendar filter effect
        if CALENDAR_FILTER_ENABLED and MATCH_FOMC_CONTEXT:
            kept_fomc = sum(1 for i in filtered_indices if train[i].is_fomc_day)
            kept_non_fomc = len(filtered_indices) - kept_fomc
            print(f"""
   Calendar Filter Effect:
   |-- Test is {'FOMC day' if test.is_fomc_day else 'NON-FOMC day'}
   +-- {'Only FOMC days kept' if test.is_fomc_day else 'Only non-FOMC days kept'}
       -> {kept_fomc if test.is_fomc_day else kept_non_fomc} windows in final pool
""")
        
        print(f"""
   ==============================================================================
   FILTER SUMMARY:
      Before filtering: {total_train:4d} windows (all regimes)
      After filtering:  {len(filtered_train):4d} windows (only {test.regime_name} regime{', matching calendar' if CALENDAR_FILTER_ENABLED else ''})
      Removed:          {total_train - len(filtered_train):4d} windows ({(1 - len(filtered_train)/total_train)*100:.1f}% filtered out)
   ==============================================================================
""")
    else:
        filtered_train = train
        print(f"\n   ** No filters applied, using all {total_train} windows **")
    
    # =========================================================================
    # KNN SIMILARITY SEARCH - DETAILED
    # =========================================================================
    print("="*80)
    print(f" STEP 4: KNN SIMILARITY SEARCH ({DISTANCE_METRIC.upper()})")
    print("="*80)
    
    print(f"""
   ALGORITHM: K-Nearest Neighbors Time Series Regressor
   LIBRARY:   aeon.regression.distance_based.KNeighborsTimeSeriesRegressor
   
   DISTANCE METRIC: {DISTANCE_METRIC.upper()}""")
    
    if DISTANCE_METRIC == "wdtw":
        print(f"""
      Weighted Dynamic Time Warping (WDTW)
      |-- Like DTW but applies weight decay to time steps
      |-- Weight parameter g = {WDTW_G}
      |-- Higher g = MORE weight on recent bars (tail of pattern)
      +-- Why? The 9:30 AM reaction (end of overnight) is most predictive
""")
    elif DISTANCE_METRIC == "dtw":
        print("""
      Dynamic Time Warping (DTW)
      |-- Allows elastic time alignment between patterns
      |-- Good for patterns that are similar but shifted in time
      +-- O(n^2) complexity - slower than Euclidean
""")
    elif DISTANCE_METRIC == "euclidean":
        print("""
      Euclidean Distance
      |-- Simple point-to-point distance
      |-- Fast: O(n) complexity
      +-- Requires patterns to be exactly aligned in time
""")
    
    print(f"""   K NEIGHBORS: {N_NEIGHBORS}
      |-- Lower K (3-5):  More reactive, captures local patterns, higher variance
      +-- Higher K (15+): Smoother forecasts, more robust, may miss recent shifts
   
   SEARCH POOL: {len(filtered_train)} windows (after filtering)
""")
    
    # Convert to format expected by similarity_search
    x_train_list = [pd.DataFrame(w.x) for w in filtered_train]
    x_test_df = pd.DataFrame(test.x)
    
    # Set distance params
    distance_params = None
    if DISTANCE_METRIC == "wdtw":
        distance_params = {"g": WDTW_G}
    
    print(f"   [SEARCH] Computing {DISTANCE_METRIC.upper()} distances against {len(filtered_train)} patterns...")
    
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
    # NEIGHBORS FOUND - DETAILED TABLE
    # =========================================================================
    print(f"\n   Found {len(neighbor_idx)} most similar patterns:\n")
    print("   +-----+--------------------------+------------+--------+---------+-----------+")
    print("   |  #  | Pattern Date             | Distance   | Regime | FOMC?   | Horizon % |")
    print("   +-----+--------------------------+------------+--------+---------+-----------+")
    
    horizon_returns = []
    neighbor_windows = []
    for i, (idx, dist) in enumerate(zip(neighbor_idx, neighbor_dists)):
        w = filtered_train[idx]
        neighbor_windows.append(w)
        fomc_str = "YES" if w.is_fomc_day else "NO"
        horizon_ret = np.sum(w.y) * 100  # Convert to percent
        horizon_returns.append(horizon_ret)
        
        # Format distance: use scientific notation for very small values
        dist_str = f"{dist:.2e}" if dist < 0.0001 and dist > 0 else f"{dist:.4f}"
        
        # Direction indicator
        dir_char = "+" if horizon_ret > 0 else "-" if horizon_ret < 0 else " "
        regime_marker = '[L]' if w.regime == 0 else '[M]' if w.regime == 1 else '[H]'
        
        print(f"   | {i+1:3d} | {str(w.cutoff)[:24]:24s} | {dist_str:>10s} | {regime_marker:6s} | {fomc_str:7s} | {dir_char}{abs(horizon_ret):8.4f}% |")
    
    print("   +-----+--------------------------+------------+--------+---------+-----------+")
    
    # Neighbor statistics
    up_neighbors = sum(1 for r in horizon_returns if r > 0)
    down_neighbors = len(horizon_returns) - up_neighbors
    avg_neighbor_ret = np.mean(horizon_returns)
    
    print(f"""
   NEIGHBOR CONSENSUS:
      Bullish neighbors:    {up_neighbors:2d} ({up_neighbors/len(horizon_returns)*100:.0f}%)
      Bearish neighbors:    {down_neighbors:2d} ({down_neighbors/len(horizon_returns)*100:.0f}%)
      Average return:       {avg_neighbor_ret:+.4f}%
""")
    
    # =========================================================================
    # FORECAST - COMPREHENSIVE METRICS
    # =========================================================================
    print("="*80)
    print(" STEP 5: FORECAST RESULT & STATISTICAL EVALUATION")
    print("="*80)
    
    # Get neighbor horizons and forecast
    neighbor_horizons = np.stack([filtered_train[i].y for i in neighbor_idx])
    forecast = forecast_from_neighbors(neighbor_horizons, neighbor_dists, impl='avg')
    
    # Calculate all metrics
    score = score_forecast(forecast, test.y)
    
    forecast_ret = np.sum(forecast) * 100
    actual_ret = np.sum(test.y) * 100
    
    forecast_dir = "LONG" if forecast_ret > 0 else "SHORT"
    actual_dir = "UP" if actual_ret > 0 else "DOWN"
    
    print(f"""
   FORECAST AGGREGATION METHOD: Distance-Weighted Average
   |-- Closer neighbors (lower distance) get higher weight
   +-- Weight = 1 / (distance + epsilon)
   
   +----------------------------------------------------------------------------+
   |                           FORECAST vs ACTUAL                               |
   +----------------------------------------------------------------------------+
   |                                                                            |
   |   FORECAST DIRECTION:  {forecast_dir:10s}                                        |
   |   FORECAST RETURN:     {forecast_ret:+.6f}%                                        |
   |                                                                            |
   |   ACTUAL DIRECTION:    {actual_dir:10s}                                          |
   |   ACTUAL RETURN:       {actual_ret:+.6f}%                                         |
   |                                                                            |
   +----------------------------------------------------------------------------+
""")
    
    # Direction accuracy
    correct = (forecast_ret > 0) == (actual_ret > 0)
    
    print(f"""
   ==============================================================================
                              STATISTICAL METRICS
   ==============================================================================
   
   DIRECTION ACCURACY:  {'CORRECT' if correct else 'WRONG':20s}
      +-- Did we predict the right direction? {'YES' if correct else 'NO'}
   
   RMSE (Root Mean Squared Error): {score['rmse']:.6f}
      +-- Average magnitude of prediction errors (lower = better)
      +-- Penalizes large errors more than small ones
   
   MAE (Mean Absolute Error):      {score['mae']:.6f}
      +-- Average absolute difference between forecast and actual
      +-- More robust to outliers than RMSE
   
   R-squared (Coefficient of Determination): {score['r2']:+.4f}
      +-- How much variance in actual returns is explained by forecast
      +-- Range: -inf to 1.0 (1.0 = perfect, 0 = baseline, <0 = worse than mean)
   
   MSE (Mean Squared Error):       {score['mse']:.8f}
      +-- RMSE^2 - useful for comparing models
   
   ==============================================================================
""")
    
    # Per-bar breakdown
    print("   PER-BAR FORECAST vs ACTUAL (first 10 bars):")
    print("   +------+------------+------------+------------+")
    print("   | Bar  | Forecast   | Actual     | Error      |")
    print("   +------+------------+------------+------------+")
    for i in range(min(10, len(forecast))):
        err = forecast[i] - test.y[i]
        print(f"   | {i+1:4d} | {forecast[i]*100:+9.4f}% | {test.y[i]*100:+9.4f}% | {err*100:+9.4f}% |")
    if len(forecast) > 10:
        print(f"   |  ... |    ...     |    ...     |    ...     |")
    print("   +------+------------+------------+------------+")
    
    # =========================================================================
    # SIGNAL QUALITY - Confidence + Anomaly Detection
    # =========================================================================
    
    signal_quality = compute_signal_quality(neighbor_horizons, neighbor_dists)
    
    up_avg = np.mean([r for r in horizon_returns if r > 0]) if up_neighbors > 0 else 0
    down_avg = np.mean([r for r in horizon_returns if r < 0]) if down_neighbors > 0 else 0
    
    # Visual indicators for signal
    if signal_quality['signal'] == "TRADE":
        signal_icon = "[***]"
        signal_bar = "##########"
    elif signal_quality['signal'] == "CAUTION":
        signal_icon = "[**-]"
        signal_bar = "######----"
    else:
        signal_icon = "[---]"
        signal_bar = "----------"
    
    conf_bar = "#" * int(signal_quality['confidence'] * 10) + "-" * (10 - int(signal_quality['confidence'] * 10))
    anom_bar = "#" * int(signal_quality['anomaly_score'] * 10) + "-" * (10 - int(signal_quality['anomaly_score'] * 10))
    
    print(f"""
   ==============================================================================
                         SIGNAL QUALITY ASSESSMENT
   ==============================================================================
   
   This system does NOT just say "price will go up."
   It tells you: "How confident am I? Is this pattern unusual?"
   
   +----------------------------------------------------------------------------+
   |                                                                            |
   |   SIGNAL:      {signal_quality['signal']:12s}  {signal_icon}  {signal_bar}                    |
   |   DIRECTION:   {signal_quality['direction']:12s}                                            |
   |   STRENGTH:    {signal_quality['signal_strength']:12s}                                            |
   |                                                                            |
   +----------------------------------------------------------------------------+
   
   CONFIDENCE SCORE: {signal_quality['confidence']*100:5.1f}%  [{conf_bar}]
      |-- What % of neighbors agree on direction?
      |-- {signal_quality['stats']['up_count']}/{len(neighbor_idx)} bullish, {signal_quality['stats']['down_count']}/{len(neighbor_idx)} bearish
      +-- High confidence (>70%) = neighbors agree, clearer signal
   
   ANOMALY SCORE:    {signal_quality['anomaly_score']*100:5.1f}%  [{anom_bar}]
      |-- Are the neighbors far away? (unusual pattern)
      |-- Avg distance: {signal_quality['stats']['avg_distance']:.2e}
      +-- High anomaly (>50%) = pattern is unusual, expect volatility
   
   ==============================================================================
                              INTERPRETATION
   ==============================================================================
""")
    
    for line in signal_quality['interpretation']:
        print(f"   {line}")
    
    # Add regime stability warning if needed
    if regime_transitions['regime_stability'] == "UNSTABLE":
        print("""
   *** REGIME INSTABILITY WARNING ***
   Multiple regime changes detected. Market structure is changing.
   >> This amplifies any ANOMALY signal - proceed with extra caution.
""")
    elif regime_transitions['is_transitioning']:
        print("""
   ** REGIME TRANSITION NOTE **
   Current window is in a different regime than recent history.
   >> Neighbors may be from an older regime - watch for divergence.
""")
    
    print(f"""
   ------------------------------------------------------------------------------
   
   WHAT THIS MEANS:
   
   We found {len(neighbor_idx)} historical patterns from {test.regime_name}-volatility days that
   looked similar to today's pattern (ending at {str(test.cutoff)[:19]}).
   
   Of those {len(neighbor_idx)} similar patterns:
   - {up_neighbors} went UP afterward (avg: {up_avg:.4f}%)
   - {down_neighbors} went DOWN afterward (avg: {down_avg:.4f}%)
   
   REGIME CONTEXT:
   - Current regime: {regime_transitions['current_regime_name']}
   - Stability: {regime_transitions['regime_stability']}
   - Consecutive windows in regime: {regime_transitions['windows_in_current_regime']}
   
   If ANOMALY is HIGH or REGIME is UNSTABLE:
      "The market is about to stop behaving normally."
      -> This is NOT a failure - it's valuable information.
      -> High anomaly/instability = unusual market structure, expect surprises.
   
   ==============================================================================
""")
    
    # =========================================================================
    # VISUALIZATION - Enhanced 4-panel analysis
    # =========================================================================
    print("-"*80)
    print(" GENERATING VISUALIZATION")
    print("-"*80)
    
    # Add regime stability info to signal_quality for HTML report
    signal_quality['regime_stability'] = regime_transitions['regime_stability']
    signal_quality['regime_history'] = regime_transitions['regime_history']
    signal_quality['is_transitioning'] = regime_transitions['is_transitioning']
    signal_quality['windows_in_regime'] = regime_transitions['windows_in_current_regime']
    
    # Create regime timeline data for visualization
    regime_timeline = {
        'cutoffs': train_cutoffs,
        'regimes': train_regimes.tolist()
    }
    
    fig = plot_forecast_analysis(
        df=df,
        cutoff=test.cutoff,
        forecast_returns=forecast,
        actual_returns=test.y,
        neighbor_windows=neighbor_windows,
        neighbor_distances=neighbor_dists,
        score_dict=score,
        regime=test.regime,
        signal_quality=signal_quality,
        regime_timeline=regime_timeline,
        title=f"Forecast Analysis (k={N_NEIGHBORS}, {DISTANCE_METRIC.upper()})",
        hist_context_bars=100
    )
    
    output_file = "report_playground.html"
    fig.write_html(output_file, full_html=True, auto_open=True)
    print(f"\n   Saved to {output_file} (opened in browser)")
    
    print("""
   PLOT CONTENTS:
   1. Top-Left:     Price forecast vs actual with regime shading
   2. Top-Right:    Neighbor quality scatter (distance vs return)
   3. Bottom-Left:  Cumulative return comparison over horizon
   4. Bottom-Right: Residual analysis (forecast errors per bar)
""")
    
    # =========================================================================
    # HOW TO EXPERIMENT
    # =========================================================================
    print("\n" + "="*80)
    print(" HOW TO EXPERIMENT")
    print("="*80)
    print("""
   Edit parameters at the TOP of this file (config_playground.py):
   
   +--------------------------------------------------------------------------+
   | FILTERS                                                                  |
   |   REGIME_FILTER_ENABLED = True/False   # Match same vol regime          |
   |   CALENDAR_FILTER_ENABLED = True/False # Match FOMC context             |
   +--------------------------------------------------------------------------+
   | KNN SETTINGS                                                             |
   |   N_NEIGHBORS = 5, 10, 15, 20...       # More = smoother forecast       |
   |   DISTANCE_METRIC = "wdtw" | "dtw" | "euclidean"                        |
   |   WDTW_G = 0.01 to 0.1                 # Tail weight (only for WDTW)    |
   +--------------------------------------------------------------------------+
   | WINDOW SETTINGS                                                          |
   |   HORIZON_LEN = 10, 20, 30...          # Bars to forecast ahead         |
   |   NORM_METHOD = "log_returns" | "pct_change" | "rolling_zscore"         |
   +--------------------------------------------------------------------------+
   
   Then run again:
      uv run python config_playground.py
""")
    print("="*80)
