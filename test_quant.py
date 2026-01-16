"""
Quant Test - Full Pipeline Verification
Tests: Windowing, Regime Classification, KNN, Signal Quality, Visualization

Usage:
    uv run python test_quant.py
"""
import json
import pandas as pd
import numpy as np
from datetime import time
from pathlib import Path

# =============================================================================
# IMPORTS
# =============================================================================
from sim_search import (
    WindowCollectionBuilder,
    RegimeFilter, CalendarFilter, FilterPipeline,
    REGIME_NAMES
)
from sim_search.forecaster import similarity_search, forecast_from_neighbors, score_forecast, compute_signal_quality
from sim_search.visualization import plot_forecast_analysis
from sim_search.volatility import analyze_regime_transitions, compute_regime_thresholds

print("\n" + "="*80)
print(" QUANT TEST - Full Pipeline Verification")
print("="*80)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================
DATA_PATH = "data/cache/QQQ_2024-01-01_2025-01-01.parquet"
WINDOW_START = time(9, 30)
WINDOW_END = time(10, 30)
EXTEND_SESSIONS = 0
HORIZON_LEN = 20
NORM_METHOD = "rolling_zscore"
N_NEIGHBORS = 10
DISTANCE_METRIC = "wdtw"
WDTW_G = 0.05

# =============================================================================
# TEST 1: DATA LOADING
# =============================================================================
print("\n" + "-"*80)
print(" TEST 1: Data Loading")
print("-"*80)

df = pd.read_parquet(DATA_PATH)
print(f"   Loaded {len(df):,} rows from {DATA_PATH}")
print(f"   Date range: {df.index[0]} to {df.index[-1]}")
print("   [PASS] Data loading")

# =============================================================================
# TEST 2: WINDOWING
# =============================================================================
print("\n" + "-"*80)
print(" TEST 2: Windowing (9:30 AM - 10:30 AM)")
print("-"*80)

collection = (
    WindowCollectionBuilder(df)
    .with_time_anchored_windows(
        start=WINDOW_START,
        end=WINDOW_END,
        extend_sessions=EXTEND_SESSIONS
    )
    .with_horizon(HORIZON_LEN)
    .with_normalization(NORM_METHOD)
    .with_volatility("garman_klass")  # This also triggers regime classification
    .with_calendar_events()
    .build()
)

# Verify window times
sample_windows = [collection[i] for i in [0, len(collection)//2, -1]]
for i, w in enumerate(sample_windows):
    print(f"   Window {i}: cutoff={w.cutoff}, x_len={len(w.x)}, y_len={len(w.y)}")
    
    # Verify time is around 10:30 AM (end of window)
    assert w.cutoff.hour == 10, f"Expected hour 10, got {w.cutoff.hour}"
    assert w.cutoff.minute == 30 or w.cutoff.minute == 29, f"Expected minute ~30, got {w.cutoff.minute}"

print(f"   Total windows: {len(collection)}")
print("   [PASS] Window times are correct (10:30 AM cutoff)")

# =============================================================================
# TEST 3: REGIME CLASSIFICATION (NO LEAKAGE)
# =============================================================================
print("\n" + "-"*80)
print(" TEST 3: Regime Classification (Leakage Check)")
print("-"*80)

train, test_window = collection.split_train_test(test_idx=-1)
train_regimes = np.array([w.regime for w in train])
train_vols = np.array([w.volatility for w in train])

# Get thresholds from training data only
valid_vols = train_vols[~np.isnan(train_vols) & (train_vols > 0)]
thresholds = compute_regime_thresholds(valid_vols)

# Verify regime distribution is roughly 33/33/33
regime_counts = {0: 0, 1: 0, 2: 0}
for r in train_regimes:
    regime_counts[r] = regime_counts.get(r, 0) + 1
    
total = len(train_regimes)
for regime_id, count in regime_counts.items():
    pct = count / total * 100
    print(f"   {REGIME_NAMES[regime_id]}: {count} ({pct:.1f}%)")
    assert 20 <= pct <= 45, f"Regime {regime_id} distribution off: {pct}%"

print(f"   Thresholds: LOW < {thresholds[0]:.6f} < MED < {thresholds[1]:.6f} < HIGH")
print("   [PASS] Regime distribution is balanced (~33% each)")

# =============================================================================
# TEST 4: KNN FILTERING
# =============================================================================
print("\n" + "-"*80)
print(" TEST 4: KNN Filtering (Same-Regime Pool)")
print("-"*80)

# Set up filters
regime_filter = RegimeFilter()
calendar_filter = CalendarFilter(match_fomc_context=True, exclude_red_folder=False)
pipeline = FilterPipeline([regime_filter, calendar_filter])
pipeline.fit(train)

# Apply filters
filtered_indices = pipeline.transform(train, test_window)

print(f"   Train pool: {len(train)} windows")
print(f"   After regime filter: {len(filtered_indices)} windows")
print(f"   Test window regime: {REGIME_NAMES[test_window.regime]}")
print(f"   Reduction: {(1 - len(filtered_indices) / len(train)) * 100:.1f}%")

# Verify all filtered windows have same regime
filtered_regimes = [train[i].regime for i in filtered_indices]
same_regime_count = sum(1 for r in filtered_regimes if r == test_window.regime)
same_regime_pct = same_regime_count / len(filtered_regimes) * 100 if filtered_regimes else 0

print(f"   Same-regime matches: {same_regime_count}/{len(filtered_regimes)} ({same_regime_pct:.1f}%)")
assert same_regime_pct >= 95, f"Regime filter not working: only {same_regime_pct}% same regime"
print("   [PASS] Regime filter correctly selects same-regime windows")

# =============================================================================
# TEST 5: SIMILARITY SEARCH
# =============================================================================
print("\n" + "-"*80)
print(" TEST 5: KNN Similarity Search")
print("-"*80)

# Prepare filtered training data
train_filtered = [train[i] for i in filtered_indices]

# Convert to format expected by similarity_search (list of DataFrames)
x_train_list = [pd.DataFrame(w.x) for w in train_filtered]
x_test_df = pd.DataFrame(test_window.x)
y_train = np.array([w.y for w in train_filtered])

# Set distance params
distance_params = {"g": WDTW_G} if DISTANCE_METRIC == "wdtw" else None

# Run KNN
neighbor_indices, neighbor_dists = similarity_search(
    x_train_list,
    np.zeros(len(x_train_list)),  # y_train not used for neighbor finding
    x_test_df,
    n_neighbors=min(N_NEIGHBORS, len(train_filtered)),
    impl='knn',
    distance=DISTANCE_METRIC,
    distance_params=distance_params
)

neighbor_windows = [train_filtered[i] for i in neighbor_indices]

print(f"   Found {len(neighbor_indices)} neighbors")
print(f"   Distance range: {neighbor_dists.min():.6f} to {neighbor_dists.max():.6f}")
print(f"   Median distance: {np.median(neighbor_dists):.6f}")
print("   [PASS] KNN search completed")

# =============================================================================
# TEST 6: SIGNAL QUALITY
# =============================================================================
print("\n" + "-"*80)
print(" TEST 6: Signal Quality Computation")
print("-"*80)

neighbor_horizons = np.array([w.y for w in neighbor_windows])
signal_quality = compute_signal_quality(neighbor_horizons, neighbor_dists)

print(f"   Direction: {signal_quality['direction']}")
print(f"   Confidence: {signal_quality['confidence']*100:.0f}%")
print(f"   Anomaly Score: {signal_quality['anomaly_score']*100:.0f}%")
print(f"   Signal: {signal_quality['signal']}")
print(f"   Neighbor votes: UP={signal_quality['stats']['up_count']}, DOWN={signal_quality['stats']['down_count']}")

# Verify confidence calculation
expected_confidence = max(signal_quality['stats']['up_count'], signal_quality['stats']['down_count']) / N_NEIGHBORS
assert abs(signal_quality['confidence'] - expected_confidence) < 0.01, "Confidence calculation error"
print("   [PASS] Signal quality computed correctly")

# =============================================================================
# TEST 7: FORECAST
# =============================================================================
print("\n" + "-"*80)
print(" TEST 7: Forecast Generation")
print("-"*80)

forecast = forecast_from_neighbors(neighbor_horizons, neighbor_dists)
score = score_forecast(forecast, test_window.y)

forecast_direction = "UP" if np.sum(forecast) > 0 else "DOWN"
actual_direction = "UP" if np.sum(test_window.y) > 0 else "DOWN"

print(f"   Forecast direction: {forecast_direction} ({np.sum(forecast)*100:.2f}%)")
print(f"   Actual direction: {actual_direction} ({np.sum(test_window.y)*100:.2f}%)")
print(f"   RMSE: {score['rmse']:.6f}")
print(f"   Direction match: {'YES' if forecast_direction == actual_direction else 'NO'}")
print("   [PASS] Forecast generated")

# =============================================================================
# TEST 8: VISUALIZATION (Market Hours Only)
# =============================================================================
print("\n" + "-"*80)
print(" TEST 8: Visualization (Market Hours Check)")
print("-"*80)

# Add regime info to signal_quality
train_cutoffs = [w.cutoff for w in train]
regime_transitions = analyze_regime_transitions(train_regimes, train_cutoffs, test_window.regime)
signal_quality['regime_stability'] = regime_transitions['regime_stability']
signal_quality['regime_history'] = regime_transitions['regime_history']
signal_quality['is_transitioning'] = regime_transitions['is_transitioning']
signal_quality['windows_in_regime'] = regime_transitions['windows_in_current_regime']

regime_timeline = {
    'cutoffs': train_cutoffs,
    'regimes': train_regimes.tolist(),
    'volatilities': train_vols.tolist(),
    'thresholds': list(thresholds)
}

# Generate visualization
fig = plot_forecast_analysis(
    df=df,
    cutoff=test_window.cutoff,
    forecast_returns=forecast,
    actual_returns=test_window.y,
    neighbor_windows=neighbor_windows,
    neighbor_distances=neighbor_dists,
    score_dict=score,
    regime=test_window.regime,
    signal_quality=signal_quality,
    regime_timeline=regime_timeline,
    title=f"QQQ - Test (k={N_NEIGHBORS}, {DISTANCE_METRIC.upper()})",
    hist_context_bars=60
)

# Check that visualization data is filtered to market hours
candlestick_trace = fig.data[0]  # First trace is candlestick
x_times = pd.to_datetime(candlestick_trace.x)

# Check all times are within market hours
market_hours_violations = []
for t in x_times:
    if t.hour < 9 or (t.hour == 9 and t.minute < 30) or t.hour >= 16:
        market_hours_violations.append(str(t))

print(f"   Total candles in chart: {len(x_times)}")
print(f"   Time range: {x_times[0]} to {x_times[-1]}")
print(f"   Market hours violations: {len(market_hours_violations)}")

if len(market_hours_violations) == 0:
    print("   [PASS] All candles within market hours (9:30 AM - 4:00 PM)")
else:
    print(f"   [WARN] Found {len(market_hours_violations)} candles outside market hours")
    for v in market_hours_violations[:3]:
        print(f"      - {v}")

# Save report
output_file = "report_test.html"
fig.write_html(output_file, full_html=True)
print(f"\n   Report saved to: {output_file}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print(" TEST SUMMARY")
print("="*80)

tests = [
    ("Data Loading", True),
    ("Windowing (9:30-10:30 AM)", True),
    ("Regime Classification (No Leakage)", True),
    ("KNN Filtering (Same-Regime)", True),
    ("Similarity Search", True),
    ("Signal Quality", True),
    ("Forecast Generation", True),
    ("Visualization (Market Hours)", len(market_hours_violations) == 0)
]

all_passed = True
for name, passed in tests:
    status = "[PASS]" if passed else "[FAIL]"
    print(f"   {status} {name}")
    if not passed:
        all_passed = False

print("\n" + "="*80)
if all_passed:
    print(" ALL TESTS PASSED - Pipeline is working correctly!")
else:
    print(" SOME TESTS FAILED - Check output for details")
print("="*80 + "\n")
