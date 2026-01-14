"""Fast parameter comparison using resampled 5-min data."""
from datetime import time
from sim_search.backtester import BacktestConfig, run_backtest_from_file

# Use 5-min resample for MUCH faster testing
base = {
    'data_path': 'data/test/NQ_2024-09-06_2025-09-13.parquet',
    'window_start_time': time(8, 0),
    'window_end_time': time(9, 30),
    'extend_sessions': 1,
    'min_train_windows': 50,
    'tp_threshold': 0.005,
    'sl_threshold': 0.005,
    'max_test_days': 50,
    'timezone': 'America/New_York',
    'resample': '5min',  # KEY: Resample to 5-min bars (~5x faster)
}

configs = [
    ('Baseline k=7', 7, 'euclidean', 'log_returns', 20),
    ('More k=15', 15, 'euclidean', 'log_returns', 20),
    ('Horizon=10', 10, 'euclidean', 'log_returns', 10),
    ('k=20', 20, 'euclidean', 'log_returns', 20),
]

print("=" * 60)
print("FAST PARAMETER COMPARISON (5-min bars)")
print("=" * 60)

results = []
for name, k, dist, norm, h in configs:
    print(f"\nTesting: {name}...")
    cfg = BacktestConfig(**base, n_neighbors=k, distance_metric=dist, norm_method=norm, forecast_horizon=h)
    r = run_backtest_from_file(cfg.data_path, config=cfg, verbose=False)
    results.append((name, r.total_trades, r.direction_accuracy, r.hit_ratio, r.profit_factor))
    print(f'  Dir={r.direction_accuracy:.1%}, Hit={r.hit_ratio:.1%}, PF={r.profit_factor:.2f}')

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Config':<20} {'Trades':>7} {'Dir%':>7} {'Hit%':>7} {'PF':>7}")
print("-" * 60)
for name, trades, dir_acc, hit, pf in results:
    print(f"{name:<20} {trades:>7} {dir_acc:>6.1%} {hit:>6.1%} {pf:>7.2f}")

best = max(results, key=lambda x: x[4])
print(f"\n>>> Best by Profit Factor: {best[0]} (PF={best[4]:.2f})")
