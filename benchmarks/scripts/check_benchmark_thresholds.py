#!/usr/bin/env python3
"""
Check benchmark results against performance thresholds.

Usage:
    python check_benchmark_thresholds.py results_stats.csv
"""

import sys
import csv
from pathlib import Path


# Performance thresholds
THRESHOLDS = {
    "min_rps": 100,  # Minimum requests per second
    "max_p50_ms": 3000,  # Maximum P50 latency (ms)
    "max_p95_ms": 8000,  # Maximum P95 latency (ms)
    "max_p99_ms": 15000,  # Maximum P99 latency (ms)
    "max_failure_rate": 0.05,  # Maximum 5% failure rate
    "min_users": 10,  # Minimum concurrent users for valid test
}


def parse_csv_stats(csv_path):
    """Parse Locust stats CSV file."""
    stats = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Type'] == 'Aggregated':
                stats.append({
                    'name': row['Name'],
                    'requests': int(row['Request Count']),
                    'failures': int(row['Failure Count']),
                    'median': float(row['Median Response Time']),
                    'p95': float(row['95%']),
                    'p99': float(row['99%']),
                    'avg': float(row['Average Response Time']),
                    'min': float(row['Min Response Time']),
                    'max': float(row['Max Response Time']),
                    'rps': float(row['Requests/s']),
                })
    
    return stats


def check_thresholds(stats):
    """Check if stats meet performance thresholds."""
    if not stats:
        print("❌ No aggregated stats found in CSV")
        return False
    
    aggregated = stats[0]  # First row is typically aggregated
    
    failures = []
    warnings = []
    successes = []
    
    # Check requests per second
    if aggregated['rps'] < THRESHOLDS['min_rps']:
        failures.append(
            f"Throughput too low: {aggregated['rps']:.1f} req/s "
            f"(threshold: {THRESHOLDS['min_rps']} req/s)"
        )
    else:
        successes.append(
            f"✅ Throughput: {aggregated['rps']:.1f} req/s "
            f"(target: {THRESHOLDS['min_rps']}+)"
        )
    
    # Check P50 latency
    if aggregated['median'] > THRESHOLDS['max_p50_ms']:
        failures.append(
            f"P50 latency too high: {aggregated['median']:.0f}ms "
            f"(threshold: {THRESHOLDS['max_p50_ms']}ms)"
        )
    else:
        successes.append(
            f"✅ P50 latency: {aggregated['median']:.0f}ms "
            f"(target: <{THRESHOLDS['max_p50_ms']}ms)"
        )
    
    # Check P95 latency
    if aggregated['p95'] > THRESHOLDS['max_p95_ms']:
        failures.append(
            f"P95 latency too high: {aggregated['p95']:.0f}ms "
            f"(threshold: {THRESHOLDS['max_p95_ms']}ms)"
        )
    else:
        successes.append(
            f"✅ P95 latency: {aggregated['p95']:.0f}ms "
            f"(target: <{THRESHOLDS['max_p95_ms']}ms)"
        )
    
    # Check P99 latency
    if aggregated['p99'] > THRESHOLDS['max_p99_ms']:
        warnings.append(
            f"⚠️  P99 latency high: {aggregated['p99']:.0f}ms "
            f"(threshold: {THRESHOLDS['max_p99_ms']}ms)"
        )
    else:
        successes.append(
            f"✅ P99 latency: {aggregated['p99']:.0f}ms "
            f"(target: <{THRESHOLDS['max_p99_ms']}ms)"
        )
    
    # Check failure rate
    failure_rate = aggregated['failures'] / aggregated['requests'] if aggregated['requests'] > 0 else 0
    if failure_rate > THRESHOLDS['max_failure_rate']:
        failures.append(
            f"Failure rate too high: {failure_rate*100:.2f}% "
            f"(threshold: {THRESHOLDS['max_failure_rate']*100}%)"
        )
    else:
        successes.append(
            f"✅ Failure rate: {failure_rate*100:.2f}% "
            f"(target: <{THRESHOLDS['max_failure_rate']*100}%)"
        )
    
    # Print results
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\nTotal Requests: {aggregated['requests']}")
    print(f"Total Failures: {aggregated['failures']}")
    print(f"Test Duration: ~{aggregated['requests'] / aggregated['rps']:.0f}s")
    
    print("\n--- Performance Checks ---")
    for success in successes:
        print(success)
    
    if warnings:
        print("\n--- Warnings ---")
        for warning in warnings:
            print(warning)
    
    if failures:
        print("\n--- FAILURES ---")
        for failure in failures:
            print(f"❌ {failure}")
        print("\n" + "="*80)
        print("BENCHMARK FAILED - Performance thresholds not met")
        print("="*80)
        return False
    else:
        print("\n" + "="*80)
        print("BENCHMARK PASSED - All thresholds met ✅")
        print("="*80)
        return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_benchmark_thresholds.py results_stats.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        sys.exit(1)
    
    try:
        stats = parse_csv_stats(csv_path)
        passed = check_thresholds(stats)
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"❌ Error processing benchmark results: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
