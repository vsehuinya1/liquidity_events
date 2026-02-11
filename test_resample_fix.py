#!/usr/bin/env python3
"""
test_resample_fix.py — Deterministic regression tests for bucket-based 5m aggregation.

Run: python3 test_resample_fix.py
All tests must pass before VPS deployment.
"""

import sys
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Any, Optional

# Import the functions and class under test
from websocket_handler_improved import (
    floor_to_5min,
    validate_5min_bucket,
    resample_bars_to_5min,
    BinanceWebSocketFeed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bar(minute_offset: int, base_time: datetime = None, symbol: str = 'SOLUSDT') -> Dict[str, Any]:
    """Create a synthetic 1m bar at base_time + minute_offset minutes."""
    if base_time is None:
        base_time = datetime(2026, 2, 8, 12, 0)
    ts = base_time + timedelta(minutes=minute_offset)
    price = 100.0 + minute_offset * 0.1
    return {
        'symbol': symbol,
        'timestamp': ts,
        'timestamp_ms': int(ts.timestamp() * 1000),
        'open': price,
        'high': price + 0.05,
        'low': price - 0.05,
        'close': price + 0.02,
        'volume': 1000.0 + minute_offset,
        'n_trades': 100,
        'quote_volume': 100000.0,
        'taker_buy_base': 500.0,
        'taker_buy_quote': 50000.0,
        'is_closed': True,
    }


def make_bars(start_minute: int, end_minute: int, **kwargs) -> List[Dict[str, Any]]:
    """Create consecutive bars from start_minute to end_minute (inclusive)."""
    return [make_bar(m, **kwargs) for m in range(start_minute, end_minute + 1)]


class TestFeed:
    """Minimal wrapper around BinanceWebSocketFeed for testing."""

    def __init__(self):
        self.emitted: List[Dict[str, Any]] = []
        self.log_messages: List[str] = []

        # Suppress actual logging, capture messages
        self.feed = BinanceWebSocketFeed.__new__(BinanceWebSocketFeed)
        self.feed.symbols = ['solusdt']
        self.feed.buffer_size = 100
        self.feed.one_min_buffers = {'solusdt': deque(maxlen=100)}
        self.feed.five_min_buffers = {'solusdt': deque(maxlen=100)}
        self.feed._active_buckets = {}
        self.feed._last_5m_emit_time = {'solusdt': None}
        self.feed.buffer_lock = __import__('threading').Lock()
        self.feed.on_5min_bar_callback = lambda bar: self.emitted.append(bar)
        self.feed.on_1min_bar_callback = None

        # Capture logger
        self.feed.logger = logging.getLogger(f'TestFeed_{id(self)}')
        self.feed.logger.setLevel(logging.DEBUG)
        handler = LogCapture(self.log_messages)
        handler.setLevel(logging.DEBUG)
        self.feed.logger.addHandler(handler)

    def feed_bar(self, bar: Dict[str, Any]):
        self.feed._try_resample('solusdt', bar)

    def feed_bars(self, bars: List[Dict[str, Any]]):
        for bar in bars:
            self.feed_bar(bar)

    def check_health(self, bar_ts: datetime):
        self.feed._check_5m_health('solusdt', bar_ts)

    def pre_fill_buffer(self, bars: List[Dict[str, Any]]):
        for bar in bars:
            self.feed.one_min_buffers['solusdt'].append(bar)

    def run_startup(self):
        self.feed._attempt_startup_aggregation()

    @property
    def emit_count(self) -> int:
        return len(self.emitted)

    def has_log_level(self, level: str) -> bool:
        return any(level in msg for msg in self.log_messages)

    def has_log_containing(self, text: str) -> bool:
        return any(text in msg for msg in self.log_messages)


class LogCapture(logging.Handler):
    def __init__(self, messages: list):
        super().__init__()
        self.messages = messages

    def emit(self, record):
        self.messages.append(f"[{record.levelname}] {record.getMessage()}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_emission():
    """Feed :00-:04, then :05 boundary trigger → exactly 1 bar emitted."""
    tf = TestFeed()
    bars = make_bars(0, 4)  # 12:00 - 12:04
    tf.feed_bars(bars)
    assert tf.emit_count == 0, f"Expected 0 emits before boundary, got {tf.emit_count}"

    # Trigger: first bar of next bucket
    tf.feed_bar(make_bar(5))  # 12:05
    assert tf.emit_count == 1, f"Expected 1 emit after boundary, got {tf.emit_count}"
    assert tf.emitted[0]['timestamp'] == datetime(2026, 2, 8, 12, 0), \
        f"Expected ts 12:00, got {tf.emitted[0]['timestamp']}"
    print("  ✅ test_basic_emission")


def test_consecutive_emission():
    """Feed :00-:09, then :10 trigger → exactly 2 bars emitted."""
    tf = TestFeed()
    bars = make_bars(0, 9)  # 12:00 - 12:09
    tf.feed_bars(bars)
    assert tf.emit_count == 1, f"Expected 1 emit at :05 boundary, got {tf.emit_count}"

    tf.feed_bar(make_bar(10))  # 12:10 triggers :05 bucket
    assert tf.emit_count == 2, f"Expected 2 emits after :10, got {tf.emit_count}"
    assert tf.emitted[0]['timestamp'] == datetime(2026, 2, 8, 12, 0)
    assert tf.emitted[1]['timestamp'] == datetime(2026, 2, 8, 12, 5)
    print("  ✅ test_consecutive_emission")


def test_gap_anomaly():
    """Feed :00, :01, :03, :04 (skip :02) → invariant fail, no emit."""
    tf = TestFeed()
    bars = [make_bar(0), make_bar(1), make_bar(3), make_bar(4)]
    tf.feed_bars(bars)

    # Trigger with :05
    tf.feed_bar(make_bar(5))
    assert tf.emit_count == 0, f"Expected 0 emits with gap, got {tf.emit_count}"
    assert tf.has_log_containing("INVARIANT FAIL"), "Expected INVARIANT FAIL log"
    assert tf.has_log_containing("1m GAP"), "Expected 1m GAP log"
    print("  ✅ test_gap_anomaly")


def test_out_of_order():
    """Feed :00, :01, :00 → rejected at insert time."""
    tf = TestFeed()
    tf.feed_bar(make_bar(0))
    tf.feed_bar(make_bar(1))
    tf.feed_bar(make_bar(0))  # out-of-order
    assert tf.has_log_containing("OUT-OF-ORDER"), "Expected OUT-OF-ORDER log"
    print("  ✅ test_out_of_order")


def test_duplicate_rejection():
    """Feed :00, :01, :01 → rejected at insert time."""
    tf = TestFeed()
    tf.feed_bar(make_bar(0))
    tf.feed_bar(make_bar(1))
    tf.feed_bar(make_bar(1))  # duplicate
    assert tf.has_log_containing("OUT-OF-ORDER"), "Expected OUT-OF-ORDER for duplicate"
    print("  ✅ test_duplicate_rejection")


def test_startup_closed_bucket():
    """Pre-fill :00-:04 + :05-:06, startup() → emit :00 bucket only."""
    tf = TestFeed()
    tf.pre_fill_buffer(make_bars(0, 6))  # 12:00-12:06
    tf.run_startup()
    assert tf.emit_count == 1, f"Expected 1 startup emit, got {tf.emit_count}"
    assert tf.emitted[0]['timestamp'] == datetime(2026, 2, 8, 12, 0)
    print("  ✅ test_startup_closed_bucket")


def test_startup_skip_in_progress():
    """Pre-fill :00-:02 only, startup() → 0 emits (incomplete bucket)."""
    tf = TestFeed()
    tf.pre_fill_buffer(make_bars(0, 2))
    tf.run_startup()
    assert tf.emit_count == 0, f"Expected 0 emits for incomplete, got {tf.emit_count}"
    print("  ✅ test_startup_skip_in_progress")


def test_mid_bucket_restart():
    """
    Feed :00-:02, then simulate restart (reset active buckets),
    then feed :03-:04, :05.
    Old bucket should fail invariant (only 2 bars from post-restart).
    No spurious emits.
    """
    tf = TestFeed()
    # Phase 1: feed :00, :01, :02
    tf.feed_bars(make_bars(0, 2))
    assert tf.emit_count == 0

    # Simulate restart: clear active buckets (as _on_open does)
    tf.feed._active_buckets.clear()

    # Phase 2: feed :03, :04 (same bucket, but fresh state)
    tf.feed_bar(make_bar(3))
    tf.feed_bar(make_bar(4))

    # Trigger boundary
    tf.feed_bar(make_bar(5))

    # Should fail invariant: only 2 bars (:03, :04) in bucket :00
    assert tf.emit_count == 0, f"Expected 0 emits after restart, got {tf.emit_count}"
    assert tf.has_log_containing("INVARIANT FAIL"), "Expected INVARIANT FAIL for partial bucket"
    print("  ✅ test_mid_bucket_restart")


def test_health_stale():
    """Emit at :00, then 1m bars at :11+ → CRITICAL logged."""
    tf = TestFeed()
    # Feed a full bucket + trigger
    tf.feed_bars(make_bars(0, 4))
    tf.feed_bar(make_bar(5))
    assert tf.emit_count == 1

    # Check health at :11 (11 minutes after bucket :00 timestamp)
    bar_at_11 = make_bar(11)
    tf.check_health(bar_at_11['timestamp'])
    assert tf.has_log_containing("5m STALE"), "Expected 5m STALE log"
    assert tf.has_log_level("CRITICAL"), "Expected CRITICAL level"
    print("  ✅ test_health_stale")


def test_health_normal():
    """Emit at :00, 1m bars at :05-:09 → no CRITICAL."""
    tf = TestFeed()
    tf.feed_bars(make_bars(0, 4))
    tf.feed_bar(make_bar(5))
    assert tf.emit_count == 1

    # Check health at :09 (9 minutes after bucket :00)
    bar_at_9 = make_bar(9)
    tf.check_health(bar_at_9['timestamp'])
    assert not tf.has_log_level("CRITICAL"), "Should not have CRITICAL within 10m"
    print("  ✅ test_health_normal")


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

def test_floor_to_5min():
    """floor_to_5min deterministically assigns bucket keys."""
    cases = [
        (datetime(2026, 2, 8, 7, 0), datetime(2026, 2, 8, 7, 0)),
        (datetime(2026, 2, 8, 7, 1), datetime(2026, 2, 8, 7, 0)),
        (datetime(2026, 2, 8, 7, 4), datetime(2026, 2, 8, 7, 0)),
        (datetime(2026, 2, 8, 7, 5), datetime(2026, 2, 8, 7, 5)),
        (datetime(2026, 2, 8, 7, 9), datetime(2026, 2, 8, 7, 5)),
        (datetime(2026, 2, 8, 7, 55), datetime(2026, 2, 8, 7, 55)),
        (datetime(2026, 2, 8, 7, 59), datetime(2026, 2, 8, 7, 55)),
    ]
    for ts, expected in cases:
        result = floor_to_5min(ts)
        assert result == expected, f"floor_to_5min({ts}) = {result}, expected {expected}"
    print("  ✅ test_floor_to_5min")


def test_validate_5min_bucket_valid():
    """Valid 5-bar bucket passes validation."""
    base = datetime(2026, 2, 8, 12, 0)
    bars = make_bars(0, 4)
    valid, error = validate_5min_bucket(bars, base)
    assert valid, f"Expected valid, got error: {error}"
    print("  ✅ test_validate_5min_bucket_valid")


def test_validate_5min_bucket_wrong_count():
    """3-bar bucket fails validation."""
    base = datetime(2026, 2, 8, 12, 0)
    bars = make_bars(0, 2)
    valid, error = validate_5min_bucket(bars, base)
    assert not valid
    assert "Expected 5 bars, got 3" in error
    print("  ✅ test_validate_5min_bucket_wrong_count")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("5m Aggregation Regression Tests")
    print("=" * 60)

    tests = [
        # Pure function tests
        test_floor_to_5min,
        test_validate_5min_bucket_valid,
        test_validate_5min_bucket_wrong_count,
        # Integration tests
        test_basic_emission,
        test_consecutive_emission,
        test_gap_anomaly,
        test_out_of_order,
        test_duplicate_rejection,
        test_startup_closed_bucket,
        test_startup_skip_in_progress,
        test_mid_bucket_restart,
        test_health_stale,
        test_health_normal,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test_fn.__name__}: UNEXPECTED ERROR: {e}")
            failed += 1

    print("\n" + "-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("-" * 60)

    if failed > 0:
        print("❌ TESTS FAILED — DO NOT DEPLOY")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED — Safe to deploy")
        sys.exit(0)


if __name__ == '__main__':
    main()
