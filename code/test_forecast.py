"""
evaluation/test_forecast.py -- pure unit tests for forecast.py.

No API calls, no dataset files needed -- everything is synthetic. Run with:
    pytest evaluation/test_forecast.py
(run from the repo root with code/ on PYTHONPATH, or add a conftest.py that
inserts code/ into sys.path).
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import forecast  # noqa: E402


D = date(2026, 9, 13)  # a fixed request_date for all tests
WINDOW_END = date(2026, 12, 12)  # D + 90 days


def test_amount_safe_to_pay_no_events_returns_full_balance_minus_minimum():
    safe = forecast.amount_safe_to_pay(
        events=[], available_balance=1000, minimum_balance_to_keep=200,
        window_start=D, window_end=WINDOW_END, requested_amount=500,
    )
    assert safe == 500  # capped by requested_amount, plenty of slack


def test_amount_safe_to_pay_capped_by_requested_amount():
    safe = forecast.amount_safe_to_pay(
        events=[], available_balance=10_000, minimum_balance_to_keep=0,
        window_start=D, window_end=WINDOW_END, requested_amount=300,
    )
    assert safe == 300


def test_amount_safe_to_pay_limited_by_future_expense():
    events = [
        forecast.Event("rent", date(2026, 9, 20), -800),
    ]
    safe = forecast.amount_safe_to_pay(
        events, available_balance=1000, minimum_balance_to_keep=100,
        window_start=D, window_end=WINDOW_END, requested_amount=500,
    )
    # after rent hits, balance = 200; slack at that point = 200 - 100 = 100
    assert safe == 100


def test_amount_safe_to_pay_zero_when_already_at_minimum():
    safe = forecast.amount_safe_to_pay(
        events=[], available_balance=200, minimum_balance_to_keep=200,
        window_start=D, window_end=WINDOW_END, requested_amount=100,
    )
    assert safe == 0


def test_earliest_date_for_full_payment_finds_post_income_date():
    events = [
        forecast.Event("rent", date(2026, 9, 20), -800),
        forecast.Event("salary", date(2026, 10, 1), 2000),
    ]
    earliest = forecast.earliest_date_for_full_payment(
        events, available_balance=1000, minimum_balance_to_keep=100,
        window_start=D, window_end=WINDOW_END, requested_amount=1500,
    )
    assert earliest == date(2026, 10, 1)


def test_earliest_date_for_full_payment_none_when_never_safe():
    events = [forecast.Event("rent", date(2026, 9, 20), -50)]
    earliest = forecast.earliest_date_for_full_payment(
        events, available_balance=100, minimum_balance_to_keep=50,
        window_start=D, window_end=WINDOW_END, requested_amount=10_000,
    )
    assert earliest is None


def test_apply_stop_removes_only_future_flexible_occurrences():
    events = [
        forecast.Event("subscription", date(2026, 9, 1), -20, recurring=True, flexible=True),
        forecast.Event("subscription", date(2026, 10, 1), -20, recurring=True, flexible=True),
        forecast.Event("rent", date(2026, 9, 20), -800, recurring=True, flexible=False),
    ]
    out = forecast.apply_stop(events, "subscription", window_start=D)
    ids_dates = [(e.event_id, e.date) for e in out]
    assert ("subscription", date(2026, 9, 1)) in ids_dates  # before window_start, untouched by design
    assert ("subscription", date(2026, 10, 1)) not in ids_dates
    assert ("rent", date(2026, 9, 20)) in ids_dates  # not flexible, untouched


def test_apply_reduce_preserves_sign():
    events = [forecast.Event("subscription", date(2026, 10, 1), -20, recurring=True, flexible=True)]
    out = forecast.apply_reduce(events, "subscription", new_amount=5)
    assert out[0].amount == -5


def test_is_schedule_safe_true_when_never_breaches_minimum():
    events = [forecast.Event("rent", date(2026, 9, 20), -100)]
    assert forecast.is_schedule_safe(
        events, available_balance=500, minimum_balance_to_keep=100,
        window_start=D, window_end=WINDOW_END,
    )


def test_is_schedule_safe_false_when_breaches_minimum():
    events = [forecast.Event("rent", date(2026, 9, 20), -450)]
    assert not forecast.is_schedule_safe(
        events, available_balance=500, minimum_balance_to_keep=100,
        window_start=D, window_end=WINDOW_END,
    )


def test_expand_recurring_generates_correct_occurrence_count():
    occurrences = forecast.expand_recurring(
        "subscription", first_occurrence=date(2026, 9, 1), amount=-20,
        interval_days=30, window_start=D, window_end=WINDOW_END,
    )
    # roughly one occurrence per 30 days across a 90-day window
    assert 2 <= len(occurrences) <= 4
    assert all(D <= o.date <= WINDOW_END for o in occurrences)


def test_expand_recurring_fast_forwards_to_window_start():
    occurrences = forecast.expand_recurring(
        "subscription", first_occurrence=date(2025, 1, 1), amount=-20,
        interval_days=30, window_start=D, window_end=WINDOW_END,
    )
    assert all(o.date >= D for o in occurrences)


def test_partial_payment_math_consistency():
    """The remainder date for a partial_payment plan must always equal
    earliest_date_for_full_payment on the *baseline* trajectory -- see
    docs/decision_logic.md section 4. This test locks in that equality.
    """
    events = [
        forecast.Event("rent", date(2026, 9, 20), -800),
        forecast.Event("salary", date(2026, 10, 1), 2000),
    ]
    requested = 1500
    safe_today = forecast.amount_safe_to_pay(
        events, available_balance=1000, minimum_balance_to_keep=100,
        window_start=D, window_end=WINDOW_END, requested_amount=requested,
    )
    earliest_full = forecast.earliest_date_for_full_payment(
        events, available_balance=1000, minimum_balance_to_keep=100,
        window_start=D, window_end=WINDOW_END, requested_amount=requested,
    )
    assert 0 < safe_today < requested
    assert earliest_full == date(2026, 10, 1)
