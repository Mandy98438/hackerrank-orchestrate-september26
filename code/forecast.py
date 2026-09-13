"""
forecast.py -- deterministic 90-day balance forecasting.

Pure functions only: no network calls, no file I/O, no LLM calls.
Everything here must be fully unit-testable against synthetic event lists.

Core idea (see docs/decision_logic.md):
Balance only changes at event dates, so the minimum balance across a span
is always found immediately after some outflow event. We build a sorted
event-driven trajectory instead of simulating day by day.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


@dataclass(frozen=True)
class Event:
    """One cash-flow event already resolved by loader.py/resolver.py and
    merged with extractor.py's confirmed facts.

    amount is signed: positive for income, negative for expense, already
    converted to the user's home_currency.
    """
    event_id: str
    date: date
    amount: float
    recurring: bool = False
    flexible: bool = False  # only recurring expenses can ever be flexible


def events_in_window(events: list[Event], window_start: date, window_end: date) -> list[Event]:
    """Filter + sort events falling within [window_start, window_end] inclusive."""
    in_window = [e for e in events if window_start <= e.date <= window_end]
    return sorted(in_window, key=lambda e: e.date)


def build_trajectory(
    events: list[Event],
    available_balance: float,
    window_start: date,
    window_end: date,
) -> list[tuple[date, float]]:
    """Walk the sorted event list forward, returning (date, balance_after_event)
    for every event date in the window. The first entry represents
    available_balance at window_start, before any event fires that day.
    """
    trajectory: list[tuple[date, float]] = [(window_start, available_balance)]
    balance = available_balance
    for e in events_in_window(events, window_start, window_end):
        balance += e.amount
        trajectory.append((e.date, balance))
    return trajectory


def slack_series(
    trajectory: list[tuple[date, float]],
    minimum_balance_to_keep: float,
) -> list[tuple[date, float]]:
    """slack(d) = balance(d) - minimum_balance_to_keep, for every point."""
    return [(d, bal - minimum_balance_to_keep) for d, bal in trajectory]


def amount_safe_to_pay(
    events: list[Event],
    available_balance: float,
    minimum_balance_to_keep: float,
    window_start: date,
    window_end: date,
    requested_amount: float,
) -> float:
    """Max amount payable on window_start without breaking the 90-day safety
    check, capped at requested_amount. See docs/decision_logic.md section 2:
    the tightest day's slack across the whole window is the cap, since a
    same-day withdrawal shifts every later day's balance down by the same
    fixed amount.
    """
    trajectory = build_trajectory(events, available_balance, window_start, window_end)
    slacks = [s for _, s in slack_series(trajectory, minimum_balance_to_keep)]
    tightest = min(slacks) if slacks else 0.0
    return max(0.0, min(tightest, requested_amount))


def earliest_date_for_full_payment(
    events: list[Event],
    available_balance: float,
    minimum_balance_to_keep: float,
    window_start: date,
    window_end: date,
    requested_amount: float,
) -> Optional[date]:
    """First date t in [window_start, window_end] where paying requested_amount
    in full on t keeps every subsequent day safe too. Relies on
    min_suffix_slack(t) being monotonic non-decreasing in t (see
    docs/decision_logic.md section 3), so one backward scan is enough.
    """
    trajectory = build_trajectory(events, available_balance, window_start, window_end)
    slacks = slack_series(trajectory, minimum_balance_to_keep)
    if not slacks:
        return None

    n = len(slacks)
    min_suffix = [0.0] * n
    min_suffix[-1] = slacks[-1][1]
    for i in range(n - 2, -1, -1):
        min_suffix[i] = min(slacks[i][1], min_suffix[i + 1])

    for i, (d, _) in enumerate(slacks):
        if min_suffix[i] >= requested_amount:
            return d
    return None


def apply_stop(events: list[Event], event_id: str, window_start: date) -> list[Event]:
    """Remove future occurrences (on/after window_start) of one flexible
    recurring expense. Everything else passes through unchanged.
    """
    out = []
    for e in events:
        if e.event_id == event_id and e.recurring and e.flexible and e.date >= window_start:
            continue
        out.append(e)
    return out


def apply_reduce(events: list[Event], event_id: str, new_amount: float) -> list[Event]:
    """Cap a flexible recurring expense's magnitude to new_amount, preserving
    its sign. Only touches matching flexible recurring events.
    """
    out = []
    for e in events:
        if e.event_id == event_id and e.recurring and e.flexible:
            sign = -1 if e.amount < 0 else 1
            out.append(Event(e.event_id, e.date, sign * abs(new_amount), e.recurring, e.flexible))
        else:
            out.append(e)
    return out


def apply_installment_schedule(
    events: list[Event],
    option_payments: list[tuple[date, float]],
    option_id: str,
) -> list[Event]:
    """Layer a payment option's fixed schedule onto the event list as extra
    outflow events, tagged with the option_id for traceability.
    """
    extra = [
        Event(event_id=f"installment:{option_id}", date=d, amount=-abs(amt))
        for d, amt in option_payments
    ]
    return events + extra


def is_schedule_safe(
    events: list[Event],
    available_balance: float,
    minimum_balance_to_keep: float,
    window_start: date,
    window_end: date,
) -> bool:
    """True if the given event list never drops balance below
    minimum_balance_to_keep anywhere in the window.
    """
    trajectory = build_trajectory(events, available_balance, window_start, window_end)
    return all(bal >= minimum_balance_to_keep for _, bal in trajectory)


def expand_recurring(
    event_id: str,
    first_occurrence: date,
    amount: float,
    interval_days: int,
    window_start: date,
    window_end: date,
    flexible: bool = False,
) -> list[Event]:
    """Expand a recurring event definition into individual Event occurrences
    within the window. Anchors on first_occurrence and steps by
    interval_days; does not assume the recurrence starts at window_start.
    """
    occurrences: list[Event] = []
    d = first_occurrence
    if d < window_start and interval_days > 0:
        steps = (window_start - d).days // interval_days
        d = d + timedelta(days=steps * interval_days)
        while d < window_start:
            d += timedelta(days=interval_days)
    while d <= window_end:
        occurrences.append(Event(event_id, d, amount, recurring=True, flexible=flexible))
        d += timedelta(days=interval_days)
    return occurrences
