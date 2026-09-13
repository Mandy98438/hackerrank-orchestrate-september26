"""
resolver.py -- step 2 of the pipeline: currency conversion, linked_event_id
chain resolution, and conflict precedence, before events reach forecast.py.

See docs/problem_analysis.md for the precedence order and docs/data_analysis.md
for the quirks (blank amounts, excluded event statuses) this module handles.
"""

from datetime import date

import forecast
from loader import RawEvent, ExchangeRate, Profile


EXCLUDED_STATUSES = {"pending_credit", "failed", "cancelled", "duplicate"}
EXCLUDED_KINDS = {"unrealized_investment"}


def find_rate(rates: list[ExchangeRate], on_date: date, from_currency: str, to_currency: str) -> float:
    """Exact match on date + currency pair, as the problem statement
    guarantees fixed dated rates rather than a continuous series. Raises if
    no matching rate exists -- better to fail loudly than silently assume 1:1.
    """
    if from_currency == to_currency:
        return 1.0
    for r in rates:
        if r.rate_date == on_date and r.from_currency == from_currency and r.to_currency == to_currency:
            return r.rate
        if r.rate_date == on_date and r.from_currency == to_currency and r.to_currency == from_currency:
            return 1.0 / r.rate
    raise ValueError(f"no exchange rate for {from_currency}->{to_currency} on {on_date}")


def convert_amount(amount: float, event_date: date, from_currency: str, to_currency: str, rates: list[ExchangeRate]) -> float:
    rate = find_rate(rates, event_date, from_currency, to_currency)
    return amount * rate


def resolve_linked_chain(events: list[RawEvent]) -> list[RawEvent]:
    """Collapse linked_event_id chains to the final state of each lifecycle.
    A record pointing at an earlier one via linked_event_id supersedes it,
    per precedence rule 1/2 (explicit amendment / newer record wins). We keep
    the newest record in each chain and drop the ones it supersedes.
    """
    superseded_ids = {e.linked_event_id for e in events if e.linked_event_id}
    return [e for e in events if e.event_id not in superseded_ids]


def resolve_conflicts(events: list[RawEvent]) -> list[RawEvent]:
    """Apply the stated precedence order when multiple records describe the
    same event_id after chain resolution: explicit cancellation/settlement/
    amendment > newer record > settled over estimate. Ties that still can't
    be resolved keep the financially safer (smaller income / larger expense)
    interpretation, applied at the caller level once amounts are known --
    this function only dedupes by event_id, keeping the last (assumed
    chronologically latest as loaded) record per id.
    """
    by_id: dict[str, RawEvent] = {}
    for e in events:
        prior = by_id.get(e.event_id)
        if prior is None:
            by_id[e.event_id] = e
            continue
        # explicit cancellation/settlement always wins over a lingering estimate
        if e.status in ("cancelled", "settled") and prior.status not in ("cancelled", "settled"):
            by_id[e.event_id] = e
        elif prior.status in ("cancelled", "settled") and e.status not in ("cancelled", "settled"):
            continue
        else:
            # fall back to "newer record wins" -- later in the file assumed newer
            by_id[e.event_id] = e
    return list(by_id.values())


def fill_blank_amounts(events: list[RawEvent], image_amounts: dict[str, float]) -> list[RawEvent]:
    """image_amounts maps event_id -> amount extracted from its linked image
    (see docs/data_analysis.md: blank amount -> look up related_event_id in
    images.csv -> extract from the PNG). Never treats a blank as zero.
    """
    out = []
    for e in events:
        if e.amount is None:
            resolved = image_amounts.get(e.event_id)
            if resolved is None:
                # cannot resolve -- exclude from forecast rather than assume zero
                continue
            out.append(RawEvent(
                e.event_id, e.user_id, e.date, resolved, e.currency, e.kind,
                e.recurring, e.flexible, e.interval_days, e.status, e.linked_event_id,
            ))
        else:
            out.append(e)
    return out


def to_forecast_events(
    raw_events: list[RawEvent],
    profile: Profile,
    rates: list[ExchangeRate],
    window_start: date,
    window_end: date,
) -> list[forecast.Event]:
    """Final step: filter excluded statuses/kinds, convert currency, and
    expand recurring events into concrete occurrences within the window.
    """
    usable = [
        e for e in raw_events
        if e.status not in EXCLUDED_STATUSES and e.kind not in EXCLUDED_KINDS and e.amount is not None
    ]

    out: list[forecast.Event] = []
    for e in usable:
        converted = convert_amount(e.amount, e.date, e.currency, profile.home_currency, rates)
        if e.recurring and e.interval_days:
            out.extend(forecast.expand_recurring(
                event_id=e.event_id,
                first_occurrence=e.date,
                amount=converted,
                interval_days=e.interval_days,
                window_start=window_start,
                window_end=window_end,
                flexible=e.flexible,
            ))
        elif window_start <= e.date <= window_end:
            out.append(forecast.Event(
                event_id=e.event_id, date=e.date, amount=converted,
                recurring=False, flexible=e.flexible,
            ))
    return out
