"""
ranker.py -- candidate plan generation and the 6-step tie-break ranking.

Pure functions, no model calls. Reads only forecast.py's outputs plus
request/profile facts already loaded by loader.py. See docs/decision_logic.md
sections 7-8 for the rules encoded here.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import forecast


@dataclass(frozen=True)
class SpendingChange:
    kind: str  # "stop" | "reduce_to"
    event_id: str
    new_amount: Optional[float] = None  # only set for reduce_to


@dataclass(frozen=True)
class PaymentOption:
    option_id: str
    payments: tuple[tuple[date, float], ...]


@dataclass(frozen=True)
class Candidate:
    method: str  # full_payment | partial_payment | installments | wait | not_recommended
    plan: tuple[tuple[date, float], ...]  # chronological (date, amount) pairs
    spending_changes: tuple[SpendingChange, ...]
    total_paid: float
    start_date: Optional[date]
    payment_option_id: Optional[str] = None  # only set for installments


def _eligible(user_methods: set[str], method: str) -> bool:
    return method in user_methods


def generate_full_payment_candidate(
    events, available_balance, minimum_balance_to_keep,
    request_date, window_end, requested_amount, user_methods,
) -> Optional[Candidate]:
    if not _eligible(user_methods, "full_payment"):
        return None
    safe_today = forecast.amount_safe_to_pay(
        events, available_balance, minimum_balance_to_keep,
        request_date, window_end, requested_amount,
    )
    if safe_today < requested_amount:
        return None
    return Candidate(
        method="full_payment",
        plan=((request_date, requested_amount),),
        spending_changes=(),
        total_paid=requested_amount,
        start_date=request_date,
    )


def generate_partial_payment_candidate(
    events, available_balance, minimum_balance_to_keep,
    request_date, window_end, requested_amount,
    allows_partial_payment, user_methods, desired_completion_date,
) -> Optional[Candidate]:
    if not allows_partial_payment or not _eligible(user_methods, "partial_payment"):
        return None
    safe_today = forecast.amount_safe_to_pay(
        events, available_balance, minimum_balance_to_keep,
        request_date, window_end, requested_amount,
    )
    if not (0 < safe_today < requested_amount):
        return None
    # Reuses earliest_date_for_full_payment for the remainder date -- see
    # docs/decision_logic.md section 4 for why this is mathematically the
    # correct date, not an approximation.
    remainder_date = forecast.earliest_date_for_full_payment(
        events, available_balance, minimum_balance_to_keep,
        request_date, window_end, requested_amount,
    )
    if remainder_date is None or remainder_date > desired_completion_date:
        return None
    remainder = requested_amount - safe_today
    return Candidate(
        method="partial_payment",
        plan=((request_date, safe_today), (remainder_date, remainder)),
        spending_changes=(),
        total_paid=requested_amount,
        start_date=request_date,
    )


def generate_installment_candidates(
    events, available_balance, minimum_balance_to_keep,
    request_date, window_end, user_methods, payment_options: list[PaymentOption],
) -> list[Candidate]:
    if not _eligible(user_methods, "installments"):
        return []
    candidates = []
    for option in payment_options:
        layered = forecast.apply_installment_schedule(
            events, list(option.payments), option.option_id
        )
        if forecast.is_schedule_safe(
            layered, available_balance, minimum_balance_to_keep, request_date, window_end
        ):
            total = sum(amt for _, amt in option.payments)
            candidates.append(Candidate(
                method="installments",
                plan=option.payments,
                spending_changes=(),
                total_paid=total,
                start_date=option.payments[0][0] if option.payments else request_date,
                payment_option_id=option.option_id,
            ))
    return candidates


def generate_wait_candidate(
    events, available_balance, minimum_balance_to_keep,
    request_date, window_end, requested_amount, user_methods,
) -> Optional[Candidate]:
    if not _eligible(user_methods, "full_payment"):
        return None
    later_date = forecast.earliest_date_for_full_payment(
        events, available_balance, minimum_balance_to_keep,
        request_date, window_end, requested_amount,
    )
    if later_date is None or later_date == request_date:
        return None  # same-day case belongs to full_payment, not wait
    return Candidate(
        method="wait",
        plan=((later_date, requested_amount),),
        spending_changes=(),
        total_paid=requested_amount,
        start_date=later_date,
    )


def generate_spending_change_candidate(
    events, available_balance, minimum_balance_to_keep,
    request_date, window_end, requested_amount, user_methods,
    change: SpendingChange,
) -> Optional[Candidate]:
    """Try one optional spending change and see if it makes full_payment
    (today) safe. Only called when the unmodified baseline already failed
    every other candidate -- see docs/decision_logic.md section 5.
    """
    if not _eligible(user_methods, "full_payment"):
        return None
    if change.kind == "stop":
        adjusted = forecast.apply_stop(events, change.event_id, request_date)
    elif change.kind == "reduce_to":
        adjusted = forecast.apply_reduce(events, change.event_id, change.new_amount)
    else:
        return None

    safe_today = forecast.amount_safe_to_pay(
        adjusted, available_balance, minimum_balance_to_keep,
        request_date, window_end, requested_amount,
    )
    if safe_today < requested_amount:
        return None
    return Candidate(
        method="full_payment",
        plan=((request_date, requested_amount),),
        spending_changes=(change,),
        total_paid=requested_amount,
        start_date=request_date,
    )


def rank(candidates: list[Candidate], desired_completion_date: date) -> Optional[Candidate]:
    """Apply the 6-step tie-break in order. Returns None if candidates is
    empty -- caller emits the not_recommended fallback in that case.
    """
    if not candidates:
        return None

    def completes_by_deadline(c: Candidate) -> bool:
        last_date = max((d for d, _ in c.plan), default=c.start_date)
        return last_date is not None and last_date <= desired_completion_date

    def sort_key(c: Candidate):
        return (
            0 if completes_by_deadline(c) else 1,  # lower is better
            len(c.spending_changes),                # fewer changes is better
            c.total_paid,                           # lower total paid is better
            c.start_date or date.max,               # earlier start is better
            len(c.plan),                            # fewer payments is better
            c.payment_option_id or "",              # lowest option id, last tiebreak
        )

    return sorted(candidates, key=sort_key)[0]
