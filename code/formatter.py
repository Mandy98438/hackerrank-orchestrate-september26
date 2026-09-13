"""
formatter.py -- step 6 of the pipeline. Turns a ranked Candidate into the
exact output.csv strings. No decision-making here, purely presentation of
whatever ranker.py already decided.
"""

from datetime import date
from typing import Optional

from ranker import Candidate, SpendingChange


def format_payment_plan(plan: tuple[tuple[date, float], ...]) -> str:
    if not plan:
        return "none"
    parts = [f"{d.isoformat()}:{_fmt_amount(amt)}" for d, amt in plan]
    return "|".join(parts)


def format_spending_changes(changes: tuple[SpendingChange, ...]) -> str:
    if not changes:
        return "none"
    parts = []
    for c in changes:
        if c.kind == "stop":
            parts.append(f"stop:{c.event_id}")
        elif c.kind == "reduce_to":
            parts.append(f"reduce_to:{c.event_id}:{_fmt_amount(c.new_amount)}")
    return "|".join(parts)


def _fmt_amount(amount: float) -> str:
    # Avoid trailing ".0" noise while keeping real cents.
    if amount == int(amount):
        return str(int(amount))
    return f"{amount:.2f}"


def determine_affordability_status(
    candidate: Optional[Candidate],
    request_date: date,
    desired_completion_date: date,
) -> str:
    if candidate is None:
        return "not_affordable"
    if candidate.method == "full_payment" and candidate.start_date == request_date:
        return "affordable_now"
    if candidate.method in ("partial_payment", "installments"):
        last_date = max(d for d, _ in candidate.plan)
        return "affordable_with_plan" if last_date <= desired_completion_date else "not_affordable"
    if candidate.method == "full_payment" and candidate.spending_changes:
        return "affordable_with_plan"
    if candidate.method == "wait":
        return "affordable_later"
    return "not_affordable"


def build_decision_explanation(
    candidate: Optional[Candidate],
    requested_amount: float,
    amount_safe_to_pay: float,
) -> str:
    if candidate is None:
        return (
            f"No safe plan found within the 90-day forecast window for the full "
            f"{_fmt_amount(requested_amount)} request; paying it would break the "
            f"minimum balance requirement at some point in the forecast."
        )
    if candidate.method == "full_payment" and not candidate.spending_changes:
        return f"Full {_fmt_amount(requested_amount)} payment stays safe today against the 90-day forecast."
    if candidate.method == "full_payment" and candidate.spending_changes:
        changed = ", ".join(
            f"{c.kind} {c.event_id}" + (f" to {_fmt_amount(c.new_amount)}" if c.new_amount else "")
            for c in candidate.spending_changes
        )
        return f"Full payment becomes safe today after {changed}."
    if candidate.method == "partial_payment":
        today_amt, later_amt = candidate.plan[0][1], candidate.plan[1][1]
        later_date = candidate.plan[1][0]
        return (
            f"Only {_fmt_amount(today_amt)} of {_fmt_amount(requested_amount)} is safe today; "
            f"remaining {_fmt_amount(later_amt)} is safe on {later_date.isoformat()}."
        )
    if candidate.method == "installments":
        return f"Installment option {candidate.payment_option_id} stays safe across the full 90-day forecast."
    if candidate.method == "wait":
        return f"Full payment isn't safe today but becomes safe on {candidate.start_date.isoformat()}."
    return "No eligible payment method is safe within the forecast window."


def build_output_row(
    request_id: str,
    candidate: Optional[Candidate],
    requested_amount: float,
    amount_safe_to_pay: float,
    earliest_date_for_full_payment: Optional[date],
    request_date: date,
    desired_completion_date: date,
) -> dict:
    status = determine_affordability_status(candidate, request_date, desired_completion_date)
    method = candidate.method if candidate else "not_recommended"
    plan_str = format_payment_plan(candidate.plan) if candidate else "none"
    changes_str = format_spending_changes(candidate.spending_changes) if candidate else "none"
    explanation = build_decision_explanation(candidate, requested_amount, amount_safe_to_pay)

    return {
        "request_id": request_id,
        "amount_safe_to_pay": _fmt_amount(amount_safe_to_pay),
        "affordability_status": status,
        "recommended_payment_method": method,
        "payment_plan": plan_str,
        "earliest_date_for_full_payment": (
            earliest_date_for_full_payment.isoformat() if (status == "affordable_now" or earliest_date_for_full_payment) else ""
        ),
        "spending_changes_needed": changes_str,
        "decision_explanation": explanation,
    }
