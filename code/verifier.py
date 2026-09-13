"""
verifier.py -- step 7 of the pipeline. Checks every stated invariant on a
formatted output row before it's written. See docs/problem_analysis.md for
the full invariant list this encodes. A row that fails here should never
reach output.csv unmodified -- main.py logs the violation and substitutes
the safe not_recommended fallback instead.
"""

from datetime import date


ALLOWED_STATUSES = {"affordable_now", "affordable_with_plan", "affordable_later", "not_affordable"}
ALLOWED_METHODS = {"full_payment", "partial_payment", "installments", "wait", "not_recommended"}


def _parse_plan(plan_str: str) -> list[tuple[str, float]]:
    if plan_str == "none":
        return []
    pairs = []
    for part in plan_str.split("|"):
        d, amt = part.split(":")
        pairs.append((d, float(amt)))
    return pairs


def _parse_changes(changes_str: str) -> list[tuple[str, str, float | None]]:
    if changes_str == "none":
        return []
    out = []
    for part in changes_str.split("|"):
        pieces = part.split(":")
        if pieces[0] == "stop":
            out.append(("stop", pieces[1], None))
        elif pieces[0] == "reduce_to":
            out.append(("reduce_to", pieces[1], float(pieces[2])))
    return out


def verify_row(
    row: dict,
    requested_amount: float,
    request_date: date,
    desired_completion_date: date,
    known_payment_option_ids: set[str],
    flexible_event_ids: set[str],
) -> list[str]:
    """Returns a list of violated invariant descriptions. Empty list means
    the row passes.
    """
    violations: list[str] = []

    amount_safe = float(row["amount_safe_to_pay"])
    if not (0 <= amount_safe <= requested_amount):
        violations.append("amount_safe_to_pay out of [0, requested_amount] bounds")

    status = row["affordability_status"]
    if status not in ALLOWED_STATUSES:
        violations.append(f"unsupported affordability_status: {status}")

    method = row["recommended_payment_method"]
    if method not in ALLOWED_METHODS:
        violations.append(f"unsupported recommended_payment_method: {method}")

    earliest = row["earliest_date_for_full_payment"]
    if status == "affordable_now" and earliest != request_date.isoformat():
        violations.append("affordable_now requires earliest_date_for_full_payment == request_date")

    plan = _parse_plan(row["payment_plan"])
    dates = [d for d, _ in plan]
    if dates != sorted(dates):
        violations.append("payment_plan dates are not in chronological order")

    if method == "partial_payment":
        if status != "affordable_with_plan":
            violations.append("partial_payment must have affordability_status affordable_with_plan")
        if len(plan) != 2:
            violations.append("partial_payment plan must have exactly 2 payments")
        else:
            total = round(plan[0][1] + plan[1][1], 2)
            if round(total, 2) != round(requested_amount, 2):
                violations.append("partial_payment payments do not sum to requested_amount")
            if not (0 < plan[0][1] < requested_amount):
                violations.append("partial_payment first payment must be > 0 and < requested_amount")

    if method == "installments":
        option_id = row.get("_payment_option_id")
        if option_id and option_id not in known_payment_option_ids:
            violations.append(f"installments plan does not match a known payment_option_id: {option_id}")

    changes = _parse_changes(row["spending_changes_needed"])
    stop_ids = {c[1] for c in changes if c[0] == "stop"}
    reduce_ids = {c[1] for c in changes if c[0] == "reduce_to"}
    if stop_ids & reduce_ids:
        violations.append("same event_id targeted by both stop and reduce_to")
    for _, event_id, _ in changes:
        if event_id not in flexible_event_ids:
            violations.append(f"spending change targets non-flexible event: {event_id}")
    if len(changes) > 3:
        violations.append("more than 3 spending changes")

    return violations


def verify_all(rows: list[dict], context_by_request_id: dict) -> dict[str, list[str]]:
    """context_by_request_id maps request_id -> dict with requested_amount,
    request_date, desired_completion_date, known_payment_option_ids,
    flexible_event_ids -- as prepared by main.py for that request.
    Returns only entries that have violations.
    """
    results = {}
    for row in rows:
        ctx = context_by_request_id[row["request_id"]]
        violations = verify_row(
            row,
            ctx["requested_amount"],
            ctx["request_date"],
            ctx["desired_completion_date"],
            ctx["known_payment_option_ids"],
            ctx["flexible_event_ids"],
        )
        if violations:
            results[row["request_id"]] = violations
    return results
