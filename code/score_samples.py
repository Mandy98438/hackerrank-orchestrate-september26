"""
evaluation/score_samples.py -- runs the pipeline against sample_requests.csv
(which already has completed output columns) and reports field-level
accuracy, per docs/eval_plan.md. This is the loop to run before ever
touching the real requests.csv.

Usage:
    python evaluation/score_samples.py --dataset dataset/
"""

import sys
import csv
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import main as pipeline  # noqa: E402
import verifier  # noqa: E402
import loader  # noqa: E402
import resolver  # noqa: E402


FIELDS_EXACT = [
    "affordability_status", "recommended_payment_method",
    "payment_plan", "earliest_date_for_full_payment", "spending_changes_needed",
]
FIELD_NUMERIC = "amount_safe_to_pay"


def load_expected(path: Path) -> dict[str, dict]:
    expected = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected[row["request_id"]] = row
    return expected


def score(dataset_dir: Path) -> None:
    requests = loader.load_requests(dataset_dir / "sample_requests.csv")
    profiles = loader.load_profiles(dataset_dir / "financial_profiles.csv")
    events_by_user = loader.load_events(dataset_dir / "financial_events.csv")
    rates = loader.load_exchange_rates(dataset_dir / "exchange_rates.csv")
    options_by_request = loader.load_payment_options(dataset_dir / "request_payment_options.csv")
    messages = loader.load_messages(dataset_dir / "messages.csv")
    expected = load_expected(dataset_dir / "sample_requests.csv")

    rows, contexts = [], {}
    for req in requests:
        try:
            profile = profiles[req.user_id]
            raw_events = events_by_user.get(req.user_id, [])
            raw_events = resolver.resolve_linked_chain(raw_events)
            raw_events = resolver.resolve_conflicts(raw_events)
            options = options_by_request.get(req.request_id, [])
            msg_texts = [
                m.text for m in messages
                if m.request_id == req.request_id or m.user_id == req.user_id
            ]
            row, ctx = pipeline.process_request(req, profile, raw_events, rates, options, msg_texts)
        except Exception as exc:
            print(f"  ! {req.request_id} raised {exc!r}, scoring as a full miss")
            row = pipeline.build_fallback_row(req.request_id)
            ctx = {
                "requested_amount": req.requested_amount,
                "request_date": req.request_date,
                "desired_completion_date": req.desired_completion_date,
                "known_payment_option_ids": set(),
                "flexible_event_ids": set(),
            }
        rows.append(row)
        contexts[req.request_id] = ctx

    invariant_failures = verifier.verify_all(rows, contexts)

    field_hits = {f: 0 for f in FIELDS_EXACT}
    numeric_abs_error = []
    total = len(rows)

    for row in rows:
        exp = expected.get(row["request_id"])
        if exp is None:
            continue
        for field in FIELDS_EXACT:
            if row[field] == exp[field]:
                field_hits[field] += 1
        try:
            numeric_abs_error.append(abs(float(row[FIELD_NUMERIC]) - float(exp[FIELD_NUMERIC])))
        except (ValueError, KeyError):
            pass

    print(f"\nScored {total} sample requests")
    print(f"Invariant violations: {len(invariant_failures)} / {total}")
    for request_id, problems in invariant_failures.items():
        print(f"  ! {request_id}: {problems}")

    print("\nField-level exact-match accuracy:")
    for field in FIELDS_EXACT:
        pct = 100 * field_hits[field] / total if total else 0
        print(f"  {field}: {pct:.1f}% ({field_hits[field]}/{total})")

    if numeric_abs_error:
        avg_err = sum(numeric_abs_error) / len(numeric_abs_error)
        print(f"\n{FIELD_NUMERIC} mean absolute error: {avg_err:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    score(args.dataset)
