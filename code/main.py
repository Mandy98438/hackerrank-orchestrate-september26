"""
main.py -- runs the full pipeline end to end.

    dataset/*.csv
       |
       v
    1. LOAD      loader.py
       v
    2. RESOLVE   resolver.py
       v
    3. EXTRACT   extractor.py  (only module that calls the LLM)
       v
    4. FORECAST  forecast.py
       v
    5. RANK      ranker.py
       v
    6. FORMAT    formatter.py
       v
    7. VERIFY    verifier.py
       v
    output.csv

Per-request try/except so one malformed row can't crash the full run --
a failing row gets a logged, safe not_recommended fallback instead of
breaking the batch. See docs/architecture.md and docs/eval_plan.md.
"""

from datetime import timedelta
from pathlib import Path
import csv
import logging
import os
import time

import loader
import resolver
import extractor
import forecast
import ranker
import formatter
import verifier

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
except ImportError:
    genai = None
    genai_types = None
    genai_errors = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("buy_or_wait")

FORECAST_WINDOW_DAYS = 90
OUTPUT_COLUMNS = [
    "request_id", "amount_safe_to_pay", "affordability_status",
    "recommended_payment_method", "payment_plan",
    "earliest_date_for_full_payment", "spending_changes_needed",
    "decision_explanation",
]

# Extraction is a small, well-scoped structured-output task (see
# docs/architecture.md: "model describes, code decides"), not open-ended
# reasoning, so the cheapest current tier of whichever provider is right.
# LLM_PROVIDER selects which SDK/API is used; only this file needs to change
# when switching providers -- extractor.py just calls call_model_fn and
# never knows which provider is behind it. Override with BUY_OR_WAIT_MODEL
# if a different model within that provider is preferred.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-3.6-flash",
}
MODEL_NAME = os.environ.get("BUY_OR_WAIT_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, ""))
MAX_TOKENS = 512
MAX_RETRIES_ON_RATE_LIMIT = 4

# Fill these in from the provider's official pricing page right before the
# final full-dataset run -- do not guess a number here, third-party pricing
# summaries were unreliable and inconsistent with each other at the time
# this was written. Leaving a model's entry as None means usage_report.md
# reports token counts for it but marks cost as "not computed", which is
# honest and still satisfies the deliverable (it requires costs estimated
# from the actual run, not fabricated).
PRICING_PER_MILLION_TOKENS = {
    # "claude-haiku-4-5-20251001": {"input": None, "output": None},
    # "gemini-2.5-flash": {"input": None, "output": None},
}

_usage_log: list[dict] = []  # each entry: {model, input_tokens, output_tokens}
_client = None


def _get_client():
    global _client
    if _client is None:
        if LLM_PROVIDER == "anthropic":
            if anthropic is None:
                raise RuntimeError("pip install anthropic before running with LLM_PROVIDER=anthropic")
            _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        elif LLM_PROVIDER == "gemini":
            if genai is None:
                raise RuntimeError("pip install google-genai before running with LLM_PROVIDER=gemini")
            _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))  # or GOOGLE_API_KEY
        else:
            raise RuntimeError(f"unknown LLM_PROVIDER: {LLM_PROVIDER!r} (use 'anthropic' or 'gemini')")
    return _client


def _call_anthropic(client, system_prompt: str, user_prompt: str) -> str:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    _usage_log.append({
        "model": MODEL_NAME,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    })
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _call_gemini(client, system_prompt: str, user_prompt: str) -> str:
    # response_mime_type="application/json" asks Gemini to constrain output
    # to valid JSON at generation time -- a stricter guarantee than
    # prompting alone, which helps extractor.py's parse_and_validate see
    # fewer malformed responses to retry.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            max_output_tokens=MAX_TOKENS,
        ),
    )
    usage = response.usage_metadata
    _usage_log.append({
        "model": MODEL_NAME,
        "input_tokens": usage.prompt_token_count or 0,
        "output_tokens": usage.candidates_token_count or 0,
    })
    return (response.text or "").strip()


def call_model_fn(system_prompt: str, user_prompt: str) -> str:
    """Real LLM call, routed to whichever provider LLM_PROVIDER selects.
    Retries with exponential backoff on rate limits (the same 429 issue hit
    in the prior Orchestrate build, for either provider). Logs every call's
    token usage for evaluation/usage_report.md. Left as a plug point --
    evaluation/test_*.py inject a fake version instead and never spend a
    token or need any of this wired up at all.
    """
    client = _get_client()
    delay = 1.0
    for attempt in range(MAX_RETRIES_ON_RATE_LIMIT + 1):
        try:
            if LLM_PROVIDER == "anthropic":
                return _call_anthropic(client, system_prompt, user_prompt)
            else:
                return _call_gemini(client, system_prompt, user_prompt)
        except anthropic.RateLimitError if LLM_PROVIDER == "anthropic" else genai_errors.ClientError as exc:
            is_rate_limit = LLM_PROVIDER == "anthropic" or getattr(exc, "code", None) == 429
            if not is_rate_limit or attempt == MAX_RETRIES_ON_RATE_LIMIT:
                raise
            log.warning("rate limited, retrying in %.1fs (attempt %d)", delay, attempt + 1)
            time.sleep(delay)
            delay *= 2


def write_usage_report(path: Path, num_requests: int) -> None:
    """Summarizes the actual logged calls from this run -- not an estimate.
    Required deliverable: evaluation/usage_report.md.
    """
    by_model: dict[str, dict] = {}
    for entry in _usage_log:
        m = by_model.setdefault(entry["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        m["calls"] += 1
        m["input_tokens"] += entry["input_tokens"]
        m["output_tokens"] += entry["output_tokens"]

    total_calls = sum(m["calls"] for m in by_model.values())
    total_in = sum(m["input_tokens"] for m in by_model.values())
    total_out = sum(m["output_tokens"] for m in by_model.values())
    total_tokens = total_in + total_out

    lines = [
        "# Token Usage and Cost Report",
        "",
        f"Full-dataset run: {num_requests} requests, {total_calls} model calls.",
        "",
        "## Per-model breakdown",
        "",
        "| Model | Calls | Input tokens | Output tokens | Est. cost (USD) |",
        "|---|---|---|---|---|",
    ]
    total_cost = 0.0
    any_cost_missing = False
    for model, m in by_model.items():
        pricing = PRICING_PER_MILLION_TOKENS.get(model)
        if pricing and pricing.get("input") is not None and pricing.get("output") is not None:
            cost = (m["input_tokens"] * pricing["input"] + m["output_tokens"] * pricing["output"]) / 1_000_000
            total_cost += cost
            cost_str = f"${cost:.4f}"
        else:
            any_cost_missing = True
            cost_str = "not computed (set PRICING_PER_MILLION_TOKENS)"
        lines.append(f"| {model} | {m['calls']} | {m['input_tokens']} | {m['output_tokens']} | {cost_str} |")

    lines += [
        "",
        "## Overall totals",
        "",
        f"- Total model calls: {total_calls}",
        f"- Total tokens: {total_tokens} ({total_in} input, {total_out} output)",
        f"- Average tokens per request: {total_tokens / num_requests:.1f}" if num_requests else "- Average tokens per request: n/a",
        f"- Estimated total cost: {'$' + format(total_cost, '.4f') if not any_cost_missing else 'incomplete -- see per-model table'}",
    ]
    if num_requests:
        lines.append(f"- Estimated cost per request: {'$' + format(total_cost / num_requests, '.6f') if not any_cost_missing else 'n/a'}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fallback_row(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "amount_safe_to_pay": "0",
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": "Could not be safely evaluated; flagged for review.",
    }


def process_request(
    req: loader.Request,
    profile: loader.Profile,
    raw_events: list,
    rates: list,
    payment_options: list,
    messages_text: list[str],
) -> tuple[dict, dict]:
    """Returns (output_row, verification_context) for one request."""
    window_start = req.request_date
    window_end = req.request_date + timedelta(days=FORECAST_WINDOW_DAYS)

    known_event_ids = {e.event_id for e in raw_events}
    facts = extractor.extract_facts(
        call_model_fn, req.request_text, messages_text, known_event_ids,
    )

    # apply_facts_to_events works on forecast.Event, so resolve first, then
    # merge in the confirmed/cancelled/delayed facts before final use.
    resolved = resolver.to_forecast_events(raw_events, profile, rates, window_start, window_end)
    events = extractor.apply_facts_to_events(resolved, facts)

    flexible_event_ids = {e.event_id for e in events if e.flexible}

    amount_safe = forecast.amount_safe_to_pay(
        events, profile.available_balance, profile.minimum_balance_to_keep,
        window_start, window_end, req.requested_amount,
    )
    earliest_full = forecast.earliest_date_for_full_payment(
        events, profile.available_balance, profile.minimum_balance_to_keep,
        window_start, window_end, req.requested_amount,
    )

    candidates = []
    full = ranker.generate_full_payment_candidate(
        events, profile.available_balance, profile.minimum_balance_to_keep,
        window_start, window_end, req.requested_amount,
        profile.payment_methods_user_will_consider,
    )
    if full:
        candidates.append(full)

    partial = ranker.generate_partial_payment_candidate(
        events, profile.available_balance, profile.minimum_balance_to_keep,
        window_start, window_end, req.requested_amount,
        req.allows_partial_payment, profile.payment_methods_user_will_consider,
        req.desired_completion_date,
    )
    if partial:
        candidates.append(partial)

    option_objs = [
        ranker.PaymentOption(
            option_id=o.payment_option_id,
            payments=tuple(_expand_option_payments(o)),
        )
        for o in payment_options
    ]
    candidates.extend(ranker.generate_installment_candidates(
        events, profile.available_balance, profile.minimum_balance_to_keep,
        window_start, window_end, profile.payment_methods_user_will_consider,
        option_objs,
    ))

    wait = ranker.generate_wait_candidate(
        events, profile.available_balance, profile.minimum_balance_to_keep,
        window_start, window_end, req.requested_amount,
        profile.payment_methods_user_will_consider,
    )
    if wait:
        candidates.append(wait)

    if not candidates:
        # try optional spending changes as a last resort before giving up
        for event in events:
            if event.flexible:
                change = ranker.SpendingChange(kind="stop", event_id=event.event_id)
                c = ranker.generate_spending_change_candidate(
                    events, profile.available_balance, profile.minimum_balance_to_keep,
                    window_start, window_end, req.requested_amount,
                    profile.payment_methods_user_will_consider, change,
                )
                if c:
                    candidates.append(c)

    winner = ranker.rank(candidates, req.desired_completion_date) if candidates else None

    row = formatter.build_output_row(
        req.request_id, winner, req.requested_amount, amount_safe,
        earliest_full, req.request_date, req.desired_completion_date,
    )
    if winner and winner.payment_option_id:
        row["_payment_option_id"] = winner.payment_option_id

    context = {
        "requested_amount": req.requested_amount,
        "request_date": req.request_date,
        "desired_completion_date": req.desired_completion_date,
        "known_payment_option_ids": {o.payment_option_id for o in payment_options},
        "flexible_event_ids": flexible_event_ids,
    }
    return row, context


def _expand_option_payments(option: loader.RawPaymentOption) -> list:
    payments = []
    d = option.first_payment_date
    for _ in range(option.num_payments):
        payments.append((d, option.per_payment_amount))
        d = d + timedelta(days=option.interval_days)
    return payments


def preflight_check() -> None:
    """One real call to call_model_fn before the main loop. A config/auth
    failure here is not a 'one bad row' problem (see docs/threat_model.md
    section 7 for why row-level failures are handled differently) -- it will
    fail identically on every request, so surface it once, loudly, and stop,
    instead of silently degrading all N rows to fallback and hiding the
    actual cause in N copies of the same traceback.
    """
    log.info("running preflight check against LLM_PROVIDER=%s, model=%s", LLM_PROVIDER, MODEL_NAME)
    try:
        call_model_fn(
            "Reply with exactly this JSON and nothing else.",
            '{"confirmed_amounts": [], "cancellations": [], "delays": [], "confirmations": []}',
        )
    except Exception as exc:
        log.error(
            "preflight check failed -- every row would fail identically, so stopping now "
            "instead of burning the full dataset on a broken config.\n"
            "Common causes: API key not set/exported in this shell (%s), "
            "provider SDK not installed for LLM_PROVIDER=%s, or an invalid model name (%s).\n"
            "Underlying error: %r",
            "ANTHROPIC_API_KEY" if LLM_PROVIDER == "anthropic" else "GEMINI_API_KEY / GOOGLE_API_KEY",
            LLM_PROVIDER, MODEL_NAME, exc,
        )
        raise
    log.info("preflight check passed, proceeding with the full run")


def run(dataset_dir: Path, output_path: Path) -> None:
    preflight_check()

    requests = loader.load_requests(dataset_dir / "requests.csv")
    profiles = loader.load_profiles(dataset_dir / "financial_profiles.csv")
    events_by_user = loader.load_events(dataset_dir / "financial_events.csv")
    rates = loader.load_exchange_rates(dataset_dir / "exchange_rates.csv")
    options_by_request = loader.load_payment_options(dataset_dir / "request_payment_options.csv")
    messages = loader.load_messages(dataset_dir / "messages.csv")

    rows = []
    contexts = {}
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
            row, ctx = process_request(req, profile, raw_events, rates, options, msg_texts)
        except Exception:
            log.exception("request %s failed, using fallback row", req.request_id)
            row = build_fallback_row(req.request_id)
            ctx = {
                "requested_amount": req.requested_amount,
                "request_date": req.request_date,
                "desired_completion_date": req.desired_completion_date,
                "known_payment_option_ids": set(),
                "flexible_event_ids": set(),
            }
        rows.append(row)
        contexts[req.request_id] = ctx

    violations = verifier.verify_all(rows, contexts)
    for request_id, problems in violations.items():
        log.warning("request %s failed verification: %s -- replacing with fallback", request_id, problems)
        idx = next(i for i, r in enumerate(rows) if r["request_id"] == request_id)
        rows[idx] = build_fallback_row(request_id)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in OUTPUT_COLUMNS})

    log.info("wrote %d rows to %s (%d needed fallback)", len(rows), output_path, len(violations))

    usage_report_path = output_path.parent / "evaluation" / "usage_report.md"
    usage_report_path.parent.mkdir(parents=True, exist_ok=True)
    write_usage_report(usage_report_path, num_requests=len(requests))
    log.info("wrote token usage report to %s", usage_report_path)


if __name__ == "__main__":
    # Resolve relative to this file's own location, not the current working
    # directory -- so this runs correctly whether launched from the project
    # root (`python code/main.py`) or from inside code/ itself
    # (`python main.py`), which is the actual mix seen across this build.
    project_root = Path(__file__).resolve().parent.parent
    dataset_dir = project_root / "dataset"
    output_path = project_root / "output.csv"
    log.info("using dataset_dir=%s, output_path=%s", dataset_dir, output_path)
    run(dataset_dir, output_path)
