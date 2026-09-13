"""
loader.py -- step 1 of the pipeline. Reads the raw CSVs into typed objects.

No decision logic here, and no currency conversion or conflict resolution --
that's resolver.py's job. This module only parses and normalizes shapes.

Column names below follow the snake_case schema described in
problem_statement.md. See docs/data_analysis.md for open questions to
confirm once the real dataset/ folder is available -- if actual column
names differ, this is the only file that needs to change.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import csv


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()


def _parse_optional_float(s: str):
    s = (s or "").strip()
    return float(s) if s else None


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: float
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class Profile:
    user_id: str
    home_currency: str
    available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: str
    flexible_spending_preferences: str
    payment_methods_user_will_consider: set[str]


@dataclass(frozen=True)
class RawEvent:
    """Pre-resolution event record. amount may be None (blank, needs image
    lookup). currency may differ from the user's home_currency, resolved by
    resolver.py before this ever reaches forecast.py.
    """
    event_id: str
    user_id: str
    date: date
    amount: float | None
    currency: str
    kind: str          # e.g. "income" | "expense" | "transfer" | "investment"
    recurring: bool
    flexible: bool
    interval_days: int | None
    status: str         # e.g. "confirmed" | "pending" | "failed" | "cancelled"
    linked_event_id: str | None


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: float


@dataclass(frozen=True)
class RawPaymentOption:
    request_id: str
    payment_option_id: str
    first_payment_date: date
    interval_days: int
    num_payments: int
    per_payment_amount: float
    total_payable: float
    financing_fee: float


@dataclass(frozen=True)
class Message:
    user_id: str | None
    request_id: str | None
    related_event_id: str | None
    text: str


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    user_id: str | None
    request_id: str | None
    related_event_id: str | None
    path: Path


def load_requests(path: Path) -> list[Request]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Request(
                request_id=row["request_id"],
                user_id=row["user_id"],
                request_date=_parse_date(row["request_date"]),
                request_type=row["request_type"],
                requested_amount=float(row["requested_amount"]),
                desired_completion_date=_parse_date(row["desired_completion_date"]),
                allows_partial_payment=row["allows_partial_payment"].strip().lower() in ("true", "1", "yes"),
                request_text=row.get("request_text", ""),
            ))
    return out


def load_profiles(path: Path) -> dict[str, Profile]:
    profiles = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            methods = {m.strip() for m in row["payment_methods_user_will_consider"].split("|") if m.strip()}
            profiles[row["user_id"]] = Profile(
                user_id=row["user_id"],
                home_currency=row["home_currency"],
                available_balance=float(row["available_balance"]),
                minimum_balance_to_keep=float(row["minimum_balance_to_keep"]),
                financial_priorities=row.get("financial_priorities", ""),
                flexible_spending_preferences=row.get("flexible_spending_preferences", ""),
                payment_methods_user_will_consider=methods,
            )
    return profiles


def load_events(path: Path) -> dict[str, list[RawEvent]]:
    by_user: dict[str, list[RawEvent]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ev = RawEvent(
                event_id=row["event_id"],
                user_id=row["user_id"],
                date=_parse_date(row["date"]),
                amount=_parse_optional_float(row.get("amount", "")),
                currency=row.get("currency", ""),
                kind=row.get("kind", ""),
                recurring=row.get("recurring", "").strip().lower() in ("true", "1", "yes"),
                flexible=row.get("flexible", "").strip().lower() in ("true", "1", "yes"),
                interval_days=int(row["interval_days"]) if row.get("interval_days") else None,
                status=row.get("status", "confirmed"),
                linked_event_id=row.get("linked_event_id") or None,
            )
            by_user.setdefault(ev.user_id, []).append(ev)
    return by_user


def load_exchange_rates(path: Path) -> list[ExchangeRate]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(ExchangeRate(
                rate_date=_parse_date(row["rate_date"]),
                from_currency=row["from_currency"],
                to_currency=row["to_currency"],
                rate=float(row["rate"]),
            ))
    return out


def load_payment_options(path: Path) -> dict[str, list[RawPaymentOption]]:
    by_request: dict[str, list[RawPaymentOption]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            opt = RawPaymentOption(
                request_id=row["request_id"],
                payment_option_id=row["payment_option_id"],
                first_payment_date=_parse_date(row["first_payment_date"]),
                interval_days=int(row["interval_days"]),
                num_payments=int(row["num_payments"]),
                per_payment_amount=float(row["per_payment_amount"]),
                total_payable=float(row["total_payable"]),
                financing_fee=float(row.get("financing_fee", 0) or 0),
            )
            by_request.setdefault(opt.request_id, []).append(opt)
    return by_request


def load_messages(path: Path) -> list[Message]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.append(Message(
                user_id=row.get("user_id") or None,
                request_id=row.get("request_id") or None,
                related_event_id=row.get("related_event_id") or None,
                text=row.get("text", ""),
            ))
    return out


def load_images(path: Path, media_dir: Path) -> list[ImageRef]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = row["image_id"]
            out.append(ImageRef(
                image_id=image_id,
                user_id=row.get("user_id") or None,
                request_id=row.get("request_id") or None,
                related_event_id=row.get("related_event_id") or None,
                path=media_dir / f"{image_id}.png",
            ))
    return out
