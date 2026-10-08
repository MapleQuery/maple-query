"""Row shapes for the curated tables. Field names and order match
infra/terraform/schemas/curated_*.json; tests/unit/test_schema_drift.py
fails if they diverge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Person:
    person_id: str
    name: str
    given_name: str | None
    family_name: str | None
    openparliament_url: str
    current_mp: bool
    first_term_start: date | None
    last_term_end: date | None


@dataclass(frozen=True)
class PersonName:
    person_id: str
    name_norm: str
    name_raw: str
    origin: str  # openparliament_name | given_family | alternate_name


@dataclass(frozen=True)
class PersonTerm:
    term_id: str
    person_id: str
    role: str
    start_date: date
    end_date: date | None
    party: str | None
    riding_name: str | None
    province: str | None
    label: str | None


@dataclass
class PersonContribution:
    contribution_key: str
    person_id: str | None
    status: str  # linked | ambiguous
    candidates: list[str]
    match_method: str
    confidence: float
    political_entity: str
    recipient_id: str
    recipient_name: str
    recipient_party: str | None
    electoral_district: str | None
    electoral_event: str | None
    report_date: date | None
    financial_report: str | None
    contributor_type: str
    contributor_name: str | None
    contributor_province: str | None
    contribution_count: int = 0
    monetary_total: str = "0.00"  # NUMERIC as an exact decimal string
    non_monetary_total: str = "0.00"
    first_received: date | None = None
    last_received: date | None = None
    source_object: str = ""
