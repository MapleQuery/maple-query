"""The `Derivation`: a deterministic, machine-checkable account of how
one numeric result was computed.

Built from what the executor actually ran — the SQL text, the resolved
columns, the aggregation, the source documents, and the real result
payload — never from a model narration of what it thinks it did. An LLM
asked "what did you compute" would have reported the ``$8`` total-
spending sum and the ``$900.84B`` double-count as legitimate; the whole
point of capturing rather than narrating is that those figures arrive
with an inspectable provenance instead of a confident sentence.

This module only *captures*. Grounding (does the answer cite it),
magnitude checking (is it plausible), and surfacing (the trace panel)
are downstream and consume the object this builds. Construction never
raises into the loop: any parse failure or missing metadata yields a
derivation flagged ``complete=False``, and the research phase carries
on exactly as it did before derivations existed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot
import sqlglot.expressions as exp

from semantic_enrich.core.agent.derivation_units import (
    UnitScale,
    UnitSource,
    resolve_unit_scale,
)
from semantic_enrich.core.sql_normalize import _mask_string_literals

if TYPE_CHECKING:
    from semantic_enrich.core.agent_tools import LoopState


@dataclass(frozen=True)
class Derivation:
    """How one numeric ``run_sql`` result was produced. Frozen and
    JSON-serializable via :meth:`to_dict`."""

    # ── provenance ──
    source_packages: tuple[str, ...]
    source_documents: tuple[str, ...]
    dataset_titles: tuple[str, ...]
    # ── computation ──
    aggregation: str  # "SUM" | "AVG" | "COUNT" | ... | "none"
    value_columns: tuple[str, ...]
    group_by_columns: tuple[str, ...]
    predicate_shape: str
    sql_shape: str
    row_count: int  # rows RETURNED (1 for a scalar aggregate)
    source_row_estimate: int  # rows DREW FROM (sum of source-doc row_count)
    # ── the number ──
    result_value: float | None
    result_label: str | None
    # ── units ──
    unit_scale: UnitScale
    unit_source: UnitSource
    # ── capture health ──
    complete: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Aggregate expression class -> canonical function name.
_AGG_NAMES: dict[type[exp.Expression], str] = {
    exp.Sum: "SUM",
    exp.Avg: "AVG",
    exp.Count: "COUNT",
    exp.Min: "MIN",
    exp.Max: "MAX",
}


def _incomplete(sql: str, note: str) -> Derivation:
    """A capture that could not be built; safe, inert, flagged."""
    return Derivation(
        source_packages=(),
        source_documents=(),
        dataset_titles=(),
        aggregation="none",
        value_columns=(),
        group_by_columns=(),
        predicate_shape="",
        sql_shape=_safe_mask(sql),
        row_count=0,
        source_row_estimate=0,
        result_value=None,
        result_label=None,
        unit_scale="unknown",
        unit_source="unresolved",
        complete=False,
        notes=(note,),
    )


def _safe_mask(sql: str) -> str:
    try:
        return _mask_string_literals(sql)
    except Exception:
        return ""


def build_derivation(
    *, sql: str, result: dict[str, Any], state: LoopState
) -> list[Derivation]:
    """Assemble the :class:`Derivation`(s) for one ``run_sql`` from the
    executed SQL and its result payload. Pure and deterministic; no model
    or warehouse call. Never raises.

    Returns **one derivation per scalar-aggregate output column** — a
    ``SELECT SUM(a), SUM(b)`` yields two traced totals, not zero — plus a
    single shape-only derivation for a grouped or non-aggregate query."""
    try:
        return _build(sql=sql, result=result, state=state)
    except Exception as exc:  # pragma: no cover - defensive
        return [_incomplete(sql, f"build_error:{type(exc).__name__}")]


def _build(
    *, sql: str, result: dict[str, Any], state: LoopState
) -> list[Derivation]:
    from semantic_enrich.core import agent_tools

    tree = sqlglot.parse_one(sql, dialect="bigquery")
    if tree is None:
        return [_incomplete(sql, "unparseable_sql")]

    has_group_by = tree.find(exp.Group) is not None

    # ── shared provenance (same for every column of one query) ──
    source_documents = tuple(sorted(agent_tools._extract_inlined_document_ids(sql)))
    packages: list[str] = []
    titles: list[str] = []
    row_estimate = 0
    for doc_id in source_documents:
        pkg = state.doc_package.get(doc_id)
        if pkg and pkg not in packages:
            packages.append(pkg)
        title = state.doc_title.get(doc_id)
        if title and title not in titles:
            titles.append(title)
        row_estimate += int(state.doc_row_count.get(doc_id) or 0)
    shared: dict[str, Any] = {
        "source_packages": tuple(packages),
        "source_documents": source_documents,
        "dataset_titles": tuple(titles),
        "predicate_shape": _predicate_shape(tree),
        "sql_shape": _safe_mask(sql),
        "row_count": int(result.get("row_count") or 0),
        "source_row_estimate": row_estimate,
    }
    rows = result.get("rows") or []
    row0 = rows[0] if rows and isinstance(rows[0], dict) else None

    # ── grouped or non-aggregate: one shape-only derivation ──
    if has_group_by:
        agg, cols = _aggregation_and_value_columns(tree)
        unit_scale, unit_source = _resolve_units(
            value_columns=cols, aggregation=agg, state=state
        )
        return [
            Derivation(
                **shared,
                aggregation=agg,
                value_columns=cols,
                group_by_columns=_group_by_columns(tree),
                result_value=None,
                result_label=None,
                unit_scale=unit_scale,
                unit_source=unit_source,
                complete=True,
                notes=("grouped_aggregate",),
            )
        ]

    projections = _scalar_aggregate_projections(tree)
    if not projections:
        agg, cols = _aggregation_and_value_columns(tree)
        unit_scale, unit_source = _resolve_units(
            value_columns=cols, aggregation=agg, state=state
        )
        return [
            Derivation(
                **shared,
                aggregation=agg,
                value_columns=cols,
                group_by_columns=(),
                result_value=None,
                result_label=None,
                unit_scale=unit_scale,
                unit_source=unit_source,
                complete=True,
                notes=("no_scalar_aggregate",),
            )
        ]

    # ── one derivation per scalar-aggregate output column ──
    derivations: list[Derivation] = []
    for name, agg, value_cols in projections:
        notes: list[str] = []
        result_value: float | None = None
        result_label: str | None = None
        if shared["row_count"] != 1:
            notes.append(
                "multi_row_no_scalar" if shared["row_count"] > 1 else "zero_rows"
            )
        elif row0 is None:
            notes.append("no_result_rows")
        else:
            parsed = _to_float(row0.get(name))
            if parsed is None:
                notes.append("non_numeric_result")
            else:
                result_value = parsed
                result_label = name
        unit_scale, unit_source = _resolve_units(
            value_columns=value_cols, aggregation=agg, state=state
        )
        derivations.append(
            Derivation(
                **shared,
                aggregation=agg,
                value_columns=value_cols,
                group_by_columns=(),
                result_value=result_value,
                result_label=result_label,
                unit_scale=unit_scale,
                unit_source=unit_source,
                complete=True,
                notes=tuple(notes),
            )
        )
    return derivations


def _scalar_aggregate_projections(
    tree: exp.Expression,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Per ungrouped scalar-aggregate output column: ``(output_name,
    aggregation, value_columns)``. Output names match BigQuery's
    (``f0_``-style for anonymous projections), so they key the result
    row dict. Mirrors ``_scalar_aggregate_columns`` but keeps per-column
    detail so each total is traced separately."""
    from semantic_enrich.core import agent_tools

    out: list[tuple[str, str, tuple[str, ...]]] = []
    for select in tree.find_all(exp.Select):
        if select.args.get("group") is not None:
            continue
        anonymous_index = 0
        for projection in select.expressions:
            name = projection.alias_or_name
            if not name:
                name = f"f{anonymous_index}_"
                anonymous_index += 1
            scalar_aggs = [
                agg
                for agg in projection.find_all(exp.AggFunc)
                if agg.find_ancestor(exp.Window) is None
            ]
            if not scalar_aggs:
                continue
            agg_name = _dominant_agg_name(scalar_aggs)
            value_cols = tuple(
                sorted(agent_tools._extract_json_path_columns(projection.sql()))
            )
            out.append((name, agg_name, value_cols))
    return out


_AGG_RANK = {"SUM": 3, "AVG": 2, "MIN": 1, "MAX": 1, "COUNT": 0}


def _dominant_agg_name(aggs: list[exp.AggFunc]) -> str:
    """The money-bearing aggregate wins when a projection mixes them."""
    best_name = "none"
    best_rank = -1
    for agg in aggs:
        name = _agg_name(agg)
        rank = _AGG_RANK.get(name, 0)
        if rank > best_rank:
            best_name, best_rank = name, rank
    return best_name


def _aggregation_and_value_columns(
    tree: exp.Expression,
) -> tuple[str, tuple[str, ...]]:
    """The dominant scalar aggregate function and the JSON columns
    inside it. Prefers SUM/AVG (the money-bearing aggregates) over
    COUNT when both appear."""
    from semantic_enrich.core import agent_tools

    best: exp.AggFunc | None = None
    best_rank = -1
    _rank = {"SUM": 3, "AVG": 2, "MIN": 1, "MAX": 1, "COUNT": 0}
    for agg in tree.find_all(exp.AggFunc):
        if agg.find_ancestor(exp.Window) is not None:
            continue
        name = _agg_name(agg)
        rank = _rank.get(name, 0)
        if rank > best_rank:
            best, best_rank = agg, rank
    if best is None:
        return "none", ()
    name = _agg_name(best)
    columns = tuple(sorted(agent_tools._extract_json_path_columns(best.sql())))
    return name, columns


def _agg_name(agg: exp.AggFunc) -> str:
    canonical = _AGG_NAMES.get(type(agg))
    if canonical is not None:
        return canonical
    return (agg.key or type(agg).__name__).upper()


def _group_by_columns(tree: exp.Expression) -> tuple[str, ...]:
    from semantic_enrich.core import agent_tools

    keys: list[str] = []
    for group in tree.find_all(exp.Group):
        for e in group.expressions:
            json_cols = sorted(agent_tools._extract_json_path_columns(e.sql()))
            # Fall back to the bare identifier (a select alias) when the
            # GROUP BY key is not an inline JSON_VALUE path.
            names = json_cols or [e.name] if e.name else json_cols
            for col in names:
                if col and col not in keys:
                    keys.append(col)
    return tuple(keys)


def _predicate_shape(tree: exp.Expression) -> str:
    where = tree.find(exp.Where)
    if where is None:
        return ""
    return _safe_mask(where.sql(dialect="bigquery"))


def _resolve_units(
    *,
    value_columns: tuple[str, ...],
    aggregation: str,
    state: LoopState,
) -> tuple[UnitScale, UnitSource]:
    if aggregation == "COUNT":
        return resolve_unit_scale(
            column_name="", semantic_type=None, description=None, aggregation="COUNT"
        )
    # Resolve from the first value column carrying metadata; a monetary
    # scale on any contributing column wins over an unresolved one.
    best: tuple[UnitScale, UnitSource] | None = None
    for col in value_columns:
        meta = state.column_metadata.get(col)
        scale, source = resolve_unit_scale(
            column_name=col,
            semantic_type=(meta or {}).get("semantic_type"),
            description=(meta or {}).get("description"),
            aggregation=aggregation,
        )
        if scale not in ("unknown", "not_monetary"):
            return scale, source
        if best is None or (best[0] == "not_monetary" and scale == "unknown"):
            best = (scale, source)
    if best is not None:
        return best
    return "unknown", "unresolved"


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        try:
            f = float(value.replace(",", "").strip())
        except (ValueError, AttributeError):
            return None
    else:
        return None
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf guard
        return None
    return f
