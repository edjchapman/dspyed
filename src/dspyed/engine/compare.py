"""Result-set comparison for execution accuracy.

The headline metric hinges on this module: two executed result sets are
compared as multisets of normalized rows (ordered comparison only when the
gold query demands an order). Every normalization rule below is a documented
measurement decision, not a convenience:

- NULL is a singleton sentinel — equal only to itself, distinct from 0 and "".
- Floats are rounded to 4 decimal places: AVG() precision noise is forgiven,
  everything coarser than 1e-4 is a real difference. (NaN never matches —
  SQLite essentially never produces it, and a NaN match would be meaningless.)
- bool → int (bool is a subclass of int in Python, so the bool branch must be
  checked first); bytes → hex string (stable, hashable, printable).
- Duplicates matter: multisets are compared via Counter, so a missing
  DISTINCT is correctly scored as a miss.

The *strict* comparison (column values in each query's written order) is the
headline. The *loose* retry (cells sorted within each row) is a diagnostic
only — it quantifies how much strict column-order costs us versus the
official Spider evaluator, and is never counted as a match.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

type RawCell = int | float | str | bytes | bool | None
type RawRow = tuple[RawCell, ...]

_FLOAT_PLACES: Final = 4


class _Null:
    """Singleton NULL sentinel: equal only to itself (default object identity)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NULL"


NULL: Final = _Null()

type NormCell = int | float | str | _Null
type NormRow = tuple[NormCell, ...]


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing predicted rows against gold rows.

    ``matched`` is the only field that feeds the headline metric. The rest are
    diagnostics whose aggregate rates are published alongside the number.
    """

    matched: bool
    matched_loose: bool  # strict failed but cells-sorted-within-row matched (diagnostic only)
    both_empty: bool  # matched, but on zero rows each — weak evidence, rate is reported


def normalize_cell(cell: RawCell) -> NormCell:
    # bool first: isinstance(True, int) is True, so the int branch would swallow it.
    if isinstance(cell, bool):
        return int(cell)
    if cell is None:
        return NULL
    if isinstance(cell, float):
        return round(cell, _FLOAT_PLACES)
    if isinstance(cell, bytes):
        return cell.hex()
    return cell


def normalize_row(row: RawRow) -> NormRow:
    return tuple(normalize_cell(cell) for cell in row)


def _loose_key(cell: NormCell) -> tuple[str, str]:
    """Total order over mixed-type cells: sort by (type name, repr)."""
    return (type(cell).__name__, repr(cell))


def _sorted_within_row(row: NormRow) -> NormRow:
    return tuple(sorted(row, key=_loose_key))


def _match(gold: list[NormRow], pred: list[NormRow], *, ordered: bool) -> bool:
    if ordered:
        return gold == pred
    return Counter(gold) == Counter(pred)


def compare_results(
    gold_rows: list[RawRow],
    pred_rows: list[RawRow],
    *,
    ordered: bool,
) -> ComparisonResult:
    """Compare executed result sets.

    ``ordered`` must be True iff the *gold* query has a top-level ORDER BY —
    the caller (the metric) decides that by parsing the gold SQL; this module
    never sees SQL text.
    """
    gold = [normalize_row(row) for row in gold_rows]
    pred = [normalize_row(row) for row in pred_rows]

    matched = _match(gold, pred, ordered=ordered)
    both_empty = matched and not gold

    # Loose diagnostic: only attempted when strict failed and the shapes agree.
    matched_loose = False
    if not matched and gold and pred and len(gold[0]) == len(pred[0]):
        matched_loose = _match(
            [_sorted_within_row(row) for row in gold],
            [_sorted_within_row(row) for row in pred],
            ordered=ordered,
        )

    return ComparisonResult(matched=matched, matched_loose=matched_loose, both_empty=both_empty)
