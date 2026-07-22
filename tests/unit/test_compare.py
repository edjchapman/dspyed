"""Table-driven edge cases for result-set comparison.

Every case here encodes a measurement decision from engine/compare.py's
docstring — if one of these flips, the headline metric changed meaning.
"""

import pytest

from dspyed.engine import ComparisonResult, compare_results

# (test id, gold rows, pred rows, ordered, expected matched)
CASES = [
    ("identical-unordered", [(1, "a")], [(1, "a")], False, True),
    ("identical-ordered", [(1,), (2,)], [(1,), (2,)], True, True),
    ("reordered-rows-unordered", [(1,), (2,)], [(2,), (1,)], False, True),
    ("reordered-rows-ordered", [(1,), (2,)], [(2,), (1,)], True, False),
    ("duplicate-row-counts-differ", [(1,), (1,)], [(1,)], False, False),
    ("duplicate-row-counts-match", [(1,), (1,)], [(1,), (1,)], False, True),
    ("float-noise-forgiven", [(1 / 3,)], [(0.3333,)], False, True),
    ("float-real-difference", [(0.3334,)], [(0.3333,)], False, False),
    ("null-is-not-zero", [(None,)], [(0,)], False, False),
    ("null-is-not-empty-string", [(None,)], [("",)], False, False),
    ("null-equals-null", [(None,)], [(None,)], False, True),
    ("bool-equals-int", [(True,)], [(1,)], False, True),
    ("bytes-normalize-to-hex", [(b"\x01\xff",)], [("01ff",)], False, True),
    ("empty-vs-nonempty", [], [(1,)], False, False),
    ("value-mismatch", [(1,)], [(2,)], False, False),
]


@pytest.mark.parametrize(
    ("gold", "pred", "ordered", "matched"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_matched(gold, pred, ordered, matched):
    assert compare_results(gold, pred, ordered=ordered).matched is matched


def test_both_empty_matches_but_is_flagged():
    result = compare_results([], [], ordered=False)
    assert result == ComparisonResult(matched=True, matched_loose=False, both_empty=True)


def test_loose_diagnostic_fires_on_column_order_swap():
    result = compare_results([(1, "a")], [("a", 1)], ordered=False)
    assert result.matched is False  # never counted in the headline
    assert result.matched_loose is True


def test_loose_not_attempted_on_differing_column_counts():
    result = compare_results([(1, "a")], [(1,)], ordered=False)
    assert result.matched is False
    assert result.matched_loose is False


def test_loose_respects_ordering():
    # Same cells per row, but row order differs under ordered comparison:
    # loose sorts within rows, not across them.
    result = compare_results([(1, "a"), (2, "b")], [("b", 2), ("a", 1)], ordered=True)
    assert result.matched is False
    assert result.matched_loose is False
