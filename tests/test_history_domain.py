"""History is a causal proof: immutable metadata must commit to every plan input."""

from dataclasses import FrozenInstanceError, replace

import pytest

from opencollab.domain.history import HistoryEntry, ReplayPlan, TurnRecord, validate_lineage


def entry(key, parents=()):
    return HistoryEntry(key, "team", 0, 0, "initial", "stable", "2026-01-01T00:00:00Z", parent_entry_ids=parents)


def test_history_values_are_frozen_and_digest_covers_lineage():
    value = entry("a")
    assert not hasattr(value, "__dict__")
    with pytest.raises(FrozenInstanceError):
        value.epoch = 2
    assert value.digest() == replace(value).digest()
    assert value.digest() != replace(value, epoch=1).digest()
    turn = TurnRecord("team", "turn", 0, 0, 1, "completed", "time")
    assert not hasattr(turn, "__dict__")
    plan = ReplayPlan("entry", "cp", ((0, "cp"),), ((0, 0),), (), ((0, ()),), "history")
    assert plan.digest() != replace(plan, epoch_by_agent=((0, 1),)).digest()
    assert plan.digest() != replace(plan, effect_fingerprints=(("effect", "digest"),)).digest()


@pytest.mark.parametrize("parents", [("a",), ("b", "b")])
def test_self_and_duplicate_parent_rejected(parents):
    with pytest.raises(ValueError):
        entry("a", parents)


def test_graph_rejects_unknown_cycle_and_invalidated_parent():
    with pytest.raises(ValueError, match="unknown"):
        validate_lineage({"a": entry("a", ("missing",))})
    with pytest.raises(ValueError, match="cycle"):
        validate_lineage({"a": entry("a", ("b",)), "b": entry("b", ("a",))})
    with pytest.raises(ValueError, match="invalidated"):
        validate_lineage({"a": replace(entry("a"), status="invalidated"), "b": entry("b", ("a",))})


def test_long_lineage_uses_iteration():
    nodes = {}
    for index in range(2048):
        key = str(index)
        nodes[key] = entry(key, (str(index - 1),) if index else ())
    validate_lineage(nodes)
