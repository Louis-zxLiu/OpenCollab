"""Long causal chains must not depend on Python's recursion limit."""

from dataclasses import replace

import pytest

from opencollab.application.rollback import RollbackService


def _effect(service, effect_id, parents=()):
    return service.create_effect(
        effect_id=effect_id, producer_aid=0, kind="tool_result", epoch=0,
        attempt=0, parent_effect_ids=parents,
    )


def test_long_chain_can_be_created_and_planned_without_recursion():
    service = RollbackService()
    length = 2048
    for index in range(length):
        _effect(service, f"e{index}", (f"e{index - 1}",) if index else ())
    sibling = _effect(service, "sibling", ("e0",))

    plan = service.preview_rollback({"e1024"})

    assert plan.invalidated_effect_ids == frozenset(f"e{index}" for index in range(1024, length))
    assert plan.affected_agent_ids == frozenset({0})
    assert sibling.effect_id not in plan.invalidated_effect_ids
    assert "e0" not in plan.invalidated_effect_ids
    assert plan.digest() == service.preview_rollback({"e1024"}).digest()


def test_cycle_in_reachable_history_is_rejected_without_partial_insert():
    service = RollbackService()
    first = _effect(service, "a")
    _effect(service, "b", ("a",))
    # Fault injection retains the defensive check for corrupted internal state.
    service._effects["a"] = replace(first, parent_effect_ids=("b",))
    before = dict(service.effects)

    with pytest.raises(ValueError, match="cycle"):
        _effect(service, "c", ("b",))

    assert dict(service.effects) == before


def test_converging_dag_is_not_a_cycle():
    service = RollbackService()
    _effect(service, "root")
    _effect(service, "left", ("root",))
    _effect(service, "right", ("root",))
    joined = _effect(service, "joined", ("left", "right"))

    assert joined.parent_effect_ids == ("left", "right")
    assert service.preview_rollback({"left"}).invalidated_effect_ids == frozenset({"left", "joined"})
