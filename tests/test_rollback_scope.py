from __future__ import annotations

import os

import pytest

from opencollab.adapters._env_scope import _ScopeState
from opencollab.domain.rollback import EffectRef, reduce_causal_frontier


def _effect(effect_id: str, parents: tuple[str, ...] = ()) -> EffectRef:
    return EffectRef(
        effect_id=effect_id,
        producer_aid=1,
        kind="tool_result",
        epoch=0,
        attempt=0,
        parent_effect_ids=parents,
    )


def test_frontier_reduction_removes_all_represented_ancestors() -> None:
    effects = {"a": _effect("a"), "b": _effect("b", ("a",)), "c": _effect("c", ("b",))}
    assert reduce_causal_frontier(effects, {"a"}, "c") == frozenset({"c"})


def test_scope_is_exact_and_does_not_mutate_process_environment() -> None:
    original = os.environ.get("OPENCollab_TEST_SCOPE")
    scope = _ScopeState({"A": "one"})
    snapshot = scope.snapshot()
    scope.set("B", "two")
    scope.unset("A")
    scope.replace(snapshot)
    assert dict(scope.view()) == {"A": "one"}
    assert os.environ.get("OPENCollab_TEST_SCOPE") == original


def test_scope_rejects_control_plane_and_invalid_names() -> None:
    scope = _ScopeState({})
    with pytest.raises(ValueError):
        scope.set("OPENCOLLAB_ROLLBACK_KEY", "secret")
    with pytest.raises(ValueError):
        scope.set("bad-name", "value")
