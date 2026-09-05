from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencollab.application.coordination_protocol import CoordinationProtocol
from opencollab.domain.coordination_protocol import (
    AgentLifecycleStatus,
    CoordinationWait,
    VerificationEvidence,
    WaitKind,
    wait_matches,
)


def _state(**kwargs):
    state = SimpleNamespace(
        lifecycle_status=AgentLifecycleStatus.ACTIVE,
        required_effect_ids=set(),
        wait_condition=None,
        verification_evidence={},
        **kwargs,
    )
    state.register_visible_effect = lambda effect_id: state.required_effect_ids.add(effect_id)
    state.adopt_visible_effect = lambda effect_id: state.required_effect_ids.discard(effect_id)
    state.can_execute_normal_work = lambda: not state.required_effect_ids
    state.mark_superseded = lambda: setattr(
        state, "lifecycle_status", AgentLifecycleStatus.SUPERSEDED
    )
    return state


def test_wait_matching_and_lifecycle_are_generic():
    wait = CoordinationWait(WaitKind.EFFECT, frozenset({"e1"}), "waiting")
    assert wait_matches(wait, WaitKind.EFFECT, "e1")
    assert not wait_matches(wait, WaitKind.EFFECT, "e2")
    assert not wait_matches(wait, WaitKind.MESSAGE)

    protocol = CoordinationProtocol()
    state = _state()
    protocol.register_agent(7, state, "arbitrary-role", None)
    protocol.register_visible_effect(7, "e1")
    assert not protocol.can_execute(7, "file_read")
    assert "e1" in state.required_effect_ids
    state.adopt_visible_effect("e1")
    assert protocol.can_execute(7, "file_read")
    protocol.supersede(7)
    assert not protocol.is_addressable(7)
    with pytest.raises(ValueError, match="no longer addressable"):
        protocol.require_addressable(7)


def test_wait_wakes_only_for_requested_effect_and_sources_need_adoption():
    protocol = CoordinationProtocol()
    state = _state()
    protocol.register_agent(1, state, "coordinator", None)
    protocol.set_wait(1, CoordinationWait(WaitKind.EFFECT, frozenset({"e2"})))
    assert not protocol.clear_wait_if_matches(1, WaitKind.EFFECT, "e1")
    assert state.wait_condition is not None
    assert protocol.clear_wait_if_matches(1, WaitKind.EFFECT, "e2")
    protocol.register_visible_effect(1, "e3")
    assert protocol.validate_source_effects(1, ["e3"]) is not None
    state.adopt_visible_effect("e3")
    assert protocol.validate_source_effects(1, ["e3"]) is None


def test_verification_requires_positive_evidence_for_exact_effect():
    protocol = CoordinationProtocol()
    state = _state()
    protocol.register_agent(1, state, "verifier", None)
    evidence = VerificationEvidence(
        evidence_id="v1",
        effect_id="e1",
        tool_call_id="tc1",
        tool_name="run_tests",
        command="python -m pytest tests/test_x.py",
        exit_code=0,
        verified=True,
    )
    protocol.record_verification(1, evidence)
    assert protocol.validate_verification(1, "e1", ["v1"])
    assert not protocol.validate_verification(1, "e2", ["v1"])
