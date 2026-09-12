"""Regression probes for graph commit and rollback trace boundaries."""

from types import SimpleNamespace

import pytest
from test_rollback_audit import _effect, _ready_scheduler

from opencollab.application import rollback
from opencollab.domain.rollback import EnvironmentSnapshot


async def test_public_consumer_failure_reverts_consumer_and_frontier(monkeypatch):
    scheduler, _ = await _ready_scheduler()
    effect = _effect(scheduler)
    service = scheduler._rollback_service
    original = service.register_consumer

    def fail_after_write(*args):
        original(*args)
        raise RuntimeError("consumer commit failed")

    monkeypatch.setattr(service, "register_consumer", fail_after_write)
    with pytest.raises(RuntimeError, match="consumer commit"):
        scheduler.consume_effect_ref(effect.effect_id, 1)
    assert not service._consumers
    assert not service.causal_frontier(1)
    assert service.effects[effect.effect_id].status == "active"


async def test_invalidation_failure_reverts_graph_and_keeps_retryable_fence(monkeypatch):
    scheduler, _ = await _ready_scheduler()
    first = _effect(scheduler)
    scheduler.consume_effect_ref(first.effect_id, 1)
    second = _effect(scheduler, 1)
    service = scheduler._rollback_service
    original = rollback.replace
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("invalidation commit failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(rollback, "replace", fail_second)
    plan = scheduler.preview_rollback({first.effect_id})
    with pytest.raises(RuntimeError, match="invalidation commit"):
        await scheduler.rollback_effect({first.effect_id}, expected_plan_digest=plan.digest())
    assert service.effects[first.effect_id].status == "active"
    assert service.effects[second.effect_id].status == "active"
    assert service.causal_frontier(1) == frozenset({first.effect_id, second.effect_id})
    assert scheduler._rollback_failed == {0, 1}
    assert scheduler._rollback_fenced == {0, 1}
    monkeypatch.setattr(rollback, "replace", original)
    result = await scheduler.rollback_effect({first.effect_id}, expected_plan_digest=plan.digest())
    assert result.invalidated
    scheduler.resume_after_rollback({0, 1})


@pytest.mark.parametrize("reason", [
    "stale_or_fenced", "rollback_epoch_invalidated", "sender_epoch_stale", "target_epoch_stale",
])
async def test_rollback_refusal_trace_excludes_payload_and_summary(reason):
    scheduler, _ = await _ready_scheduler()
    records = []
    scheduler._tracer = SimpleNamespace(log_step=lambda **entry: records.append(entry))
    scheduler._trace_message_decision(
        "message_refused", from_aid=0, to_aid=1,
        summary="private-summary", content="private-body 1234567", reason=reason,
    )
    assert len(records) == 1
    assert records[0]["payload"]["reason"] == reason
    assert records[0]["payload"]["content_chars"] > 0
    assert "private" not in str(records)
    assert "1234567" not in str(records)


async def test_checkpoint_and_operation_trace_contain_no_scope_environment_values():
    scheduler, _ = await _ready_scheduler()
    records = []
    scheduler._rollback_service._trace = SimpleNamespace(log_step=lambda **entry: records.append(entry))
    scheduler._sessions[0].env.replace_environment(
        EnvironmentSnapshot.from_mapping({"PRIVATE_TOKEN": "not-for-the-trace"})
    )
    checkpoint = await scheduler.create_checkpoint(0)
    await scheduler.rollback_to_checkpoint(0, checkpoint.checkpoint_id)
    assert {entry["step_type"] for entry in records} >= {"checkpoint_created", "rollback_operation_settled"}
    assert "PRIVATE_TOKEN" not in str(records)
    assert "not-for-the-trace" not in str(records)
