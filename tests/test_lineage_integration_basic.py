"""Focused contracts for causal lineage and quarantine-based rollback."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

from session_characterization_test_support import FakeAgent

from opencollab.adapters.tools.invalidate_effect import InvalidateEffectTool
from opencollab.application.rollback import RollbackService
from opencollab.application.session import Session
from opencollab.application.session_snapshot import (
    _restore_pending_row,
    _serialize_pending_row,
)
from opencollab.application.shaping.rollback import RollbackQuarantineShaper
from opencollab.application.tool_execution import ToolRuntime
from opencollab.domain.pending import PendingEventTable, PendingRow, RowKind
from opencollab.domain.session import SessionState


def _effect(
    controller: RollbackService,
    *,
    producer: int,
    content: str,
    parents: tuple[str, ...] = (),
):
    return controller.create_effect(
        producer_aid=producer,
        attempt=0,
        branch_id="main",
        epoch=0,
        kind="child_result",
        parent_effect_ids=parents,
        content=content,
    )


def test_quarantine_closes_over_descendants_and_reports_consumers() -> None:
    controller = RollbackService(None)
    root = _effect(controller, producer=2, content="incorrect premise")
    child = _effect(
        controller,
        producer=1,
        content="derived answer",
        parents=(root.effect_id,),
    )
    controller.register_consumer(root.effect_id, 1)
    controller.register_consumer(child.effect_id, 0)

    affected = controller.quarantine(root.effect_id, "test disproved premise")

    assert affected == {root.effect_id, child.effect_id}
    assert controller.compute_affected_agents(affected) == {0, 1}
    assert all(controller.is_quarantined(effect_id) for effect_id in affected)


def test_pending_sidecar_survives_delivery_and_shaper_masks_only_model_view() -> None:
    controller = RollbackService(None)
    effect = _effect(controller, producer=1, content="bad output")
    envelope = {"effect": asdict(effect), "consumer_aid": 0}
    table = PendingEventTable()
    table.add(
        PendingRow(
            tool_call_id="call-1",
            kind=RowKind.CHILD_AGENT,
            order=0,
            lineage=envelope,
        )
    )
    table.fill("call-1", result="bad output")
    message = table.ordered_results()[0]
    original = dict(message)

    assert message["content"] == f"[effect_id: {effect.effect_id}]\nbad output"

    controller.quarantine(effect.effect_id, "invalid")
    shaped = RollbackQuarantineShaper(controller).shape([message])

    assert message == original
    assert shaped[0]["content"] == f"[quarantined evidence: {effect.effect_id}]"
    assert shaped[0]["_lineage"] == envelope

    restored_row = _restore_pending_row(_serialize_pending_row(table.rows["call-1"]))
    assert restored_row is not None
    assert restored_row.lineage == envelope


def test_invalidate_tool_delegates_reason_and_evidence_to_scheduler() -> None:
    controller = RollbackService(None)
    effect = _effect(controller, producer=1, content="result")
    calls = []

    async def handle(effect_id, reporter_aid, reason, evidence):
        calls.append((effect_id, reporter_aid, reason, evidence))
        return controller.quarantine(effect_id, reason, evidence)

    scheduler = SimpleNamespace(
        _lineage=controller,
        _tracer=None,
        _handle_invalidation=handle,
    )
    runtime = ToolRuntime(None, None, None, aid=7)

    result = asyncio.run(
        InvalidateEffectTool(scheduler).execute_with_runtime(
            {
                "effect_id": effect.effect_id,
                "reason": "wrong assumption",
                "evidence": "pytest failure",
            },
            runtime,
        )
    )

    assert calls == [(effect.effect_id, 7, "wrong assumption", "pytest failure")]
    assert "Invalidated effect" in result


def test_lineage_state_is_written_to_session_snapshot() -> None:
    session = Session.__new__(Session)
    session.agent = FakeAgent()
    session.state = SessionState(messages=[])
    session.state.lineage_branch_id = "repair"
    session.state.lineage_epoch = 2
    session.state.lineage_attempt = 3
    session.state.consumed_effect_ids = {"e_parent"}
    session.state.quarantined_effect_ids = {"e_bad"}

    _, meta = session._snapshot_for_save()
    state = meta["session_state"]

    assert state["lineage_branch_id"] == "repair"
    assert state["lineage_epoch"] == 2
    assert state["lineage_attempt"] == 3
    assert state["consumed_effect_ids"] == ["e_parent"]
    assert state["quarantined_effect_ids"] == ["e_bad"]
