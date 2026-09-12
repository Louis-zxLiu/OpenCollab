"""Redacted history and replay facades over the application contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from opencollab.application.history import HistoryPort, HistoryView
from opencollab.application.scheduler_types import SchedulerTurnError

_SAFE_REASONS = frozenset({
    "provider_unavailable", "provider_rate_limited", "provider_timeout",
    "step_limit_reached", "context_overflow", "budget_exhausted",
    "effect_registration_failed", "persistence_failure", "rollback_interrupted",
    "replay_start_failed", "cancelled", "timeout",
})


def _failure_reason(exc: BaseException) -> str:
    if isinstance(exc, SchedulerTurnError):
        reason = exc.terminal_reason
        if reason in _SAFE_REASONS:
            return reason
        lowered = (reason or "").lower()
        if "ratelimiterror" in lowered or "rate_limit_error" in lowered or "429" in lowered:
            return "provider_rate_limited"
        if "internalservererror" in lowered or "server_error" in lowered or "503" in lowered:
            return "provider_unavailable"
        if "timeout" in lowered or "timed out" in lowered:
            return "provider_timeout"
        if "step limit reached" in lowered:
            return "step_limit_reached"
        if "context overflow" in lowered or "context window" in lowered:
            return "context_overflow"
        return exc.phase.value
    return "replay_failed"


class HistoryClient:
    def __init__(self, service: HistoryPort, *, is_closed):
        self._service = service
        self._is_closed = is_closed

    def list(self) -> tuple[HistoryView, ...]:
        return self._service.history_list()

    def get(self, entry_id: str) -> HistoryView:
        return self._service.history_get(entry_id)

    def replayable(self, checkpoint_id: str) -> bool:
        return not self._is_closed() and self._service.history_replayable(checkpoint_id)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    replay_id: str | None
    status: str
    affected_agent_ids: tuple[int, ...]
    old_epochs: tuple[tuple[int, int], ...]
    new_epochs: tuple[tuple[int, int], ...]
    phase_latency_ms: tuple[tuple[str, float], ...] = ()
    failure_reason: str | None = None


class ReplayHandle:
    def __init__(self, execution):
        self.replay_id = execution.replay_id
        self._execution = execution

    async def wait(self) -> ReplayResult:
        execution = self._execution
        status = execution.status
        failure_reason = execution.failure_reason
        if execution.task is not None:
            try:
                await asyncio.shield(execution.task)
                status = "completed"
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                status = "replay_start_failed"
                failure_reason = "cancelled"
            except Exception as exc:
                status = "replay_start_failed"
                failure_reason = _failure_reason(exc)
        phases = execution.phase_latency_ms
        timing = execution.timing
        if timing is not None and timing.finished_at is not None:
            if timing.turn_started_at is not None:
                phases += (("replay_turn", (timing.finished_at - timing.turn_started_at) * 1000),)
            phases += (("complete_replay_task", (timing.finished_at - timing.started_at) * 1000),)
        return ReplayResult(
            execution.replay_id, status, execution.affected_agent_ids,
            execution.old_epochs, execution.new_epochs, phases,
            failure_reason,
        )


__all__ = ["HistoryClient", "ReplayHandle", "ReplayResult"]
