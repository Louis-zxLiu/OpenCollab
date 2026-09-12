"""Live Team handle and the small public rollback facade."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

from opencollab.application.scheduler_types import SchedulerTurnError
from opencollab.bootstrap.programmatic_team import close_team_runtime

from .history import HistoryClient, ReplayHandle
from .result import (
    CheckpointResult,
    RestoreResult,
    RollbackPlan,
    RollbackResult,
    RunResult,
)


def _plan_view(plan: Any) -> RollbackPlan:
    def identity_digest(checkpoint: Any) -> str | None:
        if checkpoint is None:
            return None
        return hashlib.sha256(checkpoint.workspace_identity.encode("utf-8")).hexdigest()

    return RollbackPlan(
        target_effect_ids=plan.target_effect_ids,
        invalidated_effect_ids=plan.invalidated_effect_ids,
        affected_agent_ids=plan.affected_agent_ids,
        checkpoint_by_agent={
            aid: checkpoint.checkpoint_id if checkpoint is not None else None
            for aid, checkpoint in plan.checkpoint_by_agent.items()
        },
        checkpoint_filesystem_digests={
            aid: checkpoint.filesystem_digest if checkpoint is not None else None
            for aid, checkpoint in plan.checkpoint_by_agent.items()
        },
        checkpoint_environment_digests={
            aid: checkpoint.environment.digest() if checkpoint is not None else None
            for aid, checkpoint in plan.checkpoint_by_agent.items()
        },
        checkpoint_identity_digests={
            aid: identity_digest(checkpoint)
            for aid, checkpoint in plan.checkpoint_by_agent.items()
        },
        digest=plan.digest(),
    )


def _restore_view(result: Any) -> RestoreResult:
    safe_reasons = {
        "checkpoint reference is unavailable", "checkpoint revision does not match its reference",
        "checkpoint filesystem digest does not match its revision", "workspace identity changed",
        "filesystem contents differ after restore", "environment differs after restore",
        "workspace identity changed during restore",
    }
    return RestoreResult(
        agent_id=result.agent_id,
        checkpoint_id=result.checkpoint_id,
        status=result.status,
        filesystem_digest=result.filesystem_digest,
        environment_digest=result.environment_digest,
        reason=result.reason if result.reason is None or result.reason in safe_reasons else "scope_restore_failed",
    )


def _rollback_view(result: Any) -> RollbackResult:
    return RollbackResult(
        plan=_plan_view(result.plan),
        restores=tuple(_restore_view(item) for item in result.restores),
        invalidated=result.invalidated,
    )


class RollbackClient:
    """Explicit rollback operations scoped to one live Team."""

    def __init__(self, scheduler: Any, continue_turn=None, *, ensure_open=None) -> None:
        self._scheduler = scheduler
        self._continue_turn = continue_turn
        self._executed = False
        self._ensure_open = ensure_open or (lambda: None)

    def preview(self, effect_ids: set[str]) -> RollbackPlan:
        self._ensure_open()
        return _plan_view(self._scheduler.preview_rollback(set(effect_ids)))

    async def create_checkpoint(self, aid: int, boundary: str = "initial") -> CheckpointResult:
        self._ensure_open()
        checkpoint = await self._scheduler.create_checkpoint(aid, boundary)
        return CheckpointResult(
            checkpoint_id=checkpoint.checkpoint_id,
            filesystem_digest=checkpoint.filesystem_digest,
            environment_digest=checkpoint.environment.digest(),
            workspace_identity_digest=hashlib.sha256(
                checkpoint.workspace_identity.encode("utf-8")
            ).hexdigest(),
        )

    async def execute(
        self,
        effect_ids: set[str],
        *,
        expected_plan_digest: str,
    ) -> RollbackResult:
        self._ensure_open()
        result = await self._scheduler.rollback_effect(
            set(effect_ids),
            expected_plan_digest=expected_plan_digest,
        )
        self._executed = True
        return _rollback_view(result)

    async def restore_checkpoint(self, aid: int, checkpoint_id: str) -> RestoreResult:
        self._ensure_open()
        result = await self._scheduler.rollback_to_checkpoint(aid, checkpoint_id)
        self._executed = True
        return _restore_view(result)

    def resume(self, agent_ids: set[int]) -> None:
        self._ensure_open()
        if not self._executed:
            raise RuntimeError("no rollback operation has completed")
        self._scheduler.resume_after_rollback(set(agent_ids))

    async def continue_turn(
        self,
        aid: int,
        message: str,
        *,
        budget_tokens: int | None = None,
    ) -> str:
        self._ensure_open()
        if self._continue_turn is not None:
            if budget_tokens is None:
                return await self._continue_turn(aid, message)
            return await self._continue_turn(aid, message, budget_tokens=budget_tokens)
        if budget_tokens is None:
            return await self._scheduler.continue_team_turn(aid, message)
        return await self._scheduler.continue_team_turn(
            aid, message, budget_tokens=budget_tokens
        )


class TeamHandle:
    """Own a live Team until the caller explicitly closes it."""

    def __init__(
        self,
        scheduler: Any,
        context: Any,
        run_task: asyncio.Task[str],
        *,
        cleanup_timeout: float,
        artifacts: Path | None,
    ) -> None:
        self._scheduler = scheduler
        self._context = context
        self._run_task = run_task
        self._cleanup_timeout = cleanup_timeout
        self._artifacts = artifacts
        self._closed = False
        self._closing = False
        self._close_lock = asyncio.Lock()
        self.history = HistoryClient(scheduler, is_closed=lambda: self._closed or self._closing)
        self.rollback = RollbackClient(scheduler, self._continue_turn, ensure_open=self._ensure_open)

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("TeamHandle is closed or closing")

    async def wait(self) -> RunResult[str]:
        """Wait for the current explicit Team turn without cleaning it."""
        try:
            output = await asyncio.shield(self._run_task)
        except SchedulerTurnError as exc:
            return RunResult(
                output=exc.partial_answer,
                status="stopped",
                reason=exc.terminal_reason or exc.phase.value,
                tokens=self._scheduler.used_tokens,
                artifacts=self._artifacts,
            )
        except asyncio.TimeoutError:
            return RunResult(
                output=None,
                status="stopped",
                reason="timeout",
                tokens=self._scheduler.used_tokens,
                artifacts=self._artifacts,
            )
        except asyncio.CancelledError:
            # Rollback is reported by SchedulerTurnError above. A bare task
            # cancellation is not evidence that a rollback took place.
            caller = asyncio.current_task()
            if (caller is not None and caller.cancelling()) or not self._run_task.cancelled():
                raise
            return RunResult(
                output=None,
                status="stopped",
                reason="cancelled",
                tokens=self._scheduler.used_tokens,
                artifacts=self._artifacts,
            )
        except Exception as exc:
            return RunResult(
                output=None,
                status="failed",
                reason=str(exc) or type(exc).__name__,
                tokens=self._scheduler.used_tokens,
                artifacts=self._artifacts,
                error=exc,
            )
        return RunResult(
            output=output,
            status="completed",
            tokens=self._scheduler.used_tokens,
            artifacts=self._artifacts,
        )

    async def replay_from(
        self, checkpoint_id: str, *, message: str, budget_tokens: int = 600_000,
    ) -> ReplayHandle:
        self._ensure_open()
        execution = await self._scheduler.replay_from(checkpoint_id, message, budget_tokens)
        if execution.task is not None:
            self._run_task = execution.task
        return ReplayHandle(execution)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            close_history = getattr(self._scheduler, "close_history", None)
            if close_history is not None:
                close_history()
            try:
                await close_team_runtime(
                    self._scheduler,
                    self._context,
                    self._cleanup_timeout,
                )
            except BaseException:
                # Cleanup is retryable, but a partly closed Team cannot run again.
                raise
            else:
                self._closed = True

    async def _continue_turn(
        self,
        aid: int,
        message: str,
        *,
        budget_tokens: int | None = None,
    ) -> str:
        self._ensure_open()
        if not self._run_task.done():
            raise RuntimeError("the current Team turn is still active")
        if budget_tokens is None:
            continuation = self._scheduler.continue_team_turn(aid, message)
        else:
            continuation = self._scheduler.continue_team_turn(
                aid, message, budget_tokens=budget_tokens
            )
        self._run_task = asyncio.create_task(continuation)
        return await self._run_task

    async def continue_turn(
        self,
        aid: int,
        message: str,
        *,
        budget_tokens: int | None = None,
    ) -> str:
        return await self._continue_turn(aid, message, budget_tokens=budget_tokens)


__all__ = ["RollbackClient", "TeamHandle"]
