"""Scheduler agent registration, spawning, driver finalization and pending-row delivery."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any

from opencollab.application.scheduler_types import LaunchSpec
from opencollab.domain.identity import validate_role_identity
from opencollab.domain.pending import PendingRowError, RowStatus
from opencollab.domain.scheduler import SessionControlBlock
from opencollab.domain.session import SessionPhase

logger = logging.getLogger(__name__)


class LifecycleMixin:
    """Spawn, drive, finalize, and wake agents in the delegation tree."""

    def register_lead(self, session: Any) -> int:
        """Register agent zero, preserving a restored phase for pending-row settlement."""
        session.agent.name = validate_role_identity(session.agent.name)
        aid = self.table.allocate_aid()  # = 0
        session.state.aid = aid
        scb = SessionControlBlock(
            aid=aid,
            parent_aid=None,
            agent=session.agent,
            state=session.state,
        )
        self.table.add(scb)
        self._sessions[aid] = session
        self._lead_session = session
        environment = getattr(session, "env", None)
        if environment is not None:
            self.register_effect_environment(aid, environment)
        self._restore_message_inbox(aid, session.state)
        # Book agent 0's own share of the pool so the first child is granted
        # from what is left after it.
        self._seed_entry_lease()
        self._write_manifest()
        return aid

    def create_init_process(self, launch: LaunchSpec) -> int:
        """Create and register agent 0 — the init process (aid=0)."""
        session = self._session_factory.create_lead_session(
            scheduler=self,
            launch=launch,
            budget=self._entry_start_budget(),
        )
        session.apply_launch(launch)
        return self.register_lead(session)

    async def spawn(
        self,
        parent_aid: int,
        role: str,
        task: str,
        context: str = "",
        tool_call_id: str | None = None,
    ) -> int:
        """Non-blocking spawn. Creates SCB, builds session, starts task. Returns aid."""
        if self._shutting_down:
            raise RuntimeError("Cannot spawn agent: scheduler is shutting down.")
        self._assert_agent_active(parent_aid)
        if self.table.get(parent_aid) is None or parent_aid not in self._sessions:
            raise ValueError(f"Cannot spawn agent: no parent with aid {parent_aid}.")
        role = validate_role_identity(role)
        # Before the topology check, so one uniform record covers both a request
        # for a declared role and one for a role this team was never given; the
        # record carries whether the topology would have allowed the edge.
        self._refuse_spawn_when_prebuilt(parent_aid, role, task, context)
        self._check_topology(parent_aid, role, verb="spawn")
        aid = self.table.allocate_aid()
        startup_task = asyncio.current_task()
        if startup_task is not None:
            self._startup_tasks[aid] = startup_task
        if tool_call_id is not None:
            self._startup_origin[aid] = (parent_aid, tool_call_id)
        env: Any | None = None
        parent_lease: tuple[int, int] | None = None
        # Reserve this (role, task) AND its budget synchronously — before the
        # first await — so a duplicate / batched spawn later in the same
        # tool-call batch already sees the updated allocation and cannot
        # oversubscribe the global pool.
        # Everything from reservation until the driver task is scheduled may raise
        # (worktree acquire, session build, event emission). The driver task owns
        # releasing the two reservations on termination — but it only exists once
        # ``create_task`` below succeeds. So if anything raises before that, the
        # reservations would leak permanently (the budget grant inflates the pool
        # for every future spawn, and the inflight key permanently refuses any
        # re-spawn of this (role, task)). Release both and re-raise so the caller
        # (execute_deferred) still surfaces the failure into the parent's row.
        try:
            # ``spawn`` is the authoritative single-flight boundary. This
            # compare-and-set runs before the first await, so concurrent
            # coroutines cannot both reserve the same delegated-work identity.
            self._reserve_inflight(aid, parent_aid, role, task, context)
            # ``spawn`` runs after the parent's current model generation has
            # completed. Return the parent's unused turn lease before granting
            # children; the parent will acquire a fresh lease when the complete
            # pending batch wakes it for its next generation.
            parent_driver = self._tasks.get(parent_aid)
            parent_scb = self.table.get(parent_aid)
            if (
                parent_driver is not None
                and not parent_driver.done()
                and parent_scb is not None
                and (
                    parent_driver is asyncio.current_task()
                    or parent_scb.state.phase is SessionPhase.EXECUTING_TOOLS
                )
            ):
                parent_lease = self._release_turn_lease(parent_aid)
                if parent_lease is not None:
                    self._track_review_parent_lease_release(parent_aid, 1)
            budget = self._reserve_child_budget(aid)
            if budget <= 0:
                raise RuntimeError(
                    "Cannot spawn agent: team token budget is fully allocated."
                )

            # Build environment
            parent_session = self._sessions.get(parent_aid)
            parent_environment = getattr(parent_session, "env", None)
            parent_snapshot = (
                parent_environment.snapshot_environment()
                if parent_environment is not None
                else None
            )
            env = await self._worktree_pool.acquire(role)
            self._assert_agent_active(parent_aid)
            if parent_snapshot is not None:
                env.replace_environment(parent_snapshot)
                env.bind_workspace(env.workspace)
            self._startup_envs[aid] = env
            if self._shutting_down:
                raise RuntimeError("Cannot spawn agent: scheduler is shutting down.")

            # Build session via factory. The task is seeded as the agent's first
            # user-context message (the TASK-layer ContextSource) inside the
            # factory, so the whole startup context is assembled in one place.
            session = self._session_factory.build_spawn_session(
                role=role,
                env=env,
                budget=budget,
                aid=aid,
                scheduler=self,
                task=task,
                context=context,
            )
            session.agent.name = role
            self.register_effect_environment(aid, env)

            # Create SCB
            scb = SessionControlBlock(
                aid=aid,
                parent_aid=parent_aid,
                agent=session.agent,
                state=session.state,
            )
            self.table.add(scb)
            self._sessions[aid] = session
            # A live checkpointed Team must snapshot dynamic children before
            # their first driver can mutate the newly inherited Scope.
            if parent_aid in self._rollback_checkpointed_scopes:
                await self.create_checkpoint(aid, "initial")
                self._assert_agent_active(parent_aid)
            if tool_call_id is not None:
                self._spawn_origin[aid] = (parent_aid, tool_call_id)
                self._startup_origin.pop(aid, None)

            # Emit spawn event
            await self.emit_scheduler_event(
                self._events.agent_spawned(aid, parent_aid, role, task)
            )
            if self._shutting_down:
                raise RuntimeError("Cannot spawn agent: scheduler is shutting down.")
            self._assert_agent_active(parent_aid)

            # Start async task. Once this succeeds, _drive_agent owns the
            # reservation release — must be the last statement that can hand off
            # ownership, so the except below never double-releases on success.
            if self._history_scope(parent_aid) and self._history_scope(aid):
                self._record_lifecycle_effect(
                    producer_aid=parent_aid, kind="spawn", content="", consumer_aid=aid,
                )
            self._start_agent_task(aid, session)
            self._startup_tasks.pop(aid, None)
            self._startup_envs.pop(aid, None)
            self._startup_origin.pop(aid, None)
        except asyncio.CancelledError:
            # Driver task was never scheduled, so nothing will release these.
            await self._rollback_failed_spawn(aid, env)
            if not self._shutting_down:
                self._restore_turn_lease(parent_aid, parent_lease)
                if parent_lease is not None:
                    self._track_review_parent_lease_release(parent_aid, -1)
            if tool_call_id is not None:
                await self._fail_cancelled_origin(
                    parent_aid,
                    tool_call_id,
                    "agent spawn cancelled before startup completed",
                )
            raise
        except BaseException:
            await self._rollback_failed_spawn(aid, env)
            if not self._shutting_down:
                self._restore_turn_lease(parent_aid, parent_lease)
                if parent_lease is not None:
                    self._track_review_parent_lease_release(parent_aid, -1)
            raise

        self._write_manifest()
        self._autosave_session(parent_aid)
        return aid

    async def _rollback_failed_spawn(self, aid: int, env: Any | None) -> None:
        """Undo every side effect created before a child driver task exists."""
        self._release_leases(aid)
        self.table.entries.pop(aid, None)
        self._sessions.pop(aid, None)
        self._spawn_origin.pop(aid, None)
        self._startup_tasks.pop(aid, None)
        self._startup_envs.pop(aid, None)
        self._startup_origin.pop(aid, None)
        self._tasks.pop(aid, None)
        self._locks.pop(aid, None)
        self._run_locks.pop(aid, None)
        self._message_inbox.pop(aid, None)
        self._rollback_checkpointed_scopes.discard(aid)

        if env is None:
            return
        try:
            await self._worktree_pool.release_env(env)
        except Exception as exc:
            logger.error("failed-spawn environment cleanup failed for aid %s: %s", aid, exc)

    async def _fail_cancelled_origin(
        self, parent_aid: int, tool_call_id: str, reason: str
    ) -> None:
        """Resolve a pre-driver cancellation when its pending row already exists."""
        parent = self.table.get(parent_aid)
        if parent is None or tool_call_id not in parent.state.pending_events.rows:
            return
        try:
            await self._wake(
                parent_aid,
                tool_call_id,
                f"Error: {reason}",
                RowStatus.FAILED,
            )
        except PendingRowError:
            logger.error(
                "cancelled spawn could not fail pending row %s on parent %s",
                tool_call_id,
                parent_aid,
            )

    def _release_leases(self, aid: int) -> None:
        """Release a terminal child's single-flight and budget leases."""
        self._clear_inflight(aid)
        self._release_turn_lease(aid)

    def _start_agent_task(self, aid: int, session: Any) -> asyncio.Task[None]:
        """Start and track one driver, reaping its references when it settles."""
        self._assert_agent_active(aid)
        token = self._rollback_turn_epoch.set((aid, self.current_effect_epoch(aid)))
        try:
            task = asyncio.create_task(self._drive_agent(aid, session))
        finally:
            self._rollback_turn_epoch.reset(token)
        self._tasks[aid] = task
        task.add_done_callback(
            lambda finished, owned_aid=aid: self._agent_task_done(
                owned_aid,
                finished,
            )
        )
        return task

    def _agent_task_done(self, aid: int, task: asyncio.Task[None]) -> None:
        """Consume a finished driver and release only its own registry entry."""
        if self._tasks.get(aid) is task:
            self._tasks.pop(aid, None)
        # Cleanup retains committed-delivery markers until its finalizer observes them.
        if not self._shutting_down:
            self._delivery_committed.discard(aid)
        scb = self.table.get(aid)
        if scb is not None and scb.state.phase.is_terminal():
            self._turn_started_at.pop(aid, None)
            if not self._shutting_down:
                self._write_manifest()
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("background task for aid %s was cancelled", aid)
        except Exception as exc:
            logger.error("background task for aid %s failed: %s", aid, exc)

    def _turn_gate(self) -> Any:
        """Optionally serialize drivers; deferred suspension releases the gate."""
        if not self._serialize_turns:
            return contextlib.nullcontext()
        if self._turn_gate_lock is None:
            self._turn_gate_lock = asyncio.Lock()
        return self._turn_gate_lock

    async def _drive_agent(self, aid: int, session: Any) -> None:
        """Drive once, retaining suspended turns or delivering a terminal child result."""
        scb = self.table.get(aid)
        if scb is None:
            return

        start = self._turn_started_at.setdefault(aid, time.monotonic())
        drive_epoch = self.current_effect_epoch(aid)

        try:
            cancel_event = self._turn_cancel_events.get(aid)
            epoch_token = self._rollback_turn_epoch.set(
                (aid, self.current_effect_epoch(aid))
            )
            try:
                async with self._turn_gate():
                    self._begin_history_turn(aid)
                    if self._history_scope(aid) and aid not in self._rollback_checkpointed_scopes:
                        await self.create_checkpoint(aid, "initial")
                    runner = getattr(session, "runner", None)
                    if runner is not None:
                        runner.tool_effect_transaction = lambda messages: self._tool_history_transaction(aid, messages)
                    result = (
                        await session.run_loop(cancel_event)
                        if cancel_event is not None
                        else await session.run_loop()
                    )
            finally:
                self._rollback_turn_epoch.reset(epoch_token)
        except asyncio.CancelledError:
            self._release_leases(aid)
            scb.state.cancel()
            reason = "Error: agent cancelled before completing delegated work"
            scb.result = reason
            try:
                await self.emit_scheduler_event(
                    self._events.agent_cancelled(aid, scb.agent.name)
                )
            except Exception as event_exc:
                logger.error(
                    "agent_cancelled event failed for aid %s: %s", aid, event_exc
                )
            await self._deliver_to_parent(aid, reason, RowStatus.FAILED, error=reason)
            if not self._shutting_down:
                await self._drain_message_inbox(aid, allow_current_task=True)
                await self._drain_ready_message_inboxes()
            raise
        except Exception as exc:
            self._release_leases(aid)
            terminal_reason = scb.state.terminal_reason or f"{type(exc).__name__}: {exc}"
            scb.state.fail(terminal_reason)
            reason = f"Error: {terminal_reason}"
            scb.result = reason
            await self._trace_worktree_evidence(aid, scb, session)
            try:
                await self.emit_scheduler_event(
                    self._events.agent_failed(aid, scb.agent.name, terminal_reason)
                )
            except Exception as event_exc:
                logger.error(
                    "agent_failed event failed for aid %s: %s", aid, event_exc
                )
            await self._deliver_to_parent(aid, reason, RowStatus.FAILED, error=reason)
            await self._drain_message_inbox(aid, allow_current_task=True)
            await self._drain_ready_message_inboxes()
            return

        # A cancellation-resistant provider/session can outlive the scheduler's
        # bounded teardown. Cleanup has already revoked its environment and
        # assigned a failure terminal; discard any late return before it can
        # overwrite that state or publish a successful completion.
        if self._shutting_down:
            self._finalize_cleanup_failure(aid)
            return

        try:
            self._assert_agent_active(aid, epoch=drive_epoch)
        except RuntimeError:
            logger.info("discarding stale or fenced driver result for aid %s", aid)
            return

        scb.result = result

        # Suspended on its own deferred work — not finished. A child's wake will
        # re-enter _drive_agent and finalize once it reaches a terminal phase.
        # The reservation stays held: the task is still genuinely in flight.
        if scb.state.phase is SessionPhase.AWAITING_EVENTS:
            return

        # Terminal — release the single-flight + budget reservations before
        # delivering, so a later spawn can reuse this child's unspent headroom.
        self._release_leases(aid)

        terminal_failure = self._terminal_failure_result(scb, result)
        if terminal_failure is not None:
            scb.result = terminal_failure
            await self._trace_worktree_evidence(aid, scb, session)
            await self._safe_emit_scheduler_event(
                self._events.agent_failed(aid, scb.agent.name, terminal_failure)
            )
            await self._deliver_to_parent(
                aid,
                terminal_failure,
                RowStatus.FAILED,
                error=terminal_failure,
            )
            await self._drain_message_inbox(aid, allow_current_task=True)
            if not self._shutting_down:
                await self._drain_ready_message_inboxes()
            return

        # A completed coding task still needs its patch evidence. Tracing and
        # event delivery are observational, but a missing diff is a technical
        # failure because the parent cannot verify what changed.
        env = getattr(session, "env", None)
        if env is not None:
            try:
                result = await self._append_worktree_diff(env, result)
            except Exception as exc:
                logger.error("worktree diff failed for aid %s: %s", aid, exc)
                scb.state.fail()
                result = f"Error: worktree diff extraction failed: {exc}"
                scb.result = result
                await self._safe_emit_scheduler_event(
                    self._events.agent_failed(aid, scb.agent.name, str(exc))
                )
                await self._deliver_to_parent(aid, result, RowStatus.FAILED, error=result)
                await self._drain_message_inbox(aid, allow_current_task=True)
                if not self._shutting_down:
                    await self._drain_ready_message_inboxes()
                return
            # Same changes, second destination: a structured, never-truncated
            # per-file record. Observational, so it is deliberately outside the
            # contract above — it may not fail the agent.
            await self._trace_worktree_changes(aid, scb.agent.name, env)

        if self._shutting_down:
            self._finalize_cleanup_failure(aid)
            return

        # Store the same post-diff artifact that is delivered to the parent so
        # spawn_with_review gives the reviewer the actual implementation diff.
        scb.result = result

        latency = time.monotonic() - start

        if self._tracer:
            try:
                self._tracer.log_step(
                    step_type="agent_completed",
                    payload={"aid": aid, "role": scb.agent.name, "result_len": len(result)},
                    tokens=session.used_tokens,
                    latency=latency,
                )
            except Exception as exc:
                logger.error("agent_completed trace failed for aid %s: %s", aid, exc)

        try:
            await self.emit_scheduler_event(
                self._events.agent_completed(
                    aid, scb.parent_aid, scb.agent.name, latency, len(result)
                )
            )
        except Exception as exc:
            logger.error("agent_completed event failed for aid %s: %s", aid, exc)

        if self._shutting_down:
            self._finalize_cleanup_failure(aid)
            return

        if not await self._complete_history_turn(aid, result):
            scb.state.fail("effect_registration_failed")
            return
        try:
            await self._deliver_to_parent(aid, result, RowStatus.DONE)
        except Exception:
            scb.state.fail("effect_registration_failed")
            origin = self._spawn_origin.get(aid)
            if origin is not None:
                parent_aid, _ = origin
                await self._recover_delivery_route(aid, parent_aid, PendingRowError("effect_registration_failed"))
            return
        await self._drain_message_inbox(aid, allow_current_task=True)
        if not self._shutting_down:
            await self._drain_ready_message_inboxes()

    async def _trace_worktree_evidence(self, aid: int, scb: Any, session: Any) -> None:
        """Record worktree evidence on every terminal path, including failure.

        The delegated tracer is observational and cannot change the outcome.
        """
        env = getattr(session, "env", None)
        if env is None:
            return
        await self._trace_worktree_changes(aid, scb.agent.name, env)

    @staticmethod
    def _terminal_failure_result(scb: Any, result: str) -> str | None:
        phase = scb.state.phase
        if phase not in {SessionPhase.ERROR, SessionPhase.STOPPED}:
            return None
        disposition = "failed" if phase is SessionPhase.ERROR else "stopped"
        reason = scb.state.terminal_reason or result.strip() or phase.value
        return f"Error: agent {disposition}: {reason}"

    async def _deliver_to_parent(
        self,
        child_aid: int,
        result: str,
        status: RowStatus,
        *,
        error: str | None = None,
    ) -> None:
        """Route a finished child's result to the pending row that suspended its."""
        origin = self._spawn_origin.get(child_aid)
        if origin is None:
            return
        parent_aid, tool_call_id = origin
        if child_aid in self._rollback_fenced or parent_aid in self._rollback_fenced:
            logger.info(
                "discarding late child result from aid %s to aid %s after rollback fence",
                child_aid,
                parent_aid,
            )
            return
        try:
            await self._wake(
                parent_aid,
                tool_call_id,
                result,
                status,
                child_aid=child_aid,
                error=error,
            )
        except PendingRowError as exc:
            retry_tool_call_id = await self._recover_delivery_route(
                child_aid,
                parent_aid,
                exc,
            )
            if retry_tool_call_id is not None:
                try:
                    await self._wake(
                        parent_aid,
                        retry_tool_call_id,
                        result,
                        status,
                        child_aid=child_aid,
                        error=error,
                    )
                    return
                except PendingRowError as retry_exc:
                    # The unique row disappeared or changed between discovery
                    # and fill. Count that as the bounded final route failure.
                    exc = retry_exc
                    await self._recover_delivery_route(child_aid, parent_aid, retry_exc)
            # A misrouted completion must surface loudly, never silently succeed.
            logger.error("misrouted completion from child %s: %s", child_aid, exc)
            await self._safe_emit_scheduler_event(
                self._events.agent_failed(parent_aid, self._role_of(parent_aid), str(exc))
            )

    async def _recover_delivery_route(
        self,
        child_aid: int,
        parent_aid: int,
        cause: PendingRowError,
    ) -> str | None:
        """Retry a unique child-ref row; otherwise explicitly fail the parent's batch."""
        parent_scb = self.table.get(parent_aid)
        parent_session = self._sessions.get(parent_aid)
        if parent_scb is None or parent_session is None:
            self._spawn_origin.pop(child_aid, None)
            return None

        lock = self._locks.setdefault(parent_aid, asyncio.Lock())
        should_resume = False
        async with lock:
            if parent_aid in self._rollback_fenced or child_aid in self._rollback_fenced:
                return None
            table = parent_scb.state.pending_events
            child_rows = [
                tool_call_id
                for tool_call_id, row in table.rows.items()
                if row.status is RowStatus.PENDING and row.ref == child_aid
            ]
            if len(child_rows) == 1:
                return child_rows[0]

            reason = f"Error: child completion routing failed: {cause}"
            # No unique target exists. This is a terminal parent-side routing
            # failure, not a child result for a different row: fail every open
            # row atomically so no producer-less PENDING row survives.
            for tool_call_id, row in tuple(table.rows.items()):
                if row.status is RowStatus.PENDING:
                    table.fill(
                        tool_call_id,
                        result=reason,
                        status=RowStatus.FAILED,
                        error=reason,
                    )
            self._spawn_origin.pop(child_aid, None)
            in_flight = self._tasks.get(parent_aid)
            should_resume = (
                not self._shutting_down
                and parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and table.is_complete()
                and (in_flight is None or in_flight.done())
            )
            if should_resume:
                self._reserve_turn_lease(parent_aid)
                self._start_agent_task(parent_aid, parent_session)
            elif (
                not self._shutting_down
                and parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and table.is_complete()
                and in_flight is not None
                and not in_flight.done()
            ):
                in_flight.add_done_callback(
                    lambda finished: asyncio.create_task(
                        self._resume_after_parent_task(
                            parent_aid,
                            parent_session,
                            finished,
                        )
                    )
                )

        if should_resume:
            await self._safe_emit_scheduler_event(
                self._events.agent_resumed(parent_aid, self._role_of(parent_aid))
            )
        return None

    async def _wake(
        self,
        parent_aid: int,
        tool_call_id: str,
        result: str,
        status: RowStatus,
        *,
        child_aid: int | None = None,
        error: str | None = None,
    ) -> None:
        """Fill the parent's pending row and, if that completes the batch while."""
        parent_scb = self.table.get(parent_aid)
        parent_session = self._sessions.get(parent_aid)
        if parent_scb is None or parent_session is None:
            if child_aid is not None:
                self._spawn_origin.pop(child_aid, None)
            return

        lock = self._locks.setdefault(parent_aid, asyncio.Lock())
        async with lock:
            if parent_aid in self._rollback_fenced or (
                child_aid is not None and child_aid in self._rollback_fenced
            ):
                if child_aid is not None:
                    self._spawn_origin.pop(child_aid, None)
                logger.info(
                    "discarding late delivery to aid %s after rollback fence",
                    parent_aid,
                )
                return
            table = parent_scb.state.pending_events
            cleanup_forced = False
            fill_error = error
            if child_aid is not None:
                child_scb = self.table.get(child_aid)
                cleanup_forced = self._shutting_down
                if cleanup_forced:
                    result = "Error: scheduler cleanup cancelled delegated work"
                    status = RowStatus.FAILED
                    fill_error = result
                    if child_scb is not None:
                        child_scb.state.cancel(result)
                        child_scb.result = result
            if child_aid is not None and not cleanup_forced:
                try:
                    self._assert_agent_active(child_aid)
                    self._assert_agent_active(parent_aid)
                except RuntimeError:
                    logger.info("discarding stale or unregistered child delivery to aid %s", parent_aid)
                    return
            previous_rows = table.rows.copy()
            with self._rollback_service.effect_transaction(), self._history.transaction():
                try:
                    table.fill(tool_call_id, result=result, status=status, error=fill_error)
                    if (status is RowStatus.DONE and child_aid is not None and not cleanup_forced
                            and self._has_effect_scopes(child_aid, parent_aid)):
                        self._record_child_effect(child_aid, parent_aid, result)
                except BaseException:
                    table.rows = previous_rows
                    raise
            if child_aid is not None and status is RowStatus.DONE and not cleanup_forced:
                await self._autosave_history(parent_aid, "child_result_accepted")
            if child_aid is not None:
                self._spawn_origin.pop(child_aid, None)
                if not cleanup_forced:
                    self._delivery_committed.add(child_aid)
            in_flight = self._tasks.get(parent_aid)
            should_resume = (
                not self._shutting_down
                and parent_aid not in self._rollback_fenced
                and
                parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and table.is_complete()
                and (in_flight is None or in_flight.done())
            )
            if should_resume:
                self._reserve_turn_lease(parent_aid)
                self._start_agent_task(parent_aid, parent_session)
            elif (
                not self._shutting_down
                and parent_aid not in self._rollback_fenced
                and
                parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and table.is_complete()
                and in_flight is not None
                and not in_flight.done()
            ):
                in_flight.add_done_callback(
                    lambda finished: asyncio.create_task(
                        self._resume_after_parent_task(
                            parent_aid,
                            parent_session,
                            finished,
                        )
                    )
                )

        if should_resume:
            await self._safe_emit_scheduler_event(
                self._events.agent_resumed(parent_aid, self._role_of(parent_aid))
            )

    async def _resume_after_parent_task(
        self,
        parent_aid: int,
        parent_session: Any,
        finished_task: asyncio.Task,
    ) -> None:
        """Close the tail race between AWAITING_EVENTS and driver completion."""
        parent_scb = self.table.get(parent_aid)
        if parent_scb is None:
            return
        lock = self._locks.setdefault(parent_aid, asyncio.Lock())
        should_resume = False
        async with lock:
            current = self._tasks.get(parent_aid)
            no_active_replacement = (
                current is None or current is finished_task or current.done()
            )
            if (
                not self._shutting_down
                and parent_aid not in self._rollback_fenced
                and
                no_active_replacement
                and parent_scb.state.phase is SessionPhase.AWAITING_EVENTS
                and parent_scb.state.pending_events.is_complete()
            ):
                self._reserve_turn_lease(parent_aid)
                self._start_agent_task(parent_aid, parent_session)
                should_resume = True
        if should_resume:
            try:
                await self.emit_scheduler_event(
                    self._events.agent_resumed(parent_aid, self._role_of(parent_aid))
                )
            except Exception as exc:
                logger.error("agent_resumed event failed for aid %s: %s", parent_aid, exc)
