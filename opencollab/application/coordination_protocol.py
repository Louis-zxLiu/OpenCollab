"""Application coordination protocol shared by Scheduler capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from opencollab.domain.coordination_protocol import (
    CoordinationWait,
    VerificationEvidence,
    WaitKind,
    is_addressable,
    verification_passes,
    wait_matches,
)
from opencollab.domain.identity import role_collision_key
from opencollab.domain.team import Topology


class CoordinationProtocol:
    """Own protocol state without knowing concrete Scheduler or adapters."""

    def __init__(self, topology: Topology | None = None) -> None:
        self._topology = topology
        self._states: dict[int, Any] = {}
        self._roles: dict[int, str] = {}
        self._parents: dict[int, int | None] = {}
        self._can_await: dict[int, bool] = {}
        self._adopt_callback: Callable[[int, str], Awaitable[str]] | None = None

    def bind_adopter(self, callback: Callable[[int, str], Awaitable[str]] | None) -> None:
        self._adopt_callback = callback

    def register_agent(
        self,
        aid: int,
        state: Any,
        role: str,
        parent_aid: int | None,
        *,
        can_await_coordination: bool = True,
    ) -> None:
        self._states[aid] = state
        self._roles[aid] = role
        self._parents[aid] = parent_aid
        self._can_await[aid] = bool(can_await_coordination)

    def unregister_agent(self, aid: int) -> None:
        self._states.pop(aid, None)
        self._roles.pop(aid, None)
        self._parents.pop(aid, None)
        self._can_await.pop(aid, None)

    def has_agent(self, aid: int) -> bool:
        """Return whether protocol state has been registered for ``aid``."""
        return aid in self._states

    def parent_of(self, aid: int) -> int | None:
        return self._parents.get(aid)

    def nearest_active_coordinator(self, aids: Iterable[int]) -> int | None:
        """Choose the shallowest active consumer able to coordinate recovery."""
        candidates = [
            aid for aid in aids
            if self.is_addressable(aid) and self._can_await.get(aid, True)
        ]
        if not candidates:
            return None

        def depth(aid: int) -> int:
            seen: set[int] = set()
            value = aid
            result = 0
            while value in self._parents and value not in seen:
                seen.add(value)
                parent = self._parents[value]
                if parent is None:
                    break
                result += 1
                value = parent
            return result

        return min(candidates, key=lambda aid: (depth(aid), aid))

    def validate_source_effects(self, aid: int, effect_ids: Iterable[str]) -> str | None:
        """Validate optional causal sources before a new coordination action."""
        self.require_addressable(aid)
        values = tuple(effect_ids)
        if any(not isinstance(effect_id, str) or not effect_id for effect_id in values):
            return "source_effect_ids must contain non-empty strings"
        pending = set(self._states[aid].required_effect_ids)
        blocked = sorted(set(values) & pending)
        if blocked:
            return f"source effects must be adopted first; pending effect_ids: {blocked}"
        return None

    def can_await_coordination(self, aid: int) -> bool:
        self.require_addressable(aid)
        return self._can_await.get(aid, True)

    def is_addressable(self, aid: int) -> bool:
        state = self._states.get(aid)
        return state is not None and is_addressable(state.lifecycle_status)

    def require_addressable(self, aid: int) -> None:
        state = self._states.get(aid)
        if state is None:
            raise ValueError(f"no agent with aid {aid}")
        if not is_addressable(state.lifecycle_status):
            raise ValueError(f"agent aid {aid} is {state.lifecycle_status.value} and is no longer addressable")

    def resolve_role(self, role: str) -> int:
        key = role_collision_key(role)
        candidates = [
            aid
            for aid, current in self._roles.items()
            if role_collision_key(current) == key and self.is_addressable(aid)
        ]
        if not candidates:
            raise ValueError(f"no active agent with role '{role}'")
        if len(candidates) > 1:
            raise ValueError(f"role '{role}' is ambiguous; address an active aid explicitly")
        return candidates[0]

    def register_visible_effect(self, consumer_aid: int, effect_id: str) -> None:
        self.require_addressable(consumer_aid)
        self._states[consumer_aid].register_visible_effect(effect_id)

    async def adopt_effect(self, consumer_aid: int, effect_id: str) -> str:
        self.require_addressable(consumer_aid)
        state = self._states[consumer_aid]
        if effect_id not in state.required_effect_ids:
            raise ValueError(f"effect {effect_id} is not visible to aid {consumer_aid}")
        if self._adopt_callback is None:
            state.adopt_visible_effect(effect_id)
            return f"Effect {effect_id} adopted."
        result = await self._adopt_callback(consumer_aid, effect_id)
        state.adopt_visible_effect(effect_id)
        return result

    def can_execute(self, aid: int, tool_name: str) -> bool:
        self.require_addressable(aid)
        if self._states[aid].can_execute_normal_work():
            return True
        return tool_name in {
            "adopt_effect",
            "await_coordination",
            "spawn_agent",
            "spawn_with_review",
            "message_agent",
            "team_status",
            "invalidate_effect",
            "ask_user",
        }

    def set_wait(self, aid: int, condition: CoordinationWait) -> None:
        self.require_addressable(aid)
        self._states[aid].wait_condition = condition

    def clear_wait_if_matches(
        self,
        aid: int,
        event_kind: WaitKind,
        effect_id: str | None = None,
    ) -> bool:
        state = self._states.get(aid)
        if state is None or not wait_matches(state.wait_condition, event_kind, effect_id):
            return False
        state.wait_condition = None
        return True

    def record_verification(self, aid: int, evidence: VerificationEvidence) -> None:
        self.require_addressable(aid)
        self._states[aid].verification_evidence[evidence.evidence_id] = evidence

    def validate_verification(
        self,
        aid: int,
        effect_id: str,
        evidence_ids: Iterable[str],
    ) -> bool:
        self.require_addressable(aid)
        state = self._states[aid]
        if effect_id in state.required_effect_ids:
            return False
        return verification_passes(state.verification_evidence, effect_id, evidence_ids)

    def supersede(self, aid: int) -> None:
        state = self._states.get(aid)
        if state is not None:
            state.mark_superseded()


__all__ = ["CoordinationProtocol"]
