"""Focused tests for immutable task identity and bounded projections."""

import dataclasses

import pytest

from opencollab.domain.context import TaskContext, TaskContextSection
from opencollab.domain.coordination import CoordinationPolicy


def test_task_context_is_immutable_and_projects_in_stable_order():
    context = TaskContext(
        context_id="task-1",
        objective="Solve the problem.",
        constraints="Use C++17.",
        contract="Write solution.cpp.",
    )

    assert context.project() == (
        "Objective:\nSolve the problem.\n\n"
        "Constraints:\nUse C++17.\n\n"
        "Contract:\nWrite solution.cpp."
    )
    assert context.project((TaskContextSection.CONTRACT,)) == "Contract:\nWrite solution.cpp."
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.objective = "changed"


@pytest.mark.parametrize("value", ["", "bad\x00value"])
def test_task_context_rejects_invalid_identity(value):
    with pytest.raises(ValueError):
        TaskContext(context_id=value, objective="Solve")


def test_task_context_rejects_oversized_sections():
    with pytest.raises(ValueError):
        TaskContext(context_id="task", objective="x" * (32 * 1024 + 1))


def test_coordination_policy_is_shared_and_configurable():
    policy = CoordinationPolicy(assignment_bytes=8, context_bytes=16, total_bytes=24)
    assert policy.validate("12345678", "") is None
    assert policy.validate("123456789", "") == "assignment exceeds the 8-byte limit"
    assert policy.validate("12345678", "12345678901234567") == "context exceeds the 16-byte limit"
    assert policy.validate("12345678", "1234567890123456") is None


def test_coordination_policy_allows_a_stricter_combined_limit_and_rejects_bad_text():
    policy = CoordinationPolicy(assignment_bytes=8, context_bytes=16, total_bytes=12)
    assert policy.validate("12345678", "1234") is None
    assert policy.validate("12345678", "12345") == "assignment and context exceed the 12-byte limit"
    assert policy.validate("12345678", "bad\x00text") == "context must not contain NUL bytes"
    assert policy.validate(1, "") == "assignment must be a string"
