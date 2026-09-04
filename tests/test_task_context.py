"""Focused tests for immutable task identity and bounded projections."""

import dataclasses

import pytest

from opencollab.domain.context import TaskContext, TaskContextSection


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
