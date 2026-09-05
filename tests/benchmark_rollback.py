"""Measure explicit rollback primitives without exposing environment values."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from opencollab.adapters.env import LocalEnvironment
from opencollab.application.rollback import RollbackService
from opencollab.domain.rollback import EnvironmentSnapshot


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile) + 0.999999) - 1))
    return ordered[index]


def _measure(name: str, iterations: int, operation) -> dict[str, object]:
    samples: list[float] = []
    digest = ""
    for _ in range(iterations):
        started = time.perf_counter()
        digest = str(operation())
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "name": name,
        "status": "passed",
        "samples": iterations,
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "digest": digest,
    }


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=cwd, check=True, capture_output=True)


def _build_graph(size: int) -> RollbackService:
    service = RollbackService()
    previous = None
    for aid in range(size):
        parents = (previous,) if previous is not None else ()
        effect = service.create_effect(
            producer_aid=aid,
            kind="tool_result",
            epoch=0,
            attempt=0,
            parent_effect_ids=parents,
            content=str(aid),
        )
        if previous is not None:
            service.register_consumer(previous, aid)
        previous = effect.effect_id
    return service


def _run(iterations: int, graph_size: int, file_count: int, env_count: int) -> dict[str, object]:
    service = _build_graph(graph_size)
    target = next(iter(service.effects))
    plan = service.preview_rollback({target})
    results = [
        _measure(
            "graph_plan",
            iterations,
            lambda: len(service.preview_rollback({target}).invalidated_effect_ids),
        )
    ]

    environment_values = {f"OC_BENCH_{index}": str(index) for index in range(env_count)}
    environment = EnvironmentSnapshot.from_mapping(environment_values)
    scope_state = {"snapshot": EnvironmentSnapshot.from_mapping({"OC_BENCH_MUTATED": "1"})}

    def restore_environment() -> str:
        scope_state["snapshot"] = environment
        return scope_state["snapshot"].digest()

    results.append(
        _measure(
            "environment_restore",
            iterations,
            restore_environment,
        )
    )

    with tempfile.TemporaryDirectory(prefix="opencollab-rollback-benchmark-") as raw:
        workspace = Path(raw)
        _git(workspace, "init", "-q")
        _git(workspace, "config", "user.name", "OpenCollab Benchmark")
        _git(workspace, "config", "user.email", "benchmark@opencollab.invalid")
        for index in range(file_count):
            (workspace / f"tracked-{index}.txt").write_text("baseline\n", encoding="utf-8")
        _git(workspace, "add", ".")
        _git(workspace, "commit", "-qm", "benchmark baseline")
        environment_adapter = LocalEnvironment(str(workspace))
        checkpoint = awaitable_checkpoint(environment_adapter)
        results.append(
            _measure(
                "filesystem_restore",
                iterations,
                lambda: awaitable_restore(environment_adapter, checkpoint),
            )
        )
        awaitable_cleanup(environment_adapter)

    return {
        "schema": 1,
        "platform": os.name,
        "inputs": {
            "graph_size": graph_size,
            "file_count": file_count,
            "environment_count": env_count,
            "iterations": iterations,
        },
        "plan_affected_agents": len(plan.affected_agent_ids),
        "results": results,
    }


def awaitable_checkpoint(environment):
    import asyncio

    return asyncio.run(
        environment.checkpoint_scope(
            "initial",
            owner_aid=0,
            causal_frontier=frozenset(),
        )
    )


def awaitable_restore(environment, checkpoint):
    import asyncio

    return asyncio.run(asyncio_restore(environment, checkpoint))


async def asyncio_restore(environment, checkpoint):
    result = await environment.restore_scope(checkpoint)
    return result.filesystem_digest or result.status


def awaitable_cleanup(environment):
    import asyncio

    asyncio.run(environment.cleanup())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--graph-size", type=int, default=128)
    parser.add_argument("--file-count", type=int, default=32)
    parser.add_argument("--environment-count", type=int, default=64)
    args = parser.parse_args()
    if min(args.iterations, args.graph_size, args.file_count, args.environment_count) <= 0:
        parser.error("all benchmark sizes must be positive")
    print(json.dumps(_run(args.iterations, args.graph_size, args.file_count, args.environment_count), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
