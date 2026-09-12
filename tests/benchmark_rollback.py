"""Auditable graph and real LocalEnvironment checkpoint benchmarks on Linux."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollback_measurements import ResourceSampler, git_storage, provenance, stamp, summarize, write_json

from opencollab.adapters.env import LocalEnvironment
from opencollab.application.rollback import RollbackService
from opencollab.domain.rollback import EnvironmentSnapshot


def _measure(name, iterations, operation, *, warmups=1):
    """Retain failed warmups and samples; every observation must pass."""
    records = []
    for index in range(-warmups, iterations):
        started = time.perf_counter()
        record = {"sample": index + 1, "warmup": index < 0, "started_utc": stamp()}
        try:
            value = operation()
            result_status = getattr(value, "status", None)
            if result_status is not None and result_status != "restored":
                raise RuntimeError(getattr(value, "reason", None) or f"restore status: {result_status}")
            record["status"] = "passed"
        except Exception as exc:
            record.update(status="failed", error_reason=str(exc))
        record.update(elapsed_ms=(time.perf_counter() - started) * 1000, ended_utc=stamp())
        records.append(record)
    failures = [r for r in records if r["status"] != "passed"]
    stats = summarize([r["elapsed_ms"] for r in records if not r["warmup"] and r["status"] == "passed"])
    return {
        "name": name,
        "status": "failed" if failures else "passed",
        "samples": iterations,
        "error_reason": failures[0]["error_reason"] if failures else None,
        "median_ms": stats.get("median"),
        "p95_ms": stats.get("p95"),
        "raw_samples": records,
    }


def _spec(size, shape, agents=4):
    """Topological edges; the extra last node is always an independent sibling."""
    rows = []
    for index in range(size):
        if shape == "chain":
            parents, aid = (() if not index else (index - 1,)), 0
        elif shape == "balanced":
            parents = () if not index else ((index - 1) // 2,)
            branch = index
            while branch > 2:
                branch = (branch - 1) // 2
            aid = branch
        elif shape == "fanout":
            parents, aid = (() if not index else (0,)), index
        elif shape in {"diamond", "multi-agent"}:
            width = max(1, agents - 3) if shape == "multi-agent" else 4
            layer = (index - 1) // width
            if index == 0:
                parents = ()
            elif layer == 0:
                parents = (0,)
            else:
                parents = tuple(range(1 + (layer - 1) * width, 1 + layer * width))
            aid = 0 if index == 0 else 1 + (index - 1) % width
        else:
            raise ValueError(f"unknown shape: {shape}")
        rows.append((index, aid, parents))
    sibling_aid = max(aid for _, aid, _ in rows) + 1
    rows.append((size, sibling_aid, ()))
    target = 1 if shape == "balanced" and size > 1 else size // 2 if shape == "chain" else 0
    expected = {target}
    for index, _, parents in rows:
        if expected.intersection(parents):
            expected.add(index)
    return rows, target, expected, sibling_aid


def _build_graph(size, shape="chain", agents=4):
    service = RollbackService()
    rows, target, expected, sibling = _spec(size, shape, agents)
    for index, aid, parents in rows:
        service.create_effect(
            effect_id=f"e{index}",
            producer_aid=aid,
            kind="tool_result",
            epoch=0,
            attempt=0,
            parent_effect_ids=tuple(f"e{p}" for p in parents),
            content=str(index),
        )
        for parent in parents:
            service.register_consumer(f"e{parent}", aid)
    if shape == "multi-agent":
        service.register_consumer(f"e{size - 1}", sibling + 1)
    plan = service.preview_rollback({f"e{target}"})
    assert plan.invalidated_effect_ids == frozenset(f"e{i}" for i in expected)
    assert sibling not in plan.affected_agent_ids and f"e{size}" not in plan.invalidated_effect_ids
    return service, f"e{target}"


def _git(workspace, *args):
    subprocess.run(("git", *args), cwd=workspace, capture_output=True, check=True)


async def _scenario(args, sampler):
    source, started = provenance(), stamp()
    results, storage_records = [], []
    with sampler.measure("graph_build"):
        service, target = _build_graph(args.graph_size, args.graph_shape, args.agents)
    plan = service.preview_rollback({target})
    for name, operation in (
        ("graph_preview", lambda: service.preview_rollback({target})),
        ("plan_digest", plan.digest),
        ("graph_construction", lambda: _build_graph(args.graph_size, args.graph_shape, args.agents)),
    ):
        with sampler.measure(name):
            results.append(_measure(name, args.iterations, operation, warmups=args.warmups))
    graph_metadata = len(json.dumps([asdict(e) for e in service.effects.values()]).encode())
    with tempfile.TemporaryDirectory(prefix="rollback-measured-") as directory:
        workspace = Path(directory)
        _git(workspace, "init", "-q")
        _git(workspace, "config", "user.name", "Experiment")
        _git(workspace, "config", "user.email", "experiment@example.invalid")
        for i in range(args.file_count):
            (workspace / f"file-{i}").write_text(hashlib.shake_256(str(i).encode()).hexdigest(2048))
        _git(workspace, "add", ".")
        _git(workspace, "commit", "-qm", "baseline")
        env = LocalEnvironment(str(workspace))
        snapshot = EnvironmentSnapshot.from_mapping(
            {"PWD": str(workspace), **{f"BENCH_{i}": str(i) for i in range(args.environment_count)}}
        )
        env.replace_environment(snapshot)
        baseline = git_storage(workspace)
        checkpoints = []
        try:
            for index in range(args.checkpoint_count):
                await env.write_file(f"file-{index % args.file_count}", f"retained checkpoint {index}\n")
                with sampler.measure("retained_checkpoint_creation") as record:
                    cp = await env.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
                checkpoints.append(cp)
                metadata_bytes = sum(
                    len(
                        json.dumps(
                            {
                                "id": c.checkpoint_id,
                                "revision": c.filesystem_revision,
                                "tree": c.filesystem_digest,
                                "environment_digest": c.environment.digest(),
                                "frontier": sorted(c.causal_frontier),
                                "identity_digest": hashlib.sha256(c.workspace_identity.encode()).hexdigest(),
                            }
                        ).encode()
                    )
                    for c in checkpoints
                )
                storage_records.append(
                    {
                        "checkpoint_count": index + 1,
                        "checkpoint_creation_ms": record["elapsed_ms"],
                        "metadata_export_bytes": metadata_bytes,
                        **git_storage(workspace),
                    }
                )
            checkpoint, records = checkpoints[-1], []
            for index in range(-args.warmups, args.iterations):
                entry = {"sample": index + 1, "warmup": index < 0, "started_utc": stamp()}
                try:
                    await env.write_file("file-0", f"mutation {index}\n")
                    env.set_environment_variable("BENCH_MUTATION", str(index))
                    with sampler.measure("adapter_restore") as phase:
                        restored = await env.restore_scope(checkpoint)
                    assert restored.status == "restored", restored.reason
                    assert restored.filesystem_digest == checkpoint.filesystem_digest
                    assert restored.environment_digest == snapshot.digest()
                    assert env.snapshot_environment() == snapshot
                    entry.update(
                        status="passed",
                        elapsed_ms=phase["elapsed_ms"],
                        filesystem_digest=restored.filesystem_digest,
                        environment_digest=restored.environment_digest,
                        pwd_verified=env.environment_view()["PWD"] == env.workspace,
                    )
                except Exception as exc:
                    entry.update(status="failed", error_reason=str(exc))
                entry["ended_utc"] = stamp()
                records.append(entry)
            failures = [r for r in records if r["status"] != "passed"]
            summary = summarize([r["elapsed_ms"] for r in records if r["status"] == "passed" and not r["warmup"]])
            results.append(
                {
                    "name": "adapter_restore",
                    "status": "failed" if failures else "passed",
                    "samples": args.iterations,
                    "median_ms": summary.get("median"),
                    "p95_ms": summary.get("p95"),
                    "raw_samples": records,
                }
            )

            def restore_environment():
                env.set_environment_variable("BENCH_MUTATION", "different")
                env.replace_environment(snapshot)
                assert env.snapshot_environment().digest() == snapshot.digest()

            with sampler.measure("environment_restore"):
                results.append(
                    _measure("environment_restore", args.iterations, restore_environment, warmups=args.warmups)
                )
        finally:
            with sampler.measure("cleanup"):
                await env.cleanup()
        after_cleanup = git_storage(workspace)
        assert after_cleanup["checkpoint_refs_count"] == 0
    return {
        **source,
        "schema": 2,
        "started_utc": started,
        "ended_utc": stamp(),
        "inputs": {key: value for key, value in vars(args).items() if key != "output"},
        "status": "passed" if all(r["status"] == "passed" for r in results) else "failed",
        "scope": "Graph planning plus single LocalEnvironment adapter restore. No Scheduler or LLM.",
        "cache_condition": "New disposable Git repository; warm page cache. OS cold cache not measured.",
        "graph_object_count": len(service.effects),
        "graph_metadata_export_bytes": graph_metadata,
        "affected_agents": sorted(plan.affected_agent_ids),
        "results": results,
        "storage_baseline": baseline,
        "storage_trajectory": storage_records,
        "storage_after_cleanup": after_cleanup,
        "comparison": "Graph construction is not a task rerun. No end-to-end speedup is inferred.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--graph-size", type=int, default=128)
    parser.add_argument(
        "--graph-shape", default="chain", choices=("chain", "balanced", "fanout", "diamond", "multi-agent")
    )
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--file-count", type=int, default=32)
    parser.add_argument("--environment-count", type=int, default=64)
    parser.add_argument("--checkpoint-count", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.warmups,
            args.graph_size,
            args.file_count,
            args.environment_count,
            args.checkpoint_count,
        )
        < 1
    ):
        parser.error("sample counts and workload sizes must be positive")
    sampler = ResourceSampler().start()
    try:
        payload = asyncio.run(_scenario(args, sampler))
    except Exception as exc:
        payload = {**provenance(), "status": "failed", "error_reason": f"{type(exc).__name__}: {exc}"}
    payload["resources"] = sampler.stop()
    if args.output:
        write_json(args.output, payload)
    print(json.dumps({key: value for key, value in payload.items() if key in {"status", "error_reason", "inputs"}}))
    if not args.output:
        print(json.dumps(payload))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
