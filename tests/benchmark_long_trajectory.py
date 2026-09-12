"""Measure retained snapshots and equal-endpoint deterministic replay on Linux.

Uses the real RollbackService and LocalEnvironment, not a simulated restore.
No LLM, live Scheduler task, artificial sleep, or external service is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from rollback_measurements import ResourceSampler, provenance

from opencollab.adapters.env import LocalEnvironment
from opencollab.application.rollback import RollbackService


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def git(workspace, *args):
    return subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True, text=True).stdout.strip()


def disk_usage(directory):
    files = [path for path in directory.rglob("*") if path.is_file()]
    return {
        "bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes": sum(path.stat().st_blocks * 512 for path in files),
        "files": len(files),
    }


def tree_bytes(workspace, revision):
    data = subprocess.run(
        ["git", "ls-tree", "-r", "-l", "-z", revision],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    return sum(int(entry.split(b"\t", 1)[0].split()[3]) for entry in data.split(b"\0") if entry)


def summary(values):
    if not values:
        return {"samples": 0, "median": None, "p95": None}
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
    }


def payload(label, size):
    return hashlib.shake_256(label.encode()).hexdigest((size + 1) // 2)[: size - 1] + "\n"


def metadata_bytes(checkpoint):
    # Count a hypothetical canonical export in memory; never write Scope values.
    metadata = {
        "checkpoint_id": checkpoint.checkpoint_id,
        "owner_aid": checkpoint.owner_aid,
        "sequence": checkpoint.sequence,
        "filesystem_revision": checkpoint.filesystem_revision,
        "environment": checkpoint.environment.entries,
        "causal_frontier": sorted(checkpoint.causal_frontier),
        "boundary": checkpoint.boundary,
        "workspace_identity": checkpoint.workspace_identity,
        "filesystem_digest": checkpoint.filesystem_digest,
    }
    return len(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())


def setup(workspace, files, baseline_bytes):
    git(workspace, "init", "-q")
    git(workspace, "config", "user.name", "OpenCollab Benchmark")
    git(workspace, "config", "user.email", "benchmark@opencollab.invalid")
    git(workspace, "config", "gc.auto", "0")
    (workspace / ".gitignore").write_text(".opencollab/\n", encoding="utf-8")
    (workspace / ".opencollab").mkdir()
    (workspace / ".opencollab" / "retained").write_text("control evidence\n", encoding="utf-8")
    for index in range(files):
        (workspace / f"file-{index:03d}.txt").write_text(payload(f"base-{index}", baseline_bytes))
    git(workspace, "add", ".")
    git(workspace, "commit", "-qm", "benchmark baseline")


def new_service(workspace, variables):
    env = LocalEnvironment(str(workspace))
    for index in range(variables):
        env.set_environment_variable(f"OC_BENCH_{index:03d}", f"synthetic-{index}")
    service = RollbackService()
    service.register_environment(0, env)
    return env, service


def workspace_digest(workspace, files):
    digest = hashlib.sha256()
    for index in range(files):
        digest.update((workspace / f"file-{index:03d}.txt").read_bytes())
    return digest.hexdigest()


async def replay(env, service, args, start, stop, *, previous=None, epoch=0, checkpoints=None, evidence=None):
    ids = []
    checkpoint_ms = 0.0
    tick = time.perf_counter()
    for step in range(start, stop):
        name = f"file-{step % args.files:03d}.txt"
        # Real bounded adapter I/O is identical on the original and replay paths.
        content = await env.read_file(name)
        await env.write_file(name, content + payload(f"step-{step}", args.step_bytes))
        effect = service.create_effect(
            producer_aid=0,
            kind="tool_result",
            epoch=epoch,
            attempt=step,
            parent_effect_ids=(previous,) if previous else (),
            content=f"step-{step}",
        )
        previous = effect.effect_id
        ids.append(previous)
        if evidence is not None:
            evidence.write(json.dumps({"step": step, "effect_id": previous, "digest": effect.content_digest}) + "\n")
        if checkpoints is not None and ((step + 1) % args.stride == 0 or step + 1 in (stop // 2, stop)):
            cp_tick = time.perf_counter()
            cp = await service.create_checkpoint(0, "effect", frozenset({previous}))
            elapsed = (time.perf_counter() - cp_tick) * 1000
            checkpoint_ms += elapsed
            checkpoints.append((step + 1, cp, elapsed))
    return ids, {"wall_ms": (time.perf_counter() - tick) * 1000, "checkpoint_ms": checkpoint_ms}


async def one_sample(args, root, length, number, sampler):
    row = {"sample": number, "started_utc": utc(), "status": "failed", "stages": {}}
    tick = time.perf_counter()
    owned = Path(tempfile.mkdtemp(prefix=f"n{length}-s{number}-", dir=root))
    workspace, fresh = owned / "original", owned / "fresh"
    workspace.mkdir()
    fresh.mkdir()
    env = fresh_env = None
    host_before = dict(os.environ)
    try:
        setup(workspace, args.files, args.baseline_bytes)
        setup(fresh, args.files, args.baseline_bytes)
        env, service = new_service(workspace, args.variables)
        fresh_env, fresh_service = new_service(fresh, args.variables)
        objects = workspace / ".git" / "objects"
        before = disk_usage(objects)
        cp_tick = time.perf_counter()
        initial = await service.create_checkpoint(0)
        initial_ms = (time.perf_counter() - cp_tick) * 1000
        checkpoints = [(0, initial, initial_ms)]
        with (owned / "trajectory.jsonl").open("w", encoding="utf-8") as evidence:
            with sampler.measure("original_with_checkpoints"):
                ids, original = await replay(env, service, args, 0, length, checkpoints=checkpoints, evidence=evidence)
        row["stages"]["original_with_checkpoints_ms"] = original["wall_ms"] + initial_ms
        row["stages"]["checkpoint_creation_total_ms"] = original["checkpoint_ms"] + initial_ms
        checkpoint = next(cp for step, cp, _ in checkpoints if step == length // 2)
        final_digest = workspace_digest(workspace, args.files)
        after = disk_usage(objects)
        logical_sizes = [tree_bytes(workspace, cp.filesystem_revision) for _, cp, _ in checkpoints]
        refs = disk_usage(workspace / ".git" / "refs" / "opencollab")
        object_delta = after["bytes"] - before["bytes"]
        row["storage"] = {
            "checkpoint_count": len(checkpoints),
            "object_bytes_delta": object_delta,
            "baseline_object_bytes": before["bytes"],
            "objects_after_snapshots_bytes": after["bytes"],
            "object_allocated_bytes_delta": after["allocated_bytes"] - before["allocated_bytes"],
            "new_object_count": after["files"] - before["files"],
            "ref_bytes": refs["bytes"],
            "ref_allocated_bytes": refs["allocated_bytes"],
            "snapshot_payload_bytes": object_delta + refs["bytes"],
            "snapshot_allocated_bytes": after["allocated_bytes"] - before["allocated_bytes"] + refs["allocated_bytes"],
            "latest_logical_tree_bytes": logical_sizes[-1],
            "all_logical_trees_bytes": sum(logical_sizes),
            "bytes_per_effect": (object_delta + refs["bytes"]) / length,
            "bytes_per_checkpoint": (object_delta + refs["bytes"]) / len(checkpoints),
            "trajectory_jsonl_bytes": (owned / "trajectory.jsonl").stat().st_size,
            "estimated_serialized_metadata_bytes": sum(metadata_bytes(cp) for _, cp, _ in checkpoints),
            "checkpoints": [
                {
                    "step": step,
                    "checkpoint_id": cp.checkpoint_id,
                    "revision": cp.filesystem_revision,
                    "tree_digest": cp.filesystem_digest,
                    "logical_tree_bytes": size,
                    "creation_ms": elapsed,
                    "commit_object_bytes": int(git(workspace, "cat-file", "-s", cp.filesystem_revision)),
                }
                for (step, cp, elapsed), size in zip(checkpoints, logical_sizes)
            ],
        }
        original_restore = env.restore_scope

        async def timed_restore(cp):
            restore_tick = time.perf_counter()
            try:
                with sampler.measure("filesystem_environment_restore"):
                    return await original_restore(cp)
            finally:
                row["stages"]["restore_including_verification_ms"] = (time.perf_counter() - restore_tick) * 1000

        env.restore_scope = timed_restore

        async def recover_and_replay():
            preview_tick = time.perf_counter()
            target = ids[length // 2]
            plan = service.preview_rollback({target})
            digest = plan.digest()
            row["stages"]["preview_ms"] = (time.perf_counter() - preview_tick) * 1000
            assert plan.checkpoint_by_agent[0] == checkpoint, "wrong selected checkpoint"
            rollback_tick = time.perf_counter()
            with sampler.measure("rollback_execute"):
                result = await service.rollback_effect({target}, expected_plan_digest=digest)
            row["stages"]["execute_ms"] = (time.perf_counter() - rollback_tick) * 1000
            assert result.invalidated and len(plan.invalidated_effect_ids) == length // 2
            restore = result.restores[0]
            assert restore.status == "restored"
            audit_tick = time.perf_counter()
            assert restore.filesystem_digest == checkpoint.filesystem_digest
            assert restore.environment_digest == checkpoint.environment.digest() == env.snapshot_environment().digest()
            assert os.path.realpath(env.workspace) == checkpoint.workspace_identity
            assert env.snapshot_environment().as_dict()["PWD"] == checkpoint.workspace_identity
            expected = hashlib.sha256()
            for index in range(args.files):
                content = payload(f"base-{index}", args.baseline_bytes)
                content += "".join(
                    payload(f"step-{step}", args.step_bytes)
                    for step in range(length // 2)
                    if step % args.files == index
                )
                expected.update(content.encode())
            assert workspace_digest(workspace, args.files) == expected.hexdigest()
            assert (workspace / ".opencollab" / "retained").read_text() == "control evidence\n"
            row["stages"]["independent_restore_audit_ms"] = (time.perf_counter() - audit_tick) * 1000
            with sampler.measure("suffix_retry"):
                _, retry = await replay(env, service, args, length // 2, length, previous=ids[length // 2 - 1], epoch=1)
            row["stages"]["suffix_retry_ms"] = retry["wall_ms"]
            row["verification"] = {
                "filesystem_match": True,
                "environment_match": True,
                "pwd_match": True,
                "identity_match": True,
                "control_retained": True,
                "same_final_output": workspace_digest(workspace, args.files) == final_digest,
                "host_environment_unchanged": dict(os.environ) == host_before,
            }
            row["rollback"] = {
                "plan_digest": digest,
                "checkpoint_id": checkpoint.checkpoint_id,
                "affected_agents": sorted(plan.affected_agent_ids),
                "invalidated_effects": len(plan.invalidated_effect_ids),
                "filesystem_digest": restore.filesystem_digest,
                "environment_digest": restore.environment_digest,
            }

        async def rerun():
            with sampler.measure("fresh_rerun"):
                _, measured = await replay(fresh_env, fresh_service, args, 0, length)
            row["stages"]["fresh_rerun_ms"] = measured["wall_ms"]
            assert workspace_digest(fresh, args.files) == final_digest, "fresh rerun endpoint differs"

        # Alternate paired execution order; initialization and evidence I/O are excluded from both timed replays.
        row["paired_order"] = "rerun_first" if number % 2 else "rollback_first"
        if number % 2:
            await rerun()
            await recover_and_replay()
        else:
            await recover_and_replay()
            await rerun()
        assert all(row["verification"].values())
        stages = row["stages"]
        stages["rollback_plus_retry_ms"] = stages["preview_ms"] + stages["execute_ms"] + stages["suffix_retry_ms"]
        stages["checkpoint_cost_plus_recovery_ms"] = (
            stages["checkpoint_creation_total_ms"] + stages["rollback_plus_retry_ms"]
        )
        row["comparison"] = {
            "execute_to_rerun_ratio": stages["execute_ms"] / stages["fresh_rerun_ms"],
            "equal_endpoint_speedup": stages["fresh_rerun_ms"] / stages["rollback_plus_retry_ms"],
            "saved_ms": stages["fresh_rerun_ms"] - stages["rollback_plus_retry_ms"],
        }
        row["final_output_digest"] = final_digest
        row["status"] = "passed"
    except Exception as exc:
        # Preserve type and phase without leaking infrastructure paths or environment values.
        row["error_reason"] = type(exc).__name__
        row["error_detail"] = (
            str(exc)[:240] if isinstance(exc, AssertionError) else "see exception type; no raw payload exported"
        )
    finally:
        cleanup_tick = time.perf_counter()
        try:
            if env is not None:
                await env.cleanup()
                assert git(workspace, "for-each-ref", "refs/opencollab/checkpoints") == ""
                row["object_bytes_after_ref_discard"] = disk_usage(workspace / ".git" / "objects")["bytes"]
            if fresh_env is not None:
                await fresh_env.cleanup()
            shutil.rmtree(owned)
            assert not owned.exists()
            row["cleanup"] = "passed"
        except Exception as exc:
            row["cleanup"] = type(exc).__name__
            row["status"] = "failed"
        row["cleanup_ms"] = (time.perf_counter() - cleanup_tick) * 1000
    row["finished_utc"] = utc()
    row["wall_ms"] = (time.perf_counter() - tick) * 1000
    return row


async def main(args):
    if platform.system() != "Linux":
        raise RuntimeError("Acceptance benchmark must run on Linux")
    sampler = ResourceSampler().start()
    root = Path(tempfile.mkdtemp(prefix="opencollab-long-", dir=args.root))
    report = {
        **provenance(),
        "schema": 2,
        "started_utc": utc(),
        "status": "running",
        "source_commit": git(Path.cwd(), "rev-parse", "HEAD"),
        "source_tree": git(Path.cwd(), "rev-parse", "HEAD^{tree}"),
        "source_dirty": bool(git(Path.cwd(), "status", "--porcelain")),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": {
            "system": platform.system(),
            "python": platform.python_version(),
            "git": git(Path.cwd(), "--version"),
        },
        "inputs": {
            key: getattr(args, key)
            for key in ("lengths", "files", "variables", "baseline_bytes", "step_bytes", "stride", "samples")
        },
        "scope": "LocalEnvironment + RollbackService; one idle Agent; deterministic replay; no LLM or Docker",
        "owned_root_name": root.name,
        "scenarios": [],
    }
    tick = time.perf_counter()
    output = Path(args.output)

    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    try:
        for length in args.lengths:
            scenario = {"length": length, "started_utc": utc(), "samples": [], "status": "running"}
            report["scenarios"].append(scenario)
            for number in range(args.samples + 1):
                row = await one_sample(args, root, length, number, sampler)
                if number == 0:
                    scenario["excluded_warmup"] = row
                else:
                    scenario["samples"].append(row)
                print(
                    json.dumps(
                        {"length": length, "sample": number, "status": row["status"], "error": row.get("error_reason")}
                    ),
                    flush=True,
                )
                save()
            passed = [row for row in scenario["samples"] if row["status"] == "passed"]
            scenario["passed"] = len(passed)
            scenario["failed"] = args.samples - len(passed)
            scenario["status"] = (
                "passed"
                if len(passed) == args.samples and scenario["excluded_warmup"]["status"] == "passed"
                else "failed"
            )
            scenario["metrics_ms"] = (
                {key: summary([row["stages"][key] for row in passed]) for key in passed[0]["stages"]} if passed else {}
            )
            scenario["storage_bytes"] = (
                {
                    key: summary([row["storage"][key] for row in passed])
                    for key in passed[0]["storage"]
                    if key != "checkpoints"
                }
                if passed
                else {}
            )
            scenario["comparison"] = (
                {key: summary([row["comparison"][key] for row in passed]) for key in passed[0]["comparison"]}
                if passed
                else {}
            )
            scenario["finished_utc"] = utc()
            save()
        report["status"] = "passed" if all(row["status"] == "passed" for row in report["scenarios"]) else "failed"
    finally:
        report["cleanup"] = "passed" if not list(root.iterdir()) else "failed: owned workspace retained"
        if report["cleanup"] == "passed":
            root.rmdir()
        report["resources"] = sampler.stop()
        report["finished_utc"] = utc()
        report["wall_seconds"] = time.perf_counter() - tick
        save()
    return 0 if report["status"] == report["cleanup"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--lengths", nargs="+", type=int, default=[32, 128, 512, 1024])
    parser.add_argument("--files", type=int, default=32)
    parser.add_argument("--variables", type=int, default=64)
    parser.add_argument("--baseline-bytes", type=int, default=4096)
    parser.add_argument("--step-bytes", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=32)
    arguments = parser.parse_args()
    if (
        min(
            arguments.samples,
            arguments.files,
            arguments.variables,
            arguments.baseline_bytes,
            arguments.step_bytes,
            arguments.stride,
            *arguments.lengths,
        )
        <= 0
    ):
        parser.error("sizes and samples must be positive")
    if any(length < 2 or length % 2 for length in arguments.lengths):
        parser.error("lengths must be even and at least two")
    raise SystemExit(asyncio.run(main(arguments)))
