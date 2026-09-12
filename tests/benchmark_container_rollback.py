"""Real Docker rollback measurements; deterministic sessions, no model calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollback_measurements import ResourceSampler, provenance, stamp
from scheduler_awaiting_test_support import ScriptedSession, build_scheduler

from opencollab.adapters._env_container_worktree import ContainerWorktreeEnvironment
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters.git_checkpoints import GitCheckpointAdapter
from opencollab.domain.rollback import EnvironmentSnapshot


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout.decode().strip()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def stats(rows, key):
    values = sorted(row[key] for row in rows)
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[math.ceil(len(values) * 0.95) - 1],
    }


async def checked(env, cmd):
    result = await env.exec_cmd(cmd)
    check(result.returncode == 0, "container command failed")
    return result.stdout


async def scenario(root, image, depth, files, variables, retained, samples, cancel_docker, sampler):
    result = {
        "graph_depth": depth,
        "tracked_files": files,
        "scope_variables": variables,
        "retained_checkpoints": retained,
        "requested_samples": samples,
        "warmups": 1,
        "active_docker_cancellation": cancel_docker,
        "status": "failed",
        "records": [],
        "cleanup": "pending",
        "started_utc": stamp(),
    }
    base = None
    env = None
    host_before = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="docker-matrix-", dir=root) as raw:
        repo = Path(raw)
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "Benchmark")
        git(repo, "config", "user.email", "bench@opencollab.invalid")
        (repo / ".gitignore").write_text("*.ignored\n.opencollab/\n")
        for index in range(files):
            (repo / f"f{index}.txt").write_text("baseline\n")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "baseline")
        try:
            started = time.perf_counter()
            base = DockerEnvironment(image=image, workspace="/repo")
            await base.setup()
            sampler.watch_container(base._container_id)
            await base.write_file(".gitignore", "*.ignored\n.opencollab/\n")
            for index in range(files):
                await base.write_file(f"f{index}.txt", "baseline\n")
            await checked(base, "git init -q && git add . && git commit -qm baseline")
            result["container_id"] = base._container_id
            env = ContainerWorktreeEnvironment(
                container_id=base._container_id,
                repository_root="/repo",
                worktree_root="/benchmark-scopes",
                branch_name="measurement",
            )
            await env.setup()
            result["setup_ms"] = (time.perf_counter() - started) * 1000
            values = {f"BENCH_{i}": f"value-{i}" for i in range(variables)}
            values["PWD"] = env.workspace
            env.replace_environment(EnvironmentSnapshot.from_mapping(values))
            await env.write_file("baseline.ignored", "ignored baseline\n")
            await env.write_file("baseline.untracked", "untracked baseline\n")
            await checked(env, "mkdir -p .opencollab; printf retained > .opencollab/evidence")
            retained_start = time.perf_counter()
            for _ in range(retained):
                await env.checkpoint_scope("initial", owner_aid=0, causal_frontier=frozenset())
            result["retained_checkpoint_setup_ms"] = (time.perf_counter() - retained_start) * 1000
            for sample in range(samples + 1):
                lead = ScriptedSession("lead", [])
                lead.env = env
                scheduler, events = build_scheduler(lead, [])
                scheduler.register_effect_environment(0, env)
                start = time.perf_counter()
                with sampler.measure("container_checkpoint"):
                    cp = await scheduler.create_checkpoint(0)
                row = {"sample": sample, "checkpoint_ms": (time.perf_counter() - start) * 1000}
                start = time.perf_counter()
                parent = ()
                for index in range(depth):
                    effect = scheduler.create_effect_ref(
                        producer_aid=0,
                        kind="tool_result",
                        epoch=0,
                        attempt=index,
                        parent_effect_ids=parent,
                        content=str(index),
                    )
                    if index == 0:
                        target = effect.effect_id
                    parent = (effect.effect_id,)
                row["graph_build_ms"] = (time.perf_counter() - start) * 1000
                start = time.perf_counter()
                plan = scheduler.preview_rollback({target})
                row["preview_ms"] = (time.perf_counter() - start) * 1000
                check(len(plan.invalidated_effect_ids) == depth, "descendant count mismatch")
                await checked(
                    env,
                    "printf changed > f0.txt; rm -- f1.txt; printf changed > baseline.ignored; "
                    "printf extra > new.ignored; printf extra > new.txt",
                )
                env.set_environment_variable("BENCH_0", "changed")
                start = time.perf_counter()
                with sampler.measure("container_adapter_restore"):
                    restored = await GitCheckpointAdapter(env).restore_scope(cp)
                row["adapter_restore_ms"] = (time.perf_counter() - start) * 1000
                check(restored.status == "restored", "adapter restore failed")
                check(restored.filesystem_digest == cp.filesystem_digest, "filesystem digest mismatch")
                check(restored.environment_digest == cp.environment.digest(), "environment digest mismatch")
                await checked(env, "printf changed-again > f0.txt; rm -f .opencollab/ready .opencollab/late")
                env.set_environment_variable("BENCH_0", "changed-again")
                shell_command = "printf ready > .opencollab/ready; sleep 60; printf bad > .opencollab/late"
                shell = asyncio.create_task(env.exec_cmd(shell_command) if cancel_docker else asyncio.sleep(60))
                scheduler._tasks[0] = shell
                sibling = asyncio.create_task(asyncio.sleep(60))
                scheduler._tasks[1] = sibling
                try:
                    if cancel_docker:
                        for _ in range(100):
                            if (await env.exec_cmd("test -f .opencollab/ready")).returncode == 0:
                                break
                            await asyncio.sleep(0.02)
                        else:
                            raise RuntimeError("shell readiness deadline")
                    phase_times = {}
                    for name in (
                        "_reject_rollback_messages",
                        "_cancel_fenced_tasks",
                        "_settle_rollback_turns",
                        "_quiesce_agents",
                    ):
                        original = getattr(scheduler, name)

                        async def timed(*args, original=original, name=name):
                            start = time.perf_counter()
                            try:
                                with sampler.measure(name):
                                    return await original(*args)
                            finally:
                                phase_times[name] = (time.perf_counter() - start) * 1000

                        setattr(scheduler, name, timed)
                    start = time.perf_counter()
                    with sampler.measure("container_scheduler_rollback"):
                        outcome = await scheduler.rollback_effect({target}, expected_plan_digest=plan.digest())
                    row["scheduler_rollback_ms"] = (time.perf_counter() - start) * 1000
                    row["phases_ms"] = phase_times
                    check(outcome.invalidated, "scheduler rollback failed")
                    check(shell.done(), "shell still active")
                    check(not sibling.done(), "independent sibling was cancelled")
                    check(scheduler._rollback_fenced == {0}, "wrong fence set")
                    check(
                        all(item.status == "invalidated" for item in scheduler._rollback_service.effects.values()),
                        "partial invalidation",
                    )
                    start = time.perf_counter()
                    await checked(
                        env,
                        "test ! -e new.txt && test ! -e new.ignored && test ! -e .opencollab/late "
                        '&& test -f f1.txt && test "$(cat f0.txt)" = baseline '
                        '&& test "$(cat .opencollab/evidence)" = retained && test "$BENCH_0" = value-0',
                    )
                    identity = await env.checkpoint_workspace_identity()
                    check(identity == cp.workspace_identity, "identity mismatch")
                    check(env.snapshot_environment().digest() == cp.environment.digest(), "scope digest mismatch")
                    pwd_ok = (await checked(env, "pwd -P")).strip() == identity
                    check(pwd_ok, "PWD mismatch")
                    check(dict(os.environ) == host_before, "host environment changed")
                    row["verification_ms"] = (time.perf_counter() - start) * 1000
                    start = time.perf_counter()
                    scheduler.resume_after_rollback({0})
                    row["resume_ms"] = (time.perf_counter() - start) * 1000
                    try:
                        scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=0, attempt=0)
                    except RuntimeError:
                        old_epoch_rejected = True
                    else:
                        old_epoch_rejected = False
                    check(old_epoch_rejected, "old epoch accepted")
                    row.update(
                        status="passed",
                        invalidated_effects=depth,
                        affected_agents=[0],
                        filesystem_checkpoint=cp.filesystem_digest,
                        filesystem_restored=outcome.restores[0].filesystem_digest,
                        environment_checkpoint=cp.environment.digest(),
                        environment_restored=outcome.restores[0].environment_digest,
                        identity_digest=hashlib.sha256(identity.encode()).hexdigest(),
                        pwd_match=pwd_ok,
                        sibling_isolated=True,
                        host_environment_unchanged=True,
                        old_epoch_rejected=True,
                        new_command_environment_match=True,
                        epoch_after=scheduler.current_effect_epoch(0),
                    )
                    row["ended_utc"] = stamp()
                    if sample:
                        result["records"].append(row)
                        if sample % 5 == 0:
                            print(json.dumps({"depth": depth, "completed": sample}), flush=True)
                    else:
                        result["warmup_record"] = row
                finally:
                    for task in (shell, sibling):
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(shell, sibling, return_exceptions=True)
            # A missing checkpoint ref must fail closed; explicit repair and retry recover it.
            lead = ScriptedSession("lead", [])
            lead.env = env
            scheduler, _ = build_scheduler(lead, [])
            scheduler.register_effect_environment(0, env)
            cp = await scheduler.create_checkpoint(0)
            target = scheduler.create_effect_ref(producer_aid=0, kind="tool_result", epoch=0, attempt=0)
            plan = scheduler.preview_rollback({target.effect_id})
            ref = f"refs/opencollab/checkpoints/{cp.checkpoint_id}"
            await env._checkpoint_git("update-ref", "-d", ref)
            try:
                await scheduler.rollback_effect({target.effect_id}, expected_plan_digest=plan.digest())
            except ValueError:
                pass
            else:
                raise AssertionError("missing checkpoint ref was accepted")
            check(not scheduler._rollback_fenced, "preflight failure fenced an agent")
            check(scheduler.current_effect_epoch(0) == 0, "preflight failure changed epoch")
            await env._checkpoint_git("update-ref", ref, cp.filesystem_revision)
            recovery = await scheduler.rollback_effect({target.effect_id}, expected_plan_digest=plan.digest())
            check(recovery.invalidated, "explicit retry failed")
            scheduler.resume_after_rollback({0})
            result["in_process_failure_recovery"] = "passed"
            result["status"] = "passed"
            result["metrics"] = {
                key: stats(result["records"], key)
                for key in (
                    "checkpoint_ms",
                    "graph_build_ms",
                    "preview_ms",
                    "adapter_restore_ms",
                    "scheduler_rollback_ms",
                    "verification_ms",
                    "resume_ms",
                )
            }
        except Exception as exc:
            result["error_reason"] = type(exc).__name__ + ": " + str(exc).splitlines()[0][:160]
            result["error_detail_debug"] = str(exc)
            print(json.dumps({"depth": depth, "status": "failed", "error_type": type(exc).__name__}), flush=True)
        finally:
            if base is not None and base._container_id:
                inspected = subprocess.run(
                    ["docker", "inspect", "--size", "--format", "{{.SizeRw}}", base._container_id],
                    capture_output=True,
                    text=True,
                )
                result["writable_layer_bytes"] = int(inspected.stdout) if inspected.returncode == 0 else "not measured"
            try:
                if env is not None:
                    await env.cleanup()
            except Exception as exc:
                result["worktree_cleanup_error"] = type(exc).__name__
                result["status"] = "failed"
            try:
                if base is not None:
                    await base.cleanup()
                result["cleanup"] = "passed"
            except Exception as exc:
                result["cleanup"] = type(exc).__name__
                result["status"] = "failed"
    result["ended_utc"] = stamp()
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--cancel-docker", action="store_true")
    args = parser.parse_args()
    image = os.environ["OPENCOLLAB_ROLLBACK_BENCHMARK_IMAGE"]
    subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True)
    report = {
        **provenance(),
        "schema": 2,
        "harness_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "real-container-worktree-and-scheduler; deterministic session; no LLM",
        "scenarios": [],
    }
    sampler = ResourceSampler().start()
    try:
        for sizes in ((32, 8, 16, 1), (128, 32, 64, 8), (512, 128, 128, 32)):
            report["scenarios"].append(
                await scenario(args.root, image, *sizes, args.samples, args.cancel_docker, sampler)
            )
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        report["resources"] = sampler.stop()
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"scenario_statuses": [s["status"] for s in report["scenarios"]]}), flush=True)
    return 0 if all(s["status"] == "passed" for s in report["scenarios"]) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
