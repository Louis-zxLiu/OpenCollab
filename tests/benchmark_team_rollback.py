"""Real-provider Team experiment through TeamHandle, with phase resource evidence.

Run with caller-supplied provider configuration, task, Team YAML and disposable run path.
This harness never supplies or repairs the model's solution.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rollback_measurements import ResourceSampler, provenance, stamp, write_json
from team_workload_oracles import verify_graph_solution

from opencollab.adapters.trace import Tracer
from opencollab.sdk import OpenCollab


def _command(argv, cwd, **kwargs):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60, **kwargs)


def _verify(workspace, evidence):
    source, binary = workspace / "solution.cpp", evidence / "solution-test"
    if not source.is_file():
        raise AssertionError("model did not produce solution.cpp")
    compiled = _command(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)], workspace)
    result = {
        "solution_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compile_exit": compiled.returncode,
        "groups": [],
    }
    (evidence / "compile.log").write_text(compiled.stderr)
    if compiled.returncode:
        raise AssertionError("model solution failed compilation")
    groups = {
        "sample": [([5, -1, 7, 2], [(2, 1, 4), (1, 2, 9), (2, 1, 4), (2, 2, 3)])],
        "boundary": [
            ([-(2**63)], [(2, 1, 1), (1, 1, 2**63 - 1), (2, 1, 1)]),
            ([2**63 - 1] * 4, [(2, 1, 4), (1, 4, -(2**63)), (2, 1, 3), (2, 4, 4)]),
        ],
    }
    rng = random.Random(20260910)
    cases = []
    for _ in range(300):
        n = rng.randint(1, 60)
        values, operations = [rng.randint(-(10**12), 10**12) for _ in range(n)], []
        for _ in range(100):
            left = rng.randint(1, n)
            operations.append(
                (1, left, rng.randint(-(10**12), 10**12)) if rng.random() < 0.45 else (2, left, rng.randint(left, n))
            )
        cases.append((values, operations))
    groups["random"] = cases
    try:
        for name, cases in groups.items():
            lines, expected = [str(len(cases))], []
            for values, operations in cases:
                current = values.copy()
                lines.extend([f"{len(values)} {len(operations)}", " ".join(map(str, values))])
                for kind, left, right in operations:
                    lines.append(f"{kind} {left} {right}")
                    if kind == 1:
                        current[left - 1] = right
                    else:
                        expected.append(str(min(current[left - 1 : right])))
            input_text = "\n".join(lines) + "\n"
            started = time.perf_counter()
            ran = _command([str(binary)], workspace, input=input_text)
            passed = ran.returncode == 0 and ran.stdout.split() == expected
            result["groups"].append(
                {
                    "name": name,
                    "cases": len(cases),
                    "queries": len(expected),
                    "elapsed_ms": (time.perf_counter() - started) * 1000,
                    "status": "passed" if passed else "failed",
                    "exit": ran.returncode,
                }
            )
            (evidence / f"{name}.input").write_text(input_text)
            (evidence / f"{name}.expected").write_text("\n".join(expected))
            (evidence / f"{name}.actual").write_text(ran.stdout)
        result["status"] = "passed" if all(r["status"] == "passed" for r in result["groups"]) else "failed"
        return result
    finally:
        binary.unlink(missing_ok=True)


async def _experiment(args, sampler):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    workspace, evidence = root / "workspace", root / "evidence"
    workspace.mkdir()
    evidence.mkdir()
    prompt = Path(args.prompt).read_text()
    (workspace / "TASK.md").write_text(prompt)
    for cmd in (
        ["git", "init", "-q"],
        ["git", "add", "TASK.md"],
        [
            "git",
            "-c",
            "user.name=Experiment",
            "-c",
            "user.email=experiment@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
    ):
        _command(cmd, workspace, check=True)
    report = {
        **provenance(),
        "harness_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": stamp(),
        "initial_budget": args.budget,
        "continuation_budget": args.reserve,
        "mode": args.mode,
        "workload": args.workload,
        "seed": args.seed,
        "prompt_sha256": hashlib.sha256(Path(args.prompt).read_bytes()).hexdigest(),
        "team_config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(Path(__file__).with_name("team_workload_oracles.py").read_bytes()).hexdigest(),
        "timeline": [],
        "status": "running",
        "target_source": "trace sink",
        "provider_process_memory": "not measured: remote provider",
        "agent_memory": "shared controller process",
    }
    before_host, roles, effects, completions = dict(os.environ), {}, [], []
    original_log = Tracer.log_step

    def observe(tracer, step_type, payload, *values, **kw):
        original_log(tracer, step_type, payload, *values, **kw)
        if step_type == "agent_completed":
            roles[payload["aid"]] = payload["role"]
            completions.append(dict(payload))
        if step_type == "effect_created":
            effects.append(dict(payload))

    def event(name, **kw):
        row = {"event": name, "utc": stamp(), **kw}
        report["timeline"].append(row)
        print(json.dumps(row), flush=True)
        write_json(evidence / "result.json", report)

    handle = waiter = None
    Tracer.log_step = observe
    try:
        with sampler.measure("start_team"):
            handle = await OpenCollab(
                workspace, config={"llm_timeout": 180, "llm_first_event_timeout": 90, "llm_stream_idle_timeout": 90}
            ).start_team(
                prompt,
                config=args.config,
                budget=args.budget,
                artifacts=root / "artifacts",
                trace=True,
                use_worktrees=True,
                allow_unisolated_shell=True,
                prebuild_team=False,
                max_steps=70,
            )
        waiter = asyncio.create_task(handle.wait())
        event("start_team_returned")
        sampler.phase = "initial_team"
        if args.mode == "rollback":
            selected = None
            deadline = time.monotonic() + args.window
            while time.monotonic() < deadline and not waiter.done():
                selected = next((e for e in effects if roles.get(e["producer_aid"]) == "coder"), None)
                if selected:
                    break
                await asyncio.sleep(0.01)
            if selected is None or waiter.done():
                report["status"] = "no controlled rollback window"
                if waiter.done():
                    early = waiter.result()
                    report["early_team_result"] = {"status": early.status, "reason": early.reason}
                    if not early.ok:
                        report["status"] = "failed"
                return report
            target = selected["effect_id"]
            with sampler.measure("operator_preview"):
                plan = handle.rollback.preview({target})
            report["plan"] = {
                **asdict(plan),
                "target_effect_ids": sorted(plan.target_effect_ids),
                "invalidated_effect_ids": sorted(plan.invalidated_effect_ids),
                "affected_agent_ids": sorted(plan.affected_agent_ids),
            }
            event("operator_preview", effect_id=target, affected=sorted(plan.affected_agent_ids))
            # These probes audit internal invariants, separately from the public API path.
            scheduler = handle._scheduler
            old_epochs = {aid: scheduler.current_effect_epoch(aid) for aid in plan.affected_agent_ids}
            siblings = [aid for aid, role in roles.items() if role == "independent"]
            sibling_epochs = {aid: scheduler.current_effect_epoch(aid) for aid in siblings}
            for method in ("_cancel_fenced_tasks", "_settle_rollback_turns", "_quiesce_agents"):
                original = getattr(scheduler, method)

                async def measured(*values, _original=original, _name=method, **kw):
                    with sampler.measure(_name):
                        return await _original(*values, **kw)

                setattr(scheduler, method, measured)
            with sampler.measure("operator_execute"):
                result = await handle.rollback.execute({target}, expected_plan_digest=plan.digest)
            report["restores"] = [asdict(r) for r in result.restores]
            assert result.invalidated, "rollback failed; effects remain active"
            assert all(
                r.status == "restored"
                and r.filesystem_digest == plan.checkpoint_filesystem_digests[r.agent_id]
                and r.environment_digest == plan.checkpoint_environment_digests[r.agent_id]
                for r in result.restores
            )
            report["scope_verification"] = {
                aid: {
                    "fenced": aid in scheduler._rollback_fenced,
                    "idle": scheduler._sessions[aid].state.phase.value == "idle",
                    "pwd": scheduler._sessions[aid].env.environment_view()["PWD"]
                    == scheduler._sessions[aid].env.workspace,
                    "identity": hashlib.sha256(scheduler._sessions[aid].env.workspace.encode()).hexdigest()
                    == plan.checkpoint_identity_digests[aid],
                }
                for aid in plan.affected_agent_ids
            }
            assert all(all(v.values()) for v in report["scope_verification"].values())
            report["sibling_isolation"] = bool(siblings) and all(
                a not in plan.affected_agent_ids and scheduler.current_effect_epoch(a) == sibling_epochs[a]
                for a in siblings
            )
            assert report["sibling_isolation"]
            event("operator_execute_returned", invalidated=result.invalidated)
            with sampler.measure("original_turn_settlement"):
                interrupted = await asyncio.wait_for(asyncio.shield(waiter), 90)
            report["interruption"] = {"status": interrupted.status, "reason": interrupted.reason}
            assert interrupted.status == "stopped" and "rollback" in str(interrupted.reason)
            report["used_tokens_at_rollback"] = interrupted.tokens
            with sampler.measure("explicit_resume"):
                handle.rollback.resume(result.affected_agent_ids)
            report["epochs"] = {aid: [old, scheduler.current_effect_epoch(aid)] for aid, old in old_epochs.items()}
            try:
                scheduler.create_effect_ref(
                    producer_aid=selected["producer_aid"],
                    kind="tool_result",
                    epoch=old_epochs[selected["producer_aid"]],
                    attempt=1,
                )
            except RuntimeError:
                report["old_epoch_rejected"] = True
            else:
                raise AssertionError("old epoch accepted")
            previous_completions = len(completions)
            event("explicit_resume_and_new_turn")
            with sampler.measure("explicit_retry_new_turn"):
                await asyncio.wait_for(
                    handle.rollback.continue_turn(
                        0,
                        "The operator has withdrawn the prior coder result and restored affected workspaces. "
                        "Spawn a NEW coder for TASK.md and a NEW prover to compile and independently test "
                        "the new source. "
                        "Integrate the complete accepted source into solution.cpp in your own workspace and test it.",
                        budget_tokens=args.reserve,
                    ),
                    args.continuation,
                )
            assert any(c["role"] == "coder" for c in completions[previous_completions:]), "no coder retry"
            assert any(c["role"] == "prover" for c in completions[previous_completions:]), "no prover retry"
            final = await handle.wait()
        else:
            with sampler.measure("fresh_team_completion"):
                final = await asyncio.wait_for(asyncio.shield(waiter), args.continuation)
        report["final"] = {"status": final.status, "reason": final.reason, "tokens": final.tokens}
        assert final.ok, "Team turn failed"
        with sampler.measure("compile_and_oracle_tests"):
            report["solution"] = (
                _verify(workspace, evidence) if args.workload == "range-min"
                else verify_graph_solution(workspace, evidence, args.workload, args.seed)
            )
        assert report["solution"]["status"] == "passed"
        report["status"] = "passed"
    except Exception as exc:
        import traceback

        (evidence / "failure.txt").write_text(traceback.format_exc())
        report.update(status="failed", error_reason=type(exc).__name__)
        event("failure", error_type=type(exc).__name__)
    finally:
        if handle is not None:
            try:
                with sampler.measure("close"):
                    await handle.close()
                refs = _command(
                    ["git", "for-each-ref", "--format=%(refname)", "refs/opencollab/checkpoints"],
                    workspace, check=True,
                ).stdout.splitlines()
                worktrees = _command(["git", "worktree", "list", "--porcelain"], workspace, check=True).stdout
                report["cleanup_verification"] = {
                    "checkpoint_refs_remaining": len(refs),
                    "registered_worktrees": sum(line.startswith("worktree ") for line in worktrees.splitlines()),
                }
                assert not refs and report["cleanup_verification"]["registered_worktrees"] == 1, "Scope leak"
                report["cleanup"] = "passed"
            except Exception as exc:
                report.update(status="failed", cleanup=type(exc).__name__)
        if waiter is not None and waiter.done():
            await asyncio.gather(waiter, return_exceptions=True)
        Tracer.log_step = original_log
        report.update(ended_utc=stamp(), host_environment_unchanged=dict(os.environ) == before_host)
        if not report["host_environment_unchanged"]:
            report.update(status="failed", error_reason="host_environment_changed")
        write_json(evidence / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    for field in ("run-dir", "config", "prompt"):
        parser.add_argument(f"--{field}", required=True)
    parser.add_argument("--mode", choices=("rollback", "fresh"), default="rollback")
    parser.add_argument("--workload", choices=("range-min", "shortest-path", "connectivity"), default="range-min")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--budget", type=int, default=2_000_000)
    parser.add_argument("--reserve", type=int, default=600_000)
    parser.add_argument("--window", type=float, default=600)
    parser.add_argument("--continuation", type=float, default=900)
    args = parser.parse_args()
    sampler = ResourceSampler().start()
    try:
        report = asyncio.run(_experiment(args, sampler))
    finally:
        resources = sampler.stop()
        write_json(Path(args.run_dir) / "evidence" / "resources.json", resources)
    print(json.dumps({"status": report["status"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
