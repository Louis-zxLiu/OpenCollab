"""Real-provider historical replay with public TeamHandle history and independent oracles."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import resource
import shlex
import subprocess
import tempfile
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

from benchmark_team_rollback import _verify
from rollback_measurements import provenance, stamp
from team_workload_oracles import verify_graph_solution

from opencollab.sdk import OpenCollab

TASKS = {
    "range-min": (
        "Implement point updates and inclusive range-minimum queries. Input starts with T. "
        "Each case: n q, n signed 64-bit integers, then q operations. Operation 1 i v sets "
        "a[i]=v; operation 2 l r prints min(a[l..r]). Indices are 1-based. n,q up to 200000. "
        "Use a segment tree; handle LLONG_MIN and LLONG_MAX."
    ),
    "shortest-path": (
        "Implement single-source shortest paths on a directed graph with nonnegative weights. "
        "Input starts with T; each case has n m s followed by m edges u v w. Print n distances "
        "from s, using -1 for unreachable vertices. Handle parallel edges, self loops, zero "
        "weights, and sums beyond 32-bit. n,m up to 200000; w up to 1000000000000."
    ),
    "connectivity": (
        "Implement incremental dynamic connectivity of an undirected graph. Input starts with T; "
        "each case has n q followed by q operations: 1 u v adds an edge; 2 u v prints YES if "
        "connected and NO otherwise. There are no deletions. Handle self loops, duplicate edges "
        "and disconnected components. n,q up to 200000. Use disjoint set union."
    ),
}


def memory():
    current, peak = tracemalloc.get_traced_memory()
    return {
        "python_current_bytes": current, "python_peak_bytes": peak,
        "controller_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "controller_rss": "blocked: current RSS requires out-of-workspace procfs",
        "process_tree_rss": "blocked: process-tree sampling requires out-of-workspace procfs",
    }


def _regular_file_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and not p.is_symlink())


def storage(path, run_root):
    git = path / ".git"
    temp_root = run_root / "tmp"
    scope_worktrees = [p for p in temp_root.glob("opencollab-wt-*") if p.is_dir()]
    return {
        "workspace_bytes": _regular_file_bytes(path),
        "workspace_content_bytes": sum(
            p.stat().st_size
            for p in path.rglob("*")
            if p.is_file() and not p.is_symlink() and git not in p.parents
        ),
        "git_objects_bytes": _regular_file_bytes(git / "objects"),
        "git_refs": len(subprocess.check_output(
            ["git", "for-each-ref", "--format=%(refname)"], cwd=path, text=True,
        ).splitlines()),
        "git_count_objects": subprocess.check_output(["git", "count-objects", "-v"], cwd=path, text=True),
        "scope_worktree_count": len(scope_worktrees),
        "scope_worktree_bytes": sum(_regular_file_bytes(worktree) for worktree in scope_worktrees),
    }


async def execute(args):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    temp_root = root / "tmp"
    temp_root.mkdir()
    tempfile.tempdir = str(temp_root)
    workspace = root / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=workspace, check=True)
    (workspace / "TASK.md").write_text(TASKS[args.workload])
    subprocess.run(["git", "add", "TASK.md"], cwd=workspace, check=True)
    subprocess.run(["git", "-c", "user.name=Replay Experiment", "-c", "user.email=replay@example.invalid",
                    "commit", "-qm", "test: seed algorithm workload"], cwd=workspace, check=True)
    # Load only simple assignments from the caller-selected existing configuration.
    for line in Path(args.config_env).read_text().splitlines():
        tokens = shlex.split(line, comments=True)
        if tokens and tokens[0] == "export":
            tokens = tokens[1:]
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                if key.startswith("OPENCOLLAB_"):
                    os.environ[key] = value
    tracemalloc.start()
    report = {
        **provenance(), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workload": args.workload, "mode": args.mode, "utc_start": stamp(), "status": "failed",
        "replays": [], "memory_start": memory(), "historical_replay": "not passed",
        "external_effects": "not reversible", "random_seed": 20260911,
    }
    handle = None
    started = time.perf_counter()
    try:
        client = OpenCollab(workspace, config={
            "llm_max_retries": 0, "llm_timeout": 300, "llm_connect_timeout": 60,
            "llm_first_event_timeout": 120, "llm_stream_idle_timeout": 120,
            "max_output_tokens": 4000,
        })
        handle = await client.start_team(
            TASKS[args.workload], config=Path(__file__).with_name("history_team.yaml"),
            budget=2_400_000, use_worktrees=True, allow_unisolated_shell=True,
            prebuild_team=True,
            trace=True, artifacts=root / "artifacts", max_steps=40,
        )
        if args.mode == "fresh":
            outcome = await asyncio.wait_for(handle.wait(), 480)
            report["team_status"] = outcome.status
            report["reason"] = outcome.reason
        else:
            # This is a caller-owned controlled window while the initial provider call is active.
            await asyncio.sleep(1)
            history = handle.history.list()
            sources = [e for e in history if e.agent_id == 0 and e.replayable and e.boundary == "initial"]
            if not sources:
                raise RuntimeError("no controlled replay window")
            source = sources[0]
            for replay_index in range(args.replays):
                tick = time.perf_counter()
                memory_before = memory()
                replay = await handle.replay_from(
                    checkpoint_id=source.checkpoint_id,
                    message="Complete this task from the restored state. " + TASKS[args.workload],
                    budget_tokens=600_000,
                )
                result = await asyncio.wait_for(replay.wait(), 480)
                memory_after = memory()
                history_snapshot = handle.history.list()
                history_rows = [asdict(entry) for entry in history_snapshot]
                checkpoint_rows = [entry for entry in history_rows if entry["checkpoint_id"]]
                row = {
                    **asdict(result), "source_history_entry": source.entry_id,
                    "source_checkpoint": source.checkpoint_id, "source_effect": source.effect_id,
                    "elapsed_ms": (time.perf_counter() - tick) * 1000,
                    "memory_start": memory_before,
                    "memory": {
                        **memory_after,
                        "python_current_delta_bytes": (
                            memory_after["python_current_bytes"] - memory_before["python_current_bytes"]
                        ),
                    },
                    "storage": storage(workspace, root),
                    "history_metrics": {
                        "entry_count": len(history_rows),
                        "checkpoint_count": len(checkpoint_rows),
                        "history_metadata_export_bytes": len(json.dumps(history_rows, separators=(",", ":")).encode()),
                        "checkpoint_metadata_export_bytes": len(
                            json.dumps(checkpoint_rows, separators=(",", ":")).encode()
                        ),
                    },
                }
                report["replays"].append(row)
                if result.status != "completed":
                    report["reason"] = result.status
                    break
                verify_dir = root / f"verify-{replay_index}"
                verify_dir.mkdir()
                row["verification"] = (
                    _verify(workspace, verify_dir) if args.workload == "range-min" else
                    verify_graph_solution(workspace, verify_dir, args.workload, 20260911)
                )
                after = handle.history.list()
                row["history"] = [asdict(e) for e in after]
                row["new_epoch_history"] = any(e.epoch > source.epoch and e.checkpoint_id for e in after)
                # Select a different registered stable checkpoint for the second explicit replay.
                candidates = [e for e in after if e.agent_id == 0 and e.replayable
                              and e.checkpoint_id != source.checkpoint_id and e.boundary == "turn_completed"]
                if replay_index + 1 < args.replays:
                    if not candidates:
                        report["reason"] = "no second stable checkpoint"
                        break
                    source = candidates[-1]
            passed = len(report["replays"]) == args.replays and all(
                r["status"] == "completed" and r.get("verification", {}).get("status") == "passed"
                and r["new_epoch_history"] for r in report["replays"]
            )
            report["historical_replay"] = "passed" if passed else "not passed"
            report["status"] = "passed" if passed else "failed"
        if args.mode == "fresh":
            verify_dir = root / "verify-fresh"
            verify_dir.mkdir()
            report["verification"] = (
                _verify(workspace, verify_dir) if args.workload == "range-min" else
                verify_graph_solution(workspace, verify_dir, args.workload, 20260911)
            )
            report["status"] = ("passed" if outcome.status == "completed"
                                and report["verification"]["status"] == "passed" else "failed")
    except Exception as exc:
        # Original text stays in the private run archive; public summaries use the category.
        (root / "private-error.txt").write_text(str(exc))
        report["error_reason"] = type(exc).__name__
    finally:
        report["total_ms"] = (time.perf_counter() - started) * 1000
        report["memory_end"] = memory()
        if handle is not None:
            report["history"] = [asdict(e) for e in handle.history.list()]
            try:
                await handle.close()
                report["close_status"] = "passed"
            except Exception as exc:
                (root / "private-cleanup-error.txt").write_text(str(exc))
                report["close_status"] = "failed"
                report["status"] = "failed"
        report["utc_end"] = stamp()
        (root / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"workload": args.workload, "status": report["status"],
                          "historical_replay": report["historical_replay"],
                          "error_reason": report.get("error_reason"), "reason": report.get("reason")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config-env", required=True)
    parser.add_argument("--workload", choices=TASKS, required=True)
    parser.add_argument("--mode", choices=("replay", "fresh"), default="replay")
    parser.add_argument("--replays", type=int, default=1)
    asyncio.run(execute(parser.parse_args()))
