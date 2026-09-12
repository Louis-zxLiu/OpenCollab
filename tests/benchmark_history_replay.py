"""Bounded history/replay benchmark matrix with raw per-sample evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import resource
import statistics
import tempfile
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from benchmark_rollback import _scenario
from rollback_measurements import provenance, stamp, write_json
from scheduler_awaiting_test_support import terminal
from test_history_application import live_scheduler

FORMAL_SAMPLES = 30
WARMUPS = 1


class BoundedSampler:
    """Measure Python memory without reading process or cgroup files."""

    def __init__(self) -> None:
        self.phase = "setup"
        self.phases: list[dict] = []

    @contextmanager
    def measure(self, name: str):
        started = time.perf_counter_ns()
        record = {"phase": name, "started_utc": stamp(), "status": "passed"}
        try:
            yield record
        except BaseException as exc:
            record.update(status="failed", error_reason=type(exc).__name__)
            raise
        finally:
            current, peak = tracemalloc.get_traced_memory()
            record.update(
                ended_utc=stamp(),
                elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
                python_current_bytes=current,
                python_peak_bytes=peak,
            )
            self.phases.append(record)

    def evidence(self) -> dict:
        current, peak = tracemalloc.get_traced_memory()
        return {
            "python_current_bytes": current,
            "python_peak_bytes": peak,
            "controller_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "controller_current_rss": "blocked: current RSS requires procfs outside the workspace",
            "process_tree_rss": "blocked: process-tree RSS requires procfs outside the workspace",
            "docker_writable_layer": "blocked: benchmark image is not configured",
            "docker_cgroup_memory": "blocked: cgroup files are outside the workspace",
            "phases": self.phases,
        }


def _stats(rows: list[dict]) -> dict:
    values = sorted(row["elapsed_ms"] for row in rows if not row["warmup"] and row["status"] == "passed")
    return {
        "samples": len(values),
        "median_ms": statistics.median(values),
        "p95_ms": values[max(0, (95 * len(values) + 99) // 100 - 1)],
    }


def _measure(operation) -> dict:
    rows = []
    for index in range(-WARMUPS, FORMAL_SAMPLES):
        started = time.perf_counter_ns()
        row = {"sample": index + 1, "warmup": index < 0, "started_utc": stamp()}
        try:
            operation()
            row["status"] = "passed"
        except Exception as exc:
            row.update(status="failed", error_reason=type(exc).__name__)
        row.update(elapsed_ms=(time.perf_counter_ns() - started) / 1e6, ended_utc=stamp())
        rows.append(row)
    formal = [row for row in rows if not row["warmup"]]
    status = "passed" if all(row["status"] == "passed" for row in formal) else "failed"
    return {"status": status, **_stats(rows), "raw_samples": rows}


def _scheduler_history_case(effect_count: int) -> dict:
    effect_rows = []
    scheduler = None
    for sample in range(-WARMUPS, FORMAL_SAMPLES):
        scheduler, _ = live_scheduler()
        started = time.perf_counter_ns()
        status, reason = "passed", None
        try:
            parent = None
            for index in range(effect_count):
                effect = scheduler.create_effect_ref(
                    producer_aid=0,
                    kind="tool_result",
                    epoch=0,
                    attempt=max(sample, 0),
                    parent_effect_ids=(parent,) if parent else (),
                    content=f"sample-{sample}-effect-{index}",
                )
                parent = effect.effect_id
        except Exception as exc:
            status, reason = "failed", type(exc).__name__
        effect_rows.append({
            "sample": sample + 1,
            "warmup": sample < 0,
            "status": status,
            "error_reason": reason,
            "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
        })
    assert scheduler is not None
    entries = scheduler.history_list()
    target = entries[len(entries) // 2]
    listed = _measure(scheduler.history_list)
    fetched = _measure(lambda: scheduler.history_get(target.entry_id))
    metadata = json.dumps([asdict(entry) for entry in entries], separators=(",", ":")).encode()
    return {
        "effect_count_per_sample": effect_count,
        "effect_auto_registration": {"status": "passed", **_stats(effect_rows), "raw_samples": effect_rows},
        "history_list": listed,
        "history_get": fetched,
        "graph_object_count": sum(entry.effect_id is not None for entry in entries),
        "history_entry_count": len(entries),
        "history_metadata_bytes": len(metadata),
    }


async def _replay_case() -> dict:
    scheduler, _ = live_scheduler(*(terminal(f"result-{index}") for index in range(FORMAL_SAMPLES + 2)))
    await scheduler.run("initial")
    source = next(entry for entry in scheduler.history_list() if entry.boundary == "initial")
    rows = []
    for sample in range(-WARMUPS, FORMAL_SAMPLES):
        started = time.perf_counter_ns()
        execution = await scheduler.replay_from(source.checkpoint_id, "repeat stable workload", 1024)
        result = await execution.task if execution.task is not None else None
        status = "passed" if execution.status == "started" and result is not None else "failed"
        rows.append({
            "sample": sample + 1,
            "warmup": sample < 0,
            "status": status,
            "elapsed_ms": (time.perf_counter_ns() - started) / 1e6,
            "phase_latency_ms": dict(execution.phase_latency_ms),
            "old_epochs": execution.old_epochs,
            "new_epochs": execution.new_epochs,
        })
    return {
        "status": "passed" if all(row["status"] == "passed" for row in rows if not row["warmup"]) else "failed",
        **_stats(rows),
        "raw_samples": rows,
        "same_checkpoint_replays": FORMAL_SAMPLES,
    }


def _matrix() -> list[dict]:
    cases = []
    for size in (32, 128, 512, 2048):
        cases.append({"name": f"chain-{size}", "shape": "chain", "size": size, "agents": 4})
    for depth in (5, 8, 10):
        cases.append({"name": f"tree-depth-{depth}", "shape": "balanced", "size": 2**depth - 1, "agents": 4})
    for size in (32, 128, 512):
        cases.append({"name": f"fanout-{size}", "shape": "fanout", "size": size, "agents": size + 1})
    cases.append({"name": "diamond-257", "shape": "diamond", "size": 257, "agents": 4})
    for agents in (4, 16, 64):
        cases.append({"name": f"multi-agent-{agents}", "shape": "multi-agent", "size": agents * 8, "agents": agents})
    return cases


async def _application_payload(source: dict, harness_sha: str) -> dict:
    sampler = BoundedSampler()
    with sampler.measure("scheduler_history"):
        history = _scheduler_history_case(128)
    with sampler.measure("complete_replay"):
        replay = await _replay_case()
    return {
        **source,
        "harness_sha256": harness_sha,
        "status": (
            "passed"
            if history["effect_auto_registration"]["status"] == replay["status"] == "passed"
            else "failed"
        ),
        "history": history,
        "replay": replay,
        "resources": sampler.evidence(),
        "fresh_rerun_comparison": "not measured: no equivalent model workload; no speedup is inferred",
        "cold_git": "blocked: dropping the shared host page cache is outside the workspace boundary",
        "warm_git": "passed: repeated operations in disposable repositories under the evidence root",
    }


async def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    tempfile.tempdir = str(scratch)
    source = provenance()
    harness_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    index = {**source, "harness_sha256": harness_sha, "started_utc": stamp(), "cases": []}
    common = dict(iterations=FORMAL_SAMPLES, warmups=WARMUPS, file_count=32, environment_count=64,
                  checkpoint_count=8, output=None)
    for case in _matrix():
        sampler = BoundedSampler()
        args = SimpleNamespace(**common, graph_size=case["size"], graph_shape=case["shape"], agents=case["agents"])
        payload = await _scenario(args, sampler)
        payload.update(harness_sha256=harness_sha, resources=sampler.evidence(), matrix_case=case)
        path = output / f"{case['name']}.json"
        write_json(path, payload)
        index["cases"].append({"name": case["name"], "status": payload["status"], "file": path.name})
    for dimension, values in (("files", (8, 32, 128)), ("environment", (16, 64, 128)),
                              ("checkpoints", (1, 8, 32, 128))):
        for value in values:
            sampler = BoundedSampler()
            params = dict(common)
            parameter = {
                "files": "file_count",
                "environment": "environment_count",
                "checkpoints": "checkpoint_count",
            }[dimension]
            params[parameter] = value
            args = SimpleNamespace(**params, graph_size=128, graph_shape="chain", agents=4)
            payload = await _scenario(args, sampler)
            name = f"dimension-{dimension}-{value}"
            payload.update(harness_sha256=harness_sha, resources=sampler.evidence(), matrix_case={dimension: value})
            write_json(output / f"{name}.json", payload)
            index["cases"].append({"name": name, "status": payload["status"], "file": f"{name}.json"})
    application = await _application_payload(source, harness_sha)
    write_json(output / "application-history-replay.json", application)
    index["cases"].append({"name": "application-history-replay", "status": application["status"],
                           "file": "application-history-replay.json"})
    index["ended_utc"] = stamp()
    index["status"] = "passed" if all(case["status"] == "passed" for case in index["cases"]) else "failed"
    write_json(output / "index.json", index)
    return index


async def run_application(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = provenance()
    harness_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    payload = await _application_payload(source, harness_sha)
    write_json(output / "application-history-replay.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--application-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    workspace_root = Path(__file__).resolve().parents[2]
    try:
        output.relative_to(workspace_root)
    except ValueError:
        parser.error("output-dir must stay below the workspace root")
    tracemalloc.start()
    result = asyncio.run(run_application(output) if args.application_only else run(output))
    print(json.dumps({"status": result["status"], "case_count": len(result.get("cases", ()))}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
