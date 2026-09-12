"""Linux benchmark measurements; never imported by the product runtime."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def provenance():
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(("git", *args), cwd=root).decode().strip()

    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "harness_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("benchmark_rollback.py"))
        },
        "source_content_sha256": hashlib.sha256(
            b"".join(
                p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes()
                for p in sorted((root / "opencollab").rglob("*.py"))
            )
        ).hexdigest(),
    }


def summarize(values):
    if not values:
        return {"status": "not measured"}
    ordered = sorted(values)
    return {
        "samples": len(values),
        "median": statistics.median(values),
        "p95": ordered[max(0, (95 * len(values) + 99) // 100 - 1)],
    }


class ResourceSampler:
    """Timestamped observations for this controller PID and its visible descendants."""

    def __init__(self, interval=0.02):
        self.interval = interval
        self.pid = os.getpid()
        self.phase = "setup"
        self.rows, self.phases = [], []
        self.containers = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._thread.start()
        return self

    def _sample(self):
        observed, phase = time.monotonic_ns(), self.phase
        processes, stack, included = {}, [self.pid], set()
        while stack:
            pid = stack.pop()
            if pid in included:
                continue
            included.add(pid)
            entry = Path(f"/proc/{pid}")
            try:
                raw = (entry / "stat").read_text()
                fields = raw[raw.rfind(")") + 2 :].split()
                processes[pid] = (int(fields[1]), int(fields[21]) * os.sysconf("SC_PAGE_SIZE"))
                # Children can belong to any thread, not only the process leader.
                for task in (entry / "task").iterdir():
                    try:
                        stack.extend(map(int, (task / "children").read_text().split()))
                    except (OSError, ValueError):
                        continue
            except (OSError, ValueError, IndexError):
                continue
        current, peak = tracemalloc.get_traced_memory()
        self.rows.append(
            {
                "monotonic_ns": observed,
                "utc": stamp(),
                "phase": phase,
                "controller_pid": self.pid,
                "controller_rss_bytes": processes.get(self.pid, (0, None))[1],
                "process_tree_rss_bytes": sum(processes[p][1] for p in included if p in processes) or None,
                "observed_pids": sorted(included & processes.keys()),
                "python_current_bytes": current,
                "python_peak_bytes": peak,
                "containers": {cid: self._container_sample(path) for cid, path in tuple(self.containers.items())},
            }
        )

    def watch_container(self, container_id):
        pid = int(subprocess.check_output(["docker", "inspect", "--format", "{{.State.Pid}}", container_id]).decode())
        for row in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if row.startswith("0::"):
                self.containers[container_id] = Path("/sys/fs/cgroup") / row[3:].lstrip("/")

    @staticmethod
    def _container_sample(path):
        values = {}
        for metric in ("memory.current", "memory.peak", "cpu.stat", "io.stat"):
            try:
                content = (path / metric).read_text().strip()
                values[metric] = int(content) if metric.startswith("memory.") else content
            except (OSError, ValueError):
                values[metric] = "not measured"
        return values

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    @contextmanager
    def measure(self, name):
        previous = self.phase
        self.phase = name
        start = time.monotonic_ns()
        record = {
            "phase": name,
            "started_utc": stamp(),
            "start_ns": start,
            "python_before_bytes": tracemalloc.get_traced_memory()[0],
            "status": "passed",
        }
        try:
            yield record
        except BaseException as exc:
            record.update(status="failed", error_reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record.update(
                ended_utc=stamp(),
                elapsed_ms=(time.monotonic_ns() - start) / 1e6,
                python_current_bytes=tracemalloc.get_traced_memory()[0],
                python_peak_bytes=tracemalloc.get_traced_memory()[1],
            )
            self.phases.append(record)
            self.phase = previous

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("resource sampler failed to stop")
        intervals = [(b["monotonic_ns"] - a["monotonic_ns"]) / 1e6 for a, b in zip(self.rows, self.rows[1:])]
        return {
            "requested_interval_ms": self.interval * 1000,
            "observed_intervals_ms": summarize(intervals),
            "observations": self.rows,
            "phases": self.phases,
            "limitations": "RSS sum includes shared pages; short subprocesses may be missed. "
            "Heap peak is campaign-wide, not per-phase. Sampler overhead is included.",
        }


def git_storage(workspace):
    def git(*args):
        return subprocess.check_output(("git", *args), cwd=workspace).decode()

    git_dir = Path(git("rev-parse", "--absolute-git-dir").strip())
    objects = [p for p in (git_dir / "objects").rglob("*") if p.is_file()]
    entries = git("cat-file", "--batch-all-objects", "--batch-check=%(objectsize)").splitlines()
    refs = git("for-each-ref", "--format=%(refname)", "refs/opencollab/checkpoints").splitlines()
    files = [p for p in workspace.rglob("*") if p.is_file() and ".git" not in p.relative_to(workspace).parts]
    return {
        "git_objects_count": len(entries),
        "git_objects_logical_bytes": sum(map(int, entries)),
        "git_objects_stored_bytes": sum(p.stat().st_size for p in objects),
        "git_objects_allocated_bytes": sum(p.stat().st_blocks * 512 for p in objects),
        "checkpoint_refs_count": len(refs),
        "checkpoint_ref_bytes": sum((git_dir / r).stat().st_size for r in refs),
        "workspace_file_bytes": sum(p.stat().st_size for p in files),
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
