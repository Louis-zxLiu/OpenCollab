from __future__ import annotations

from dataclasses import dataclass

import pytest
from benchmark_rollback import _build_graph, _measure


@dataclass
class _Restore:
    status: str
    reason: str | None = None
    filesystem_digest: str | None = None


def test_measure_marks_failed_restore_as_failed():
    result = _measure(
        "restore",
        2,
        lambda: _Restore("failed", reason="digest mismatch"),
    )

    assert result["status"] == "failed"
    assert result["error_reason"] == "digest mismatch"


def test_failed_warmup_is_retained_even_when_formal_sample_succeeds():
    values = iter((_Restore("failed", "warmup failed"), _Restore("restored")))
    result = _measure("restore", 1, lambda: next(values))
    assert result["status"] == "failed"
    assert result["raw_samples"][0]["warmup"]
    assert result["raw_samples"][1]["status"] == "passed"


@pytest.mark.parametrize("shape", ["chain", "balanced", "fanout", "diamond", "multi-agent"])
def test_graph_scenarios_keep_independent_sibling_outside_plan(shape):
    graph, target = _build_graph(32, shape, agents=16)
    plan = graph.preview_rollback({target})
    assert "e32" not in plan.invalidated_effect_ids
    if shape == "balanced":
        assert "e2" not in plan.invalidated_effect_ids
        assert "e3" in plan.invalidated_effect_ids
    if shape == "diamond":
        assert graph.effects["e2"].parent_effect_ids == ("e0",)
        assert graph.effects["e5"].parent_effect_ids == ("e1", "e2", "e3", "e4")

def test_sampler_observes_child_created_by_worker_thread():
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    from rollback_measurements import ResourceSampler

    if not Path("/proc/self/task").exists():
        pytest.skip("Linux procfs is required for resource observations")
    with ThreadPoolExecutor(max_workers=1) as pool:
        child = pool.submit(subprocess.Popen, [sys.executable, "-c", "import time; time.sleep(30)"]).result()
        try:
            sampler = ResourceSampler()
            sampler._sample()
            row = sampler.rows[-1]
            assert child.pid in row["observed_pids"]
            assert sampler.pid in row["observed_pids"]
            assert row["controller_rss_bytes"] > 0
            assert row["process_tree_rss_bytes"] >= row["controller_rss_bytes"]
        finally:
            child.terminate()
            child.wait(timeout=5)
