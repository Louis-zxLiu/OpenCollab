"""Known answers and negative controls for the multi-task Team verifier."""

import json

import pytest
from team_workload_oracles import _cases, _connectivity_case, _shortest_case, verify_graph_solution


def test_shortest_oracle_handles_parallel_edges_zero_cycle_and_unreachable():
    _, answers = _shortest_case(5, [(1, 2, 9), (1, 2, 0), (2, 1, 0), (2, 3, 10**12)], 1)
    assert answers == ["0", "0", "1000000000000", "-1", "-1"]


def test_connectivity_oracle_handles_self_duplicate_and_transitive_edges():
    _, answers = _connectivity_case(4, [(2, 1, 1), (2, 1, 3), (1, 1, 2), (1, 1, 2),
                                      (1, 2, 3), (2, 1, 3), (2, 1, 4)])
    assert answers == ["YES", "NO", "YES", "NO"]


@pytest.mark.parametrize("workload", ["shortest-path", "connectivity"])
def test_seeded_graph_cases_are_repeatable(workload):
    first = _cases(workload, 17)
    assert first == _cases(workload, 17)
    assert first != _cases(workload, 18)
    assert len(first["random"]) == 300


@pytest.mark.parametrize("source,reason", [(None, "missing_solution"), ("not C++", "compile_failed"),
                                          ("int main() { return 0; }", None)])
def test_graph_verifier_retains_missing_compile_and_wrong_answer_evidence(tmp_path, source, reason):
    workspace, evidence = tmp_path / "workspace", tmp_path / "evidence"
    workspace.mkdir()
    evidence.mkdir()
    if source is not None:
        (workspace / "solution.cpp").write_text(source)
    result = verify_graph_solution(workspace, evidence, "shortest-path", 17)
    assert result["status"] == "failed"
    if reason:
        assert result["error_reason"] == reason
    else:
        assert len(result["groups"]) == 3
        assert all(row["status"] == "failed" for row in result["groups"])
    assert json.loads((evidence / "solution-verification.json").read_text()) == result
    assert not (evidence / "solution-test").exists()
