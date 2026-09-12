"""Independent graph-task oracles for real-provider Team experiments.

These fixtures never write a candidate solution into the Team workspace.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time


def _shortest_case(n, edges, start):
    distances = [[float("inf")] * n for _ in range(n)]
    for i in range(n):
        distances[i][i] = 0
    for a, b, cost in edges:
        distances[a - 1][b - 1] = min(distances[a - 1][b - 1], cost)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
    lines = [f"{n} {len(edges)} {start}", *(f"{a} {b} {w}" for a, b, w in edges)]
    answers = [str(x) if x != float("inf") else "-1" for x in distances[start - 1]]
    return lines, answers


def _connectivity_case(n, operations):
    adjacency = [set() for _ in range(n)]
    answers = []
    for kind, a, b in operations:
        a, b = a - 1, b - 1
        if kind == 1:
            adjacency[a].add(b)
            adjacency[b].add(a)
        else:
            seen, pending = {a}, [a]
            while pending:
                for neighbor in adjacency[pending.pop()]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        pending.append(neighbor)
            answers.append("YES" if b in seen else "NO")
    return [f"{n} {len(operations)}", *(f"{k} {a} {b}" for k, a, b in operations)], answers


def _cases(workload, seed):
    rng = random.Random(seed)
    if workload == "shortest-path":
        groups = {
            "sample": [_shortest_case(4, [(1, 2, 5), (1, 3, 1), (3, 2, 1), (2, 4, 2)], 1)],
            "boundary": [
                _shortest_case(1, [], 1),
                _shortest_case(4, [(2, 2, 0), (2, 3, 10**12), (2, 3, 0), (3, 4, 10**12)], 2),
                _shortest_case(4, [], 4),
            ],
        }
        cases = []
        for _ in range(300):
            n = rng.randint(1, 18)
            edges = [(rng.randint(1, n), rng.randint(1, n), rng.randint(0, 10**12))
                     for _ in range(rng.randint(0, n * n))]
            cases.append(_shortest_case(n, edges, rng.randint(1, n)))
    elif workload == "connectivity":
        groups = {
            "sample": [_connectivity_case(4, [(2, 1, 2), (1, 1, 2), (1, 2, 3), (2, 1, 3), (2, 1, 4)])],
            "boundary": [
                _connectivity_case(1, [(2, 1, 1), (1, 1, 1), (2, 1, 1)]),
                _connectivity_case(4, [(1, 1, 2), (1, 1, 2), (1, 3, 4), (2, 1, 4), (1, 2, 3), (2, 1, 4)]),
                _connectivity_case(3, []),
            ],
        }
        cases = []
        for _ in range(300):
            n = rng.randint(1, 60)
            operations = [(1 if rng.random() < 0.45 else 2, rng.randint(1, n), rng.randint(1, n))
                          for _ in range(100)]
            cases.append(_connectivity_case(n, operations))
    else:
        raise ValueError(f"unknown graph workload: {workload}")
    groups["random"] = cases
    return groups


def verify_graph_solution(workspace, evidence, workload, seed):
    """Compile model output and retain correctness evidence even on failures."""
    source, binary = workspace / "solution.cpp", evidence / "solution-test"
    result = {"workload": workload, "seed": seed, "status": "failed", "groups": []}
    try:
        if not source.is_file():
            result["error_reason"] = "missing_solution"
            return result
        result["solution_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        compiled = subprocess.run(
            ["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)],
            cwd=workspace, capture_output=True, text=True, timeout=60,
        )
        result["compile_exit"] = compiled.returncode
        (evidence / "compile.log").write_text(compiled.stderr)
        if compiled.returncode:
            result["error_reason"] = "compile_failed"
            return result
        for name, cases in _cases(workload, seed).items():
            lines, expected = [str(len(cases))], []
            for case_lines, answers in cases:
                lines.extend(case_lines)
                expected.extend(answers)
            input_text = "\n".join(lines) + "\n"
            (evidence / f"{name}.input").write_text(input_text)
            (evidence / f"{name}.expected").write_text("\n".join(expected))
            tick = time.perf_counter()
            try:
                ran = subprocess.run([str(binary)], cwd=workspace, input=input_text,
                                     capture_output=True, text=True, timeout=60)
                passed = ran.returncode == 0 and ran.stdout.split() == expected
                (evidence / f"{name}.actual").write_text(ran.stdout)
                row = {"exit": ran.returncode, "status": "passed" if passed else "failed"}
            except subprocess.TimeoutExpired:
                row = {"exit": None, "status": "failed", "error_reason": "oracle_timeout"}
            result["groups"].append({"name": name, "cases": len(cases), "answers": len(expected),
                                     "elapsed_ms": (time.perf_counter() - tick) * 1000, **row})
        if all(row["status"] == "passed" for row in result["groups"]):
            result["status"] = "passed"
        return result
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["error_reason"] = type(exc).__name__
        return result
    finally:
        binary.unlink(missing_ok=True)
        (evidence / "solution-verification.json").write_text(json.dumps(result, indent=2))
