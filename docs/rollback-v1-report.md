# Explicit Effect Rollback v1 Report

## 1. Scope and Baseline

This report records the thin rollback implementation in this Fork. The code
was rebuilt from upstream `main` at commit
`7255895788400bf8a8ccb0f51d9241bc9d06b473` (OpenCollab 0.6.0 release merge).
The previous Fork-specific baseline, workspace-effect adoption, publish path,
and coordination protocol were not carried into this implementation.

The final implementation commit is `66fc53e8f0f21388a61b009a737e05507b759fbf`.
No credentials, private network details, usernames, or infrastructure paths
are included in this report.

## 2. Architecture

The change follows the repository's Clean Architecture rule:

```text
adapters -> application -> domain
bootstrap -> adapters/application composition
```

`opencollab/domain/rollback.py` contains immutable values and pure graph
operations only. `opencollab/application/rollback.py` owns the effect graph,
consumer index, checkpoint registry, plan generation, and explicit restore use
case. `opencollab/application/_scheduler_rollback.py` adds scheduler fencing
without importing Git or Docker. `opencollab/adapters/git_checkpoints.py`
implements physical Git checkpoint and restore operations through a temporary
index.

The domain module has no I/O, asyncio, scheduler, or adapter dependency. The
remote import-linter result confirms the dependency direction and reports no
broken contracts.

## 3. Rollback Model

The causal graph uses `EffectRef` nodes with producer, parent effects, epoch,
attempt, kind, status, and content digest. `compute_descendants()` computes the
invalidated closure. `compute_affected_agents()` includes the invalidated
effect producers and registered consumers, while independent sibling branches
remain outside the affected set.

`RollbackPlan` is side-effect free and contains target effects, invalidated
effects, affected Agents, checkpoint selections, and coordinator information.
The service exposes these operations:

```python
create_checkpoint(aid, boundary, causal_frontier)
preview_rollback(effect_ids)
rollback_to_checkpoint(aid, checkpoint_id)
rollback_effect(effect_ids)
```

`rollback_effect()` invalidates the selected effects and descendants, restores
the newest uncontaminated checkpoint for each affected Agent, and does not
restart Agents. `rollback_to_checkpoint()` restores only the selected Agent.
The scheduler exposes `resume_after_rollback()` as the explicit fence-release
operation for a caller that has reviewed the restored state and wants to retry.

The scheduler sequence is fence, cancel owned tasks, restore Scopes, reset
affected sessions to idle, and wait for explicit resume. Old fenced Agents
cannot create or consume new effect references through the scheduler API.

## 4. Filesystem and Environment Semantics

Git checkpoints are made with a temporary index. This preserves the Agent's
real `HEAD` and index while recording tracked files, untracked files, and
ignored outputs. Restore also uses a temporary index, enumerates the target
tree, and removes only untracked paths not present in that tree. The
`.opencollab` control plane is always excluded from capture and cleanup.

The restore path validates workspace identity and verifies the resulting
filesystem against the checkpoint tree. A mismatch returns a failed
`RestoreResult` rather than claiming success.

Environment state is an immutable sorted mapping with a digest. It belongs to
the Agent Scope and is replaced without modifying the host process's
`os.environ`. New processes launched by the Local environment inherit the
restored Scope mapping. A process already running before rollback cannot have
its inherited environment changed retroactively. Docker retains upstream's
fixed Git identity injection contract; it does not receive the host's full
environment through command-line arguments.

## 5. Verification Evidence

All functional and static commands below were executed on the remote Linux
runner against the final implementation commit
`66fc53e8f0f21388a61b009a737e05507b759fbf`. The final full suite result was:

```text
2611 passed, 2 skipped in 62.60s
```

The focused rollback and container adapter suite result was:

```text
19 passed in 1.67s
```

Static verification results:

| Check | Result |
| --- | --- |
| `ruff check .` | exit 0, all checks passed |
| `lint-imports` | exit 0, 4 contracts kept, 0 broken |
| `python scripts/check_interface_width.py` | exit 0 |
| full `pytest -q` | 2611 passed, 2 skipped |

The skipped tests are upstream environment-dependent tests. They are retained
as skipped by the existing test suite and are not counted as passed.

## 6. Performance Evidence

The benchmark is `tests/benchmark_rollback.py`. It uses temporary Git
repositories and does not modify the project workspace. Values below are
milliseconds measured on the remote Linux runner; they are observations, not
universal performance guarantees.

| Scenario | Inputs | Operation | Median | P95 | Samples |
| --- | --- | --- | ---: | ---: | ---: |
| Small | graph 32, files 8, env 16 | graph plan | 0.078 | 0.087 | 20 |
| Small | graph 32, files 8, env 16 | environment restore | 0.005 | 0.007 | 20 |
| Small | graph 32, files 8, env 16 | filesystem restore | 15.786 | 17.647 | 20 |
| Default | graph 128, files 32, env 64 | graph plan | 0.806 | 1.171 | 20 |
| Default | graph 128, files 32, env 64 | environment restore | 0.014 | 0.017 | 20 |
| Default | graph 128, files 32, env 64 | filesystem restore | 18.543 | 25.930 | 20 |
| Large | graph 512, files 64, env 128 | graph plan | 6.493 | 9.532 | 20 |
| Large | graph 512, files 64, env 128 | environment restore | 0.025 | 0.028 | 20 |
| Large | graph 512, files 64, env 128 | filesystem restore | 21.143 | 26.885 | 20 |

The benchmark reports only counts, timings, and digests. It does not emit
environment variable values.

## 7. Limits and Follow-up Work

- The current v1 is an explicit operator/application API; it does not infer
  when an Agent should roll back and does not force rollback through prompts.
- Retry is intentionally a separate operation after `resume_after_rollback()`;
  automatic restart is outside v1.
- The benchmark covers graph planning, Scope mapping replacement, and local
  Git filesystem restore. It is not a production workload benchmark.
- A running child process must be cancelled and quiesced before filesystem
  restore; rollback cannot rewrite a process environment after process birth.
- A missing or contaminated checkpoint produces a skipped/failed restore result
  and must be handled by the caller as a non-success outcome.

## 8. Publication and Identity

The code and tests described in this report were verified at implementation
commit `66fc53e8f0f21388a61b009a737e05507b759fbf`.

The Fork's protected `main` branch rejected the requested non-fast-forward
update. No force bypass was attempted. The earlier verified commit was
published to the GitHub branch `feat/explicit-effect-rollback-v1`; the existing
history was also preserved by the backup tag
`backup/pre-thin-rollback-v1-20260905-234251`.

The final container adapter commit is published to the same feature branch
after this report update.

The final local checkout, the published feature branch, and the remote Linux
checkout were compared by commit ID and Git tree ID. The remote workspace used
for testing was not treated as a source of unverified results.

## 9. Reproducibility

The following commands are the evidence-producing commands used for the final
verification:

```bash
uv run pytest -q
uv run ruff check .
uv run lint-imports
uv run python scripts/check_interface_width.py
uv run python tests/benchmark_rollback.py
uv run python tests/benchmark_rollback.py --iterations 20 --graph-size 32 --file-count 8 --environment-count 16
uv run python tests/benchmark_rollback.py --iterations 20 --graph-size 512 --file-count 64 --environment-count 128
```

The implementation was published only after the final remote verification and
the local, GitHub, and remote checkout identities were compared.
