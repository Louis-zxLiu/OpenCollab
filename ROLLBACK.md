# Unified Causal Rollback

Rollback tracks tool results, child results, and teammate messages as causal
effects. Invalidating an effect quarantines all known descendants and restores
each affected Agent Scope to the earliest relevant checkpoint. The coordinating
Agent remains runnable and dispatches the corrected attempt.

An Agent Scope owns one isolated worktree and one exact environment mapping.
Checkpoints protect initialization, result/message consumption, and ordinary
tool execution. Restore replaces files and variables together, increments the
epoch, and fences stale model or tool output. `invalidate_effect` is not itself
checkpointed.

`set_env` and `unset_env` persist Scope changes; shell `export` remains local to
that command. `list_env` redacts sensitive names. Checkpoint state is in memory
by default; optional AEAD persistence uses `OPENCOLLAB_ROLLBACK_KEY` without
storing keys or plaintext secrets in snapshots or traces.

Lead work is published only after source fingerprint validation and guarded
patch application, including untracked files. Races or conflicts retain the
worktree and patch artifact. External databases, services, packages, network
connections, and other state outside the Scope are unsupported.
