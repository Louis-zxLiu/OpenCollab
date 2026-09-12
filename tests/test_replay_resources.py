"""Real Git restores must reproduce bytes, environment and identity through the adapter."""

from pathlib import Path

from scheduler_awaiting_test_support import terminal
from test_container_worktree_env import _git, _repo
from test_history_application import live_scheduler

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.domain.rollback import EnvironmentSnapshot


async def test_real_git_historical_replay_restores_file_environment_and_pwd(tmp_path):
    repo = _repo(tmp_path / "repo")
    (repo / "value.txt").write_text("before")
    environment = LocalEnvironment(str(repo))
    environment.replace_environment(EnvironmentSnapshot.from_mapping({"PWD": str(repo), "VALUE": "before"}))

    async def mutate(session):
        await environment.write_file("value.txt", "after")
        environment.replace_environment(EnvironmentSnapshot.from_mapping({"PWD": str(repo), "VALUE": "after"}))
        return await terminal("first")(session)

    async def verify(session):
        assert (repo / "value.txt").read_text() == "before"
        assert environment.snapshot_environment().as_dict() == {"PWD": str(repo), "VALUE": "before"}
        return await terminal("restored")(session)

    scheduler, lead = live_scheduler(mutate, verify, environment=environment)
    try:
        await scheduler.run("task")
        cp = next(e for e in scheduler.history_list() if e.boundary == "initial")
        assert (repo / "value.txt").read_text() == "after"
        replay = await scheduler.replay_from(cp.checkpoint_id, "retry", 1024)
        assert replay.status == "started"
        assert await replay.task == "restored"
        assert Path(environment.workspace) == repo
        assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/opencollab/checkpoints")
    finally:
        await environment.cleanup()
