"""Real Git resource ownership at the public live Team close boundary."""

import pytest
from test_sdk_team_environment import _git, _repo

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.bootstrap import container
from opencollab.sdk import OpenCollab


class _AnswerLLM:
    def __init__(self, **kwargs):
        pass

    def context_window(self):
        return 200_000

    async def complete(self, messages, **kwargs):
        return LLMResponse(content="done", usage=Usage(input_tokens=10, output_tokens=5))

    async def close(self):
        pass


@pytest.mark.parametrize("borrowed", [False, True])
@pytest.mark.parametrize("fail_once", [False, True])
async def test_live_handle_closes_owned_lead_refs_and_preserves_borrowed_scope(
    tmp_path, monkeypatch, borrowed, fail_once,
):
    monkeypatch.setattr(container, "LLMClient", _AnswerLLM)
    repo = _repo(tmp_path / "repo", "initial")
    config = tmp_path / "team.yaml"
    config.write_text("entry: lead\nroles:\n  lead:\n    prompt: Reply done.\n    tools: []\ntopology: {}\n")
    environment = LocalEnvironment(str(repo)) if borrowed else None
    # The provider is replaced by _AnswerLLM; this credential is a test fixture.
    client_config = {
        "model": "test", "provider": "openai",
        "api_key": "test-key",  # pragma: allowlist secret
    }
    handle = await OpenCollab(repo, config=client_config, environment=environment).start_team(
        "Reply done.", config=config,
    )
    lead_env = handle._scheduler.lead_session.env
    original_cleanup = lead_env.cleanup
    calls = 0

    async def transient_failure():
        nonlocal calls
        calls += 1
        if fail_once and calls == 1:
            raise OSError("injected lead cleanup failure")
        await original_cleanup()

    monkeypatch.setattr(lead_env, "cleanup", transient_failure)
    try:
        final = await handle.wait()
        assert final.ok
        assert _git(repo, "for-each-ref", "--format=%(refname)", "refs/opencollab/checkpoints")
        if fail_once and not borrowed:
            with pytest.raises(RuntimeError):
                await handle.close()
        await handle.close()
        await handle.close()
        refs = _git(repo, "for-each-ref", "--format=%(refname)", "refs/opencollab/checkpoints")
        assert bool(refs) is borrowed
        assert lead_env.revoked is not borrowed
        assert calls == (0 if borrowed else 2 if fail_once else 1)
        assert (repo / "which.txt").read_text() == "initial\n"
    finally:
        monkeypatch.setattr(lead_env, "cleanup", original_cleanup)
        await original_cleanup()
