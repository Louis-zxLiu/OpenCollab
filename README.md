# OpenCollab

OpenCollab is a minimal multi-agent software development framework with chat,
team, and headless evaluation harnesses.

## Quick Start

Create an environment and install the package:

```bash
uv venv .venv
uv pip install -e opencollab
```

Or with pip from inside the package directory:

```bash
cd opencollab
pip install -e .
```

## Configure a Model

OpenCollab can use OpenAI-compatible APIs by setting `provider=openai` and a
compatible `base_url`.

Runtime configuration should live under `configs/`:

```bash
cp configs/.env.example configs/.env
```

For DashScope compatible mode:

```bash
export OPENCOLLAB_PROVIDER=openai
export OPENCOLLAB_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENCOLLAB_MODEL=glm-5.1
export OPENCOLLAB_API_KEY=<your-api-key>
```

Equivalent `.env` values are also supported:

```dotenv
OPENCOLLAB_PROVIDER=openai
OPENCOLLAB_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENCOLLAB_MODEL=glm-5.1
OPENCOLLAB_API_KEY=<your-api-key>
```

Do not commit real API keys.

OpenCollab loads configuration in this order:

1. Process environment variables
2. `configs/.env`
3. Legacy `.env`
4. Built-in defaults

Set `OPENCOLLAB_CONFIG_FILE=/path/to/file.env` to use a specific config file.
The final config is validated by Pydantic before it is used.

## Run OpenCollab

Launcher script:

```bash
scripts/start_opencollab.sh
```

The launcher creates `.venv` when needed, checks `configs/.env`, and starts
team mode by default. Use `scripts/start_opencollab.sh chat` for single-agent
chat mode.

Interactive single-agent mode:

```bash
.venv/bin/opencollab chat --workspace .
```

Interactive team mode:

```bash
.venv/bin/opencollab team --workspace .
```

Headless eval harness:

```bash
.venv/bin/opencollab eval tasks.jsonl --output eval_results --concurrency 1
```

Each JSONL task should look like:

```json
{"task_id":"example","description":"Fix the bug described here.","repo_path":"/path/to/repo","timeout":600,"max_tokens":100000}
```

The eval harness writes `eval_results/results.jsonl` and trajectory files under
`eval_results/trajectories/`.

## SWE-bench Docker Runner

The repository also includes a SWE-bench Docker runner:

```bash
scripts/run_swe_docker.sh --instance_ids django__django-15400
```

This path builds a `swe-collab` Docker image and controls benchmark containers
through the Docker socket. It is separate from the lightweight local eval
harness above. The SWE-bench implementation lives in `tools/swe_bench/`, and
the shell entrypoint lives in `scripts/`.
