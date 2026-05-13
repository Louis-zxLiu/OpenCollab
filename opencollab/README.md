# OpenCollab

Minimal multi-agent software development framework.

## Quick Start

```bash
pip install -e .
opencollab team
```

From the repository root with `uv`:

```bash
uv venv .venv
uv pip install -e opencollab
.venv/bin/opencollab team --workspace .
```

## Model Configuration

OpenCollab supports OpenAI-compatible APIs through the OpenAI client path.
Configure the provider, base URL, model, and API key with environment variables
or a `.env` file.

Runtime config should live in the repository-level `configs/` directory:

```bash
cp configs/.env.example configs/.env
```

DashScope compatible mode example:

```bash
export OPENCOLLAB_PROVIDER=openai
export OPENCOLLAB_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENCOLLAB_MODEL=glm-5.1
export OPENCOLLAB_API_KEY=<your-api-key>
```

Do not commit real API keys.

Config resolution order:

1. Process environment variables
2. `configs/.env`
3. Legacy `.env`
4. Built-in defaults

Set `OPENCOLLAB_CONFIG_FILE=/path/to/file.env` to use a specific config file.
The final config is validated by Pydantic before it is used.

## Commands

From the repository root, use the launcher:

```bash
scripts/start_opencollab.sh
```

It creates `.venv` when needed, checks `configs/.env`, and starts team mode by
default. Use `scripts/start_opencollab.sh chat` for single-agent chat mode.

Interactive chat:

```bash
opencollab chat --workspace .
```

Interactive team mode:

```bash
opencollab team --workspace .
```

Headless eval harness:

```bash
opencollab eval tasks.jsonl --output eval_results --concurrency 1
```

Task files are JSONL. Each line describes one task:

```json
{"task_id":"example","description":"Fix the bug described here.","repo_path":"/path/to/repo","timeout":600,"max_tokens":100000}
```

The eval harness writes a summary JSONL file and trajectory logs under the
output directory.
