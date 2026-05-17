# Configs

Runtime configuration lives in this directory.

Create `configs/.env` from the example:

```bash
cp configs/.env.example configs/.env
```

OpenCollab loads config in this order:

1. Process environment variables
2. `configs/.env`
3. Legacy `.env`
4. Built-in defaults

Use `OPENCOLLAB_CONFIG_FILE=/path/to/file.env` to point OpenCollab at a specific
env file.

## Model Settings

OpenCollab supports OpenAI-compatible APIs through the OpenAI client path. Set
`provider=openai` and a compatible `base_url` for those providers.

Environment variable example:

```bash
export OPENCOLLAB_PROVIDER=openai
export OPENCOLLAB_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENCOLLAB_MODEL=glm-5.1
export OPENCOLLAB_API_KEY=<your-api-key>
```

Equivalent `configs/.env` values:

```dotenv
OPENCOLLAB_PROVIDER=openai
OPENCOLLAB_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENCOLLAB_MODEL=glm-5.1
OPENCOLLAB_API_KEY=<your-api-key>
```

## Validation

The final resolved configuration is validated by a Pydantic model. `budget`
must be a positive integer; blank `api_key` and `base_url` values are treated as
unset.

Do not commit `configs/.env` or any file containing real API keys.
