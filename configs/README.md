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

The final resolved configuration is validated by a Pydantic model. `budget`
must be a positive integer; blank `api_key` and `base_url` values are treated as
unset.

Do not commit `configs/.env` or any file containing real API keys.
