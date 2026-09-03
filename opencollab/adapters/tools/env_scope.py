"""Tools for modifying and inspecting the current Agent Scope."""

from __future__ import annotations

from typing import Any

from opencollab.adapters.tools.base import Tool

_SENSITIVE = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def _environment(runtime: Any) -> Any:
    environment = getattr(runtime, "environment", None)
    if environment is None:
        return None
    return environment


class SetEnvTool(Tool):
    name = "set_env"
    description = "Set a persistent environment variable in this Agent Scope."
    parameters = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
        "required": ["name", "value"],
    }

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        environment = _environment(runtime)
        if environment is None:
            return "Error: no Agent Scope is available."
        try:
            environment.set_environment_variable(params.get("name"), params.get("value"))
        except (TypeError, ValueError) as exc:
            return f"Error: {exc}"
        return "Environment variable updated."

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": SetEnvTool.name,
                "description": SetEnvTool.description,
                "parameters": SetEnvTool.parameters,
            },
        }


class UnsetEnvTool(Tool):
    name = "unset_env"
    description = "Remove a persistent environment variable from this Agent Scope."
    parameters = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        environment = _environment(runtime)
        if environment is None:
            return "Error: no Agent Scope is available."
        try:
            environment.unset_environment_variable(params.get("name"))
        except (TypeError, ValueError) as exc:
            return f"Error: {exc}"
        return "Environment variable removed."

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": UnsetEnvTool.name,
                "description": UnsetEnvTool.description,
                "parameters": UnsetEnvTool.parameters,
            },
        }


class ListEnvTool(Tool):
    name = "list_env"
    description = "List the current Agent Scope environment with sensitive values redacted."
    parameters = {"type": "object", "properties": {}}

    async def execute_with_runtime(self, params: dict[str, Any], runtime: Any) -> str:
        del params
        environment = _environment(runtime)
        if environment is None:
            return "Error: no Agent Scope is available."
        rows = []
        for name, value in sorted(environment.environment_view().items()):
            rendered = "[REDACTED]" if any(part in name.upper() for part in _SENSITIVE) else value
            rows.append(f"{name}={rendered}")
        return "\n".join(rows)

    @staticmethod
    def get_definition() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": ListEnvTool.name,
                "description": ListEnvTool.description,
                "parameters": ListEnvTool.parameters,
            },
        }
