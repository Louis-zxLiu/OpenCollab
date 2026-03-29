"""MCP Client — dynamic tool discovery via Model Context Protocol.

The escape hatch: instead of writing integrations for git, github, databases, etc.,
connect to MCP servers and auto-import their tools.

Ref:
- opencode: mcp/index.ts — StdioClientTransport, tool conversion to AI SDK format
- kimi-cli: KimiToolset.load_mcp_tools() with background loading
- Design doc: connect_mcp_server() → auto-map to JSON Schema tools
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Awaitable

from opencollab.tools.base import Tool
from opencollab.core.env import Environment

logger = logging.getLogger(__name__)


class MCPTool(Tool):
    """A tool backed by a remote MCP server. Auto-generated from MCP tool definitions."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        server_command: str,
        server_args: list[str],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self._server_command = server_command
        self._server_args = server_args
        self._process: asyncio.subprocess.Process | None = None
        self._stdin = None
        self._stdout = None
        self._request_id = 0

    async def _ensure_connection(self) -> None:
        """Start the MCP server process if not running."""
        if self._process and self._process.returncode is None:
            return

        self._process = await asyncio.create_subprocess_exec(
            self._server_command, *self._server_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _jsonrpc_call(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and read response."""
        await self._ensure_connection()

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }
        msg = json.dumps(request)
        content = f"Content-Length: {len(msg)}\r\n\r\n{msg}"

        self._process.stdin.write(content.encode())
        await self._process.stdin.drain()

        # Read response (Content-Length header + body)
        header = await self._process.stdout.readline()
        await self._process.stdout.readline()  # empty line
        length = int(header.decode().split(":")[1].strip())
        body = await self._process.stdout.readexactly(length)
        return json.loads(body.decode())

    async def execute(
        self,
        params: dict[str, Any],
        env: Environment | None = None,
        confirm_fn: Callable[[str], Awaitable[bool]] | None = None,
    ) -> str:
        try:
            resp = await self._jsonrpc_call("tools/call", {"name": self.name, "arguments": params})
            if "result" in resp:
                result = resp["result"]
                if isinstance(result, dict) and "content" in result:
                    # MCP tool result format
                    parts = []
                    for c in result["content"]:
                        if c.get("type") == "text":
                            parts.append(c["text"])
                    return "\n".join(parts) if parts else json.dumps(result)
                return json.dumps(result)
            elif "error" in resp:
                return f"MCP error: {resp['error']}"
            return json.dumps(resp)
        except Exception as e:
            return f"MCP tool execution error: {type(e).__name__}: {e}"

    async def close(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()


async def load_mcp_tools(command: str, args: list[str]) -> list[MCPTool]:
    """Connect to an MCP server and discover its tools.

    Usage:
        tools = await load_mcp_tools("npx", ["-y", "@modelcontextprotocol/server-github"])

    Returns a list of MCPTool instances ready to be added to an Agent.
    """
    # Start server process
    proc = await asyncio.create_subprocess_exec(
        command, *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        # Initialize MCP connection
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "opencollab", "version": "0.1.0"},
            },
        }
        msg = json.dumps(init_request)
        content = f"Content-Length: {len(msg)}\r\n\r\n{msg}"
        proc.stdin.write(content.encode())
        await proc.stdin.drain()

        # Read init response
        header = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
        await proc.stdout.readline()
        length = int(header.decode().split(":")[1].strip())
        body = await proc.stdout.readexactly(length)
        _init_resp = json.loads(body.decode())

        # Send initialized notification
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        notif_msg = json.dumps(notif)
        notif_content = f"Content-Length: {len(notif_msg)}\r\n\r\n{notif_msg}"
        proc.stdin.write(notif_content.encode())
        await proc.stdin.drain()

        # List tools
        list_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        list_msg = json.dumps(list_request)
        list_content = f"Content-Length: {len(list_msg)}\r\n\r\n{list_msg}"
        proc.stdin.write(list_content.encode())
        await proc.stdin.drain()

        header = await asyncio.wait_for(proc.stdout.readline(), timeout=30)
        await proc.stdout.readline()
        length = int(header.decode().split(":")[1].strip())
        body = await proc.stdout.readexactly(length)
        tools_resp = json.loads(body.decode())

        # Parse tool definitions
        tools: list[MCPTool] = []
        if "result" in tools_resp and "tools" in tools_resp["result"]:
            for td in tools_resp["result"]["tools"]:
                tools.append(MCPTool(
                    name=td["name"],
                    description=td.get("description", ""),
                    parameters=td.get("inputSchema", {"type": "object", "properties": {}}),
                    server_command=command,
                    server_args=args,
                ))
                logger.info(f"Loaded MCP tool: {td['name']}")

        return tools

    except asyncio.TimeoutError:
        logger.error(f"MCP server {command} timed out during initialization")
        return []
    except Exception as e:
        logger.error(f"Failed to load MCP tools from {command}: {e}")
        return []
    finally:
        proc.terminate()
        await proc.wait()
