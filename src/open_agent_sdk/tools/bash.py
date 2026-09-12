"""Bash tool - execute shell commands."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from open_agent_sdk.types import BaseTool, ToolContext, ToolInputSchema, ToolResult


class BashTool(BaseTool):
    """Execute shell commands and return stdout/stderr."""

    _name = "Bash"
    _description = (
        "Executes a given bash command and returns its output. "
        "The working directory persists between commands."
    )
    _input_schema = ToolInputSchema(
        properties={
            "command": {
                "type": "string",
                "description": "The command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Optional timeout in milliseconds (max 600000)",
            },
            "description": {
                "type": "string",
                "description": "Clear, concise description of what this command does",
            },
        },
        required=["command"],
    )

    def is_read_only(self, input: dict[str, Any] | None = None) -> bool:
        if input:
            cmd = input.get("command", "")
            read_only_prefixes = [
                "ls", "cat", "head", "tail", "grep", "find", "which", "whoami",
                "pwd", "echo", "date", "git status", "git log", "git diff",
                "git branch", "git show", "git remote",
            ]
            cmd_stripped = cmd.strip()
            for prefix in read_only_prefixes:
                if cmd_stripped.startswith(prefix):
                    return True
        return False

    def is_concurrency_safe(self, input: dict[str, Any] | None = None) -> bool:
        return False

    async def call(self, input: dict[str, Any], context: ToolContext) -> ToolResult:
        command = input.get("command", "")
        # BASH_DEFAULT_TIMEOUT_MS / BASH_MAX_TIMEOUT_MS were being SET by a
        # consumer and read by nobody. Its runner exports both at 2,400,000 ms
        # with a help string reading "The SDK's own default is 600000, which is
        # shorter than a cold cargo test sweep" — so the operator raised the
        # limit deliberately, and every Bash call was silently capped at the
        # hardcoded 600,000 ms anyway. A cold build that needed 12 minutes was
        # killed at 10 and reported as a timeout, with nothing indicating the
        # configured budget had been ignored.
        #
        # context.env first (per-run, what the SDK is handed), then the process
        # environment, then the historical defaults.
        env = getattr(context, "env", None) or {}

        def _env_int(name: str, fallback: int) -> int:
            raw = env.get(name) or os.environ.get(name)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return fallback
            return value if value > 0 else fallback

        default_timeout_ms = _env_int("BASH_DEFAULT_TIMEOUT_MS", 120000)
        max_timeout_ms = _env_int("BASH_MAX_TIMEOUT_MS", 600000)
        # A default above the ceiling is a misconfiguration, not a licence to
        # exceed it; clamp rather than silently preferring the larger number.
        default_timeout_ms = min(default_timeout_ms, max_timeout_ms)

        requested_ms = input.get("timeout", default_timeout_ms)
        timeout_ms = requested_ms

        if not command:
            return ToolResult(
                tool_use_id="",
                content="Error: command is required",
                is_error=True,
            )

        # Capping is fine; capping SILENTLY is not. A caller that asked for 15
        # minutes and got 10 cannot tell whether its timeout was honoured, and
        # will read the eventual timeout as "the command is slow" rather than
        # "my limit was overridden".
        capped_notice = ""
        if isinstance(requested_ms, (int, float)) and requested_ms > max_timeout_ms:
            capped_notice = (
                f"\n[timeout reduced from {int(requested_ms)}ms to the "
                f"{max_timeout_ms}ms limit]"
            )
            timeout_ms = max_timeout_ms
        timeout_s = timeout_ms / 1000.0

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=context.cwd,
                env={**dict(__import__("os").environ), **context.env} if context.env else None,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ToolResult(
                    tool_use_id="",
                    content=f"Command timed out after {timeout_ms}ms{capped_notice}",
                    is_error=True,
                )

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            # Truncate large output
            max_output = 100 * 1024  # 100KB
            if len(stdout_str) > max_output:
                stdout_str = stdout_str[:max_output] + f"\n... [truncated, total {len(stdout_str)} bytes]"
            if len(stderr_str) > max_output:
                stderr_str = stderr_str[:max_output] + f"\n... [truncated, total {len(stderr_str)} bytes]"

            output = ""
            if stdout_str:
                output += stdout_str
            if stderr_str:
                if output:
                    output += "\n"
                output += stderr_str

            # `proc.returncode or 0` turned None into 0, so a process that never
            # terminated normally was reported as a SUCCESS with no output.
            if proc.returncode is None:
                output += "\n(process did not report an exit status)"
                return ToolResult(
                    tool_use_id="",
                    content=output + capped_notice,
                    is_error=True,
                )
            exit_code = proc.returncode
            if exit_code != 0:
                output += f"\n(exit code: {exit_code})"

            return ToolResult(
                tool_use_id="",
                content=(output if output else "(no output)") + capped_notice,
                is_error=exit_code != 0,
            )

        except Exception as e:
            return ToolResult(
                tool_use_id="",
                content=f"Error executing command: {e}",
                is_error=True,
            )
