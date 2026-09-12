"""Bash honours the configured timeout, and says when it overrides one.

Found auditing every tool for silent degradation (#8), which is the class that
already cost 12->4 tool calls in GrepTool (#6).

`BASH_DEFAULT_TIMEOUT_MS` and `BASH_MAX_TIMEOUT_MS` were being SET by a consumer
and read by nobody. Its runner exports both at 2,400,000 ms, with a help string
reading "The SDK's own default is 600000, which is shorter than a cold cargo test
sweep" — so the operator raised the limit deliberately, and every Bash call was
capped at the hardcoded 600,000 ms regardless. A cold build needing 12 minutes
was killed at 10 and reported as an ordinary timeout, with nothing anywhere
indicating the configured budget had been ignored.

Three defects, one shape: the tool decided something the caller had already
decided, and did not say so.
"""
import asyncio

import pytest

from open_agent_sdk.tools.bash import BashTool
from open_agent_sdk.types import ToolContext


def run(inp, env=None):
    return asyncio.run(BashTool().call(inp, ToolContext(cwd=".", env=env or {})))


# ---------------------------------------------------------------------------
# the configured budget is honoured
# ---------------------------------------------------------------------------

def test_env_default_is_used_when_no_timeout_given():
    """The consumer's 2,400,000ms was ignored in favour of a hardcoded 120,000."""
    r = run({"command": "echo hi"}, env={"BASH_DEFAULT_TIMEOUT_MS": "2400000",
                                         "BASH_MAX_TIMEOUT_MS": "2400000"})
    assert not r.is_error
    assert "hi" in r.content
    assert "reduced" not in r.content, "a within-budget call was capped"


def test_a_request_inside_the_raised_ceiling_is_not_capped():
    """900,000ms is over the OLD hardcoded 600,000 cap but inside a configured
    2,400,000 — this is precisely the cold-cargo-sweep case."""
    r = run({"command": "echo hi", "timeout": 900000},
            env={"BASH_MAX_TIMEOUT_MS": "2400000"})
    assert "reduced" not in r.content, "the configured ceiling was ignored"


def test_process_env_is_honoured_when_context_env_is_empty(monkeypatch):
    monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", "2400000")
    r = run({"command": "echo hi", "timeout": 900000})
    assert "reduced" not in r.content


def test_context_env_wins_over_process_env(monkeypatch):
    """Per-run configuration must beat whatever the daemon was started with."""
    monkeypatch.setenv("BASH_MAX_TIMEOUT_MS", "2400000")
    r = run({"command": "echo hi", "timeout": 900000},
            env={"BASH_MAX_TIMEOUT_MS": "600000"})
    assert "reduced" in r.content


# ---------------------------------------------------------------------------
# capping is allowed; capping silently is not
# ---------------------------------------------------------------------------

def test_exceeding_the_ceiling_is_announced():
    r = run({"command": "echo hi", "timeout": 900000},
            env={"BASH_MAX_TIMEOUT_MS": "600000"})
    assert "reduced from 900000ms" in r.content
    assert "600000ms" in r.content


def test_the_default_ceiling_still_applies_with_no_config():
    """No env anywhere must behave as before — 600,000ms."""
    r = run({"command": "echo hi", "timeout": 900000})
    assert "reduced" in r.content and "600000ms" in r.content


@pytest.mark.parametrize("bad", ["", "abc", "0", "-5", None])
def test_nonsense_config_falls_back_rather_than_breaking_bash(bad):
    """A bad env value must not make every command unrunnable, and must not be
    read as 'no timeout'."""
    env = {} if bad is None else {"BASH_MAX_TIMEOUT_MS": bad}
    r = run({"command": "echo hi", "timeout": 900000}, env=env)
    assert not r.is_error
    assert "reduced" in r.content, "fell back to something other than 600000"


def test_a_default_above_the_ceiling_does_not_raise_the_ceiling():
    """A misconfiguration must clamp, not silently grant more than the max."""
    r = run({"command": "echo hi"},
            env={"BASH_DEFAULT_TIMEOUT_MS": "2400000", "BASH_MAX_TIMEOUT_MS": "60000"})
    assert not r.is_error


# ---------------------------------------------------------------------------
# an exit status that never arrived is not a success
# ---------------------------------------------------------------------------

def test_ordinary_failure_still_reports_its_exit_code():
    r = run({"command": "exit 3"})
    assert r.is_error
    assert "exit code: 3" in r.content


def test_success_is_not_marked_an_error():
    r = run({"command": "echo ok"})
    assert not r.is_error and "ok" in r.content


def test_missing_exit_status_is_an_error_not_a_silent_success():
    """`proc.returncode or 0` turned None into 0, so a process that never
    terminated normally was reported as a SUCCESS with no output."""
    class NoStatus:
        returncode = None
        async def communicate(self):
            return b"partial output", b""

    async def go():
        import open_agent_sdk.tools.bash as B
        orig = B.asyncio.create_subprocess_shell

        async def fake(*a, **k):
            return NoStatus()

        B.asyncio.create_subprocess_shell = fake
        try:
            return await BashTool().call({"command": "x"}, ToolContext(cwd=".", env={}))
        finally:
            B.asyncio.create_subprocess_shell = orig

    r = asyncio.run(go())
    assert r.is_error, "a process with no exit status was reported as success"
    assert "did not report an exit status" in r.content
    assert "partial output" in r.content, "output collected before the failure was dropped"
