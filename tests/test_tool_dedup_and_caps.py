"""Repeated identical calls, and oversized results.

Both are context hygiene, and both are deliberately SMALL. Measured on real runs:
75-94% of context is cache-hit and a whole run costs ~$0.0013, so transcript
re-transmission is not where the money goes. The measured lever was tool
truthfulness, not context engineering — these two are guards, not the cure.

The one rule they share with every other fix in this file: if the result is not
what the caller literally asked for, SAY SO in the result the model reads. A
silent cache hit and a silent truncation are both lies the model cannot detect.
"""
import asyncio

import pytest

from open_agent_sdk.engine import MAX_TOOL_RESULT_CHARS, QueryEngine, QueryEngineConfig
from open_agent_sdk.types import BaseTool, ToolContext, ToolResult


class CountingTool(BaseTool):
    """Read-only by default; counts how many times it actually executed."""

    def __init__(self, name="search", read_only=True, payload="result-body", error=False):
        # BaseTool exposes `name`/`description` as read-only properties over
        # `_name`/`_description`; assigning the public names raises.
        self._name = name
        self._description = "test tool"
        self.calls = 0
        self._read_only = read_only
        self._payload = payload
        self._error = error

    def get_input_schema(self):
        return {"type": "object", "properties": {}}

    def is_read_only(self, input=None):
        return self._read_only

    def is_concurrency_safe(self, input=None):
        return True

    async def call(self, input, context):
        self.calls += 1
        return ToolResult(tool_use_id="", content=self._payload, is_error=self._error)


def engine_with(tool):
    eng = QueryEngine(QueryEngineConfig(model="m", tools=[tool]))
    return eng


def run_tool(eng, name, inp, tool_use_id="t1"):
    return asyncio.run(
        eng._execute_single_tool(
            {"id": tool_use_id, "name": name, "input": inp}, ToolContext(cwd=".")
        )
    )


# ---------------------------------------------------------------------------
# dedup
# ---------------------------------------------------------------------------

def test_identical_read_only_call_is_not_re_executed():
    t = CountingTool()
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x", "path": "src"})
    second = run_tool(eng, "search", {"pattern": "x", "path": "src"})
    assert t.calls == 1, "the repeat re-executed"
    assert "result-body" in second.content


def test_the_repeat_is_ANNOUNCED_not_served_silently():
    """A silent cache hit looks like a fresh search that happened to agree, so
    the model learns nothing and may repeat again."""
    t = CountingTool()
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x"})
    second = run_tool(eng, "search", {"pattern": "x"})
    assert "already made this run" in second.content


def test_argument_ORDER_does_not_defeat_the_cache():
    """Same call, different key order, must collide — hence sort_keys."""
    t = CountingTool()
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x", "path": "src"})
    run_tool(eng, "search", {"path": "src", "pattern": "x"})
    assert t.calls == 1


def test_a_different_argument_really_does_re_execute():
    t = CountingTool()
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x"})
    run_tool(eng, "search", {"pattern": "y"})
    assert t.calls == 2


def test_mutating_tools_are_never_deduped():
    """Re-running a Write or a Bash is the caller's INTENT. Suppressing the second
    one would silently change behaviour — far worse than a wasted call."""
    t = CountingTool(name="bash", read_only=False)
    eng = engine_with(t)
    run_tool(eng, "bash", {"command": "echo hi"})
    run_tool(eng, "bash", {"command": "echo hi"})
    assert t.calls == 2


def test_errors_are_not_cached():
    """A failure is often transient; pinning it for the whole run would turn one
    bad moment into a permanent dead end."""
    t = CountingTool(error=True)
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x"})
    run_tool(eng, "search", {"pattern": "x"})
    assert t.calls == 2


def test_two_engines_do_not_share_a_cache():
    """The cache is per RUN. Leaking across runs would serve a stale view of a
    tree that has since been edited."""
    t = CountingTool()
    run_tool(engine_with(t), "search", {"pattern": "x"})
    run_tool(engine_with(t), "search", {"pattern": "x"})
    assert t.calls == 2


def test_unserialisable_input_re_executes_rather_than_mis_keying():
    t = CountingTool()
    eng = engine_with(t)
    run_tool(eng, "search", {"bad": object()})
    run_tool(eng, "search", {"bad": object()})
    assert t.calls == 2, "keyed on a repr instead of a value"


def test_the_cached_result_keeps_the_callers_tool_use_id():
    """A tool_result carrying the FIRST call's id would not answer the second
    tool_use block, and the API rejects the turn."""
    t = CountingTool()
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x"}, tool_use_id="first")
    second = run_tool(eng, "search", {"pattern": "x"}, tool_use_id="second")
    assert second.tool_use_id == "second"


# ---------------------------------------------------------------------------
# caps
# ---------------------------------------------------------------------------

def test_a_large_result_is_truncated():
    t = CountingTool(payload="z" * (MAX_TOOL_RESULT_CHARS + 5_000))
    r = run_tool(engine_with(t), "search", {"pattern": "x"})
    assert len(r.content) < MAX_TOOL_RESULT_CHARS + 500


def test_truncation_says_what_it_dropped():
    """A cap the model cannot see makes it read part of a file as the whole file."""
    over = 5_000
    t = CountingTool(payload="z" * (MAX_TOOL_RESULT_CHARS + over))
    r = run_tool(engine_with(t), "search", {"pattern": "x"})
    assert "truncated" in r.content
    assert str(over) in r.content, "the dropped amount is not stated"
    assert "search" in r.content, "the tool is not named"


def test_a_result_under_the_cap_is_untouched():
    t = CountingTool(payload="small body")
    r = run_tool(engine_with(t), "search", {"pattern": "x"})
    assert r.content == "small body"


def test_a_result_exactly_at_the_cap_is_untouched():
    t = CountingTool(payload="z" * MAX_TOOL_RESULT_CHARS)
    r = run_tool(engine_with(t), "search", {"pattern": "x"})
    assert "truncated" not in r.content


def test_non_string_content_survives_the_cap():
    """Structured content must not be stringified on its way through."""
    class Structured(CountingTool):
        async def call(self, input, context):
            return ToolResult(tool_use_id="", content=[{"type": "text", "text": "hi"}])

    r = run_tool(engine_with(Structured()), "search", {"pattern": "x"})
    assert isinstance(r.content, list)


def test_what_is_cached_is_the_capped_form():
    """Otherwise the cap is undone on every repeat."""
    t = CountingTool(payload="z" * (MAX_TOOL_RESULT_CHARS + 9_000))
    eng = engine_with(t)
    run_tool(eng, "search", {"pattern": "x"})
    second = run_tool(eng, "search", {"pattern": "x"})
    assert len(second.content) < MAX_TOOL_RESULT_CHARS + 700
