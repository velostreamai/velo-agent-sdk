"""Tool-call arguments must always reach a tool as a dict.

A tool-use block's `input` is typed `dict[str, Any]` and was not enforced. Two
paths handed a `str` downstream, and the first tool method to call `.get()` on
it died with `AttributeError: 'str' object has no attribute 'get'` — killing the
run mid-flight, after arbitrary spend, with nothing committed. One real dispatch
burned 9.9M input tokens that way.

The silent path is the one worth testing: with DOUBLE-ENCODED arguments
`json.loads` SUCCEEDS and returns a string, so no exception is raised and
nothing downstream notices.
"""
import json

import pytest

from open_agent_sdk.providers.openai_provider import (
    OpenAIProvider,
    _normalise_tool_input,
)


def _response(arguments):
    """A minimal OpenAI-shaped response carrying one tool call."""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": arguments},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def _tool_inputs(resp):
    out = []
    for block in getattr(resp, "content", None) or []:
        b = block if isinstance(block, dict) else getattr(block, "__dict__", {})
        if b.get("type") == "tool_use":
            out.append(b.get("input"))
    return out


# --- the unit the fix hinges on ------------------------------------------------

def test_plain_json_object_decodes():
    assert _normalise_tool_input('{"command": "ls"}') == {"command": "ls"}


def test_double_encoded_decodes_to_dict():
    """The silent path: json.loads succeeds and returns a STRING."""
    once = json.dumps({"command": "ls -la"})
    twice = json.dumps(once)
    assert isinstance(json.loads(twice), str)          # the trap, made explicit
    assert _normalise_tool_input(twice) == {"command": "ls -la"}


def test_dict_passes_through_untouched():
    d = {"command": "ls"}
    assert _normalise_tool_input(d) is d


@pytest.mark.parametrize("raw", [None, "", "{}", "not json at all", "[1,2,3]", "42"])
def test_anything_that_is_not_an_object_fails_closed(raw):
    """Fail CLOSED. A tool given {} does nothing; a tool given a half-decoded
    string is how a shell command gets misread."""
    assert _normalise_tool_input(raw) == {}


def test_decoding_is_bounded():
    """A pathological nesting must terminate rather than spin."""
    v = json.dumps({"command": "ls"})
    for _ in range(20):
        v = json.dumps(v)
    assert _normalise_tool_input(v) == {}      # bounded: gives up, does not hang


# --- and through the real provider path ---------------------------------------

def test_provider_never_emits_a_string_input():
    p = OpenAIProvider.__new__(OpenAIProvider)
    for arguments in (
        json.dumps({"command": "ls"}),                    # normal
        json.dumps(json.dumps({"command": "ls"})),        # double-encoded
        "definitely not json",                            # the old except branch
        "",
    ):
        for value in _tool_inputs(p._convert_response(_response(arguments))):
            assert isinstance(value, dict), f"{arguments!r} produced {type(value).__name__}"
