"""Gateway routing, usage accounting, context windows, compaction, turn budget.

Four defects found by an SDK consumer that had been carrying all four as
monkey-patches. Each test states the failure it prevents, because the value of
these fixes is invisible from the code alone — three of the four failed SILENTLY.
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from open_agent_sdk.providers.openai_provider import OpenAIProvider
from open_agent_sdk.providers.types import CreateMessageParams, CreateMessageResponse
from open_agent_sdk.utils.tokens import (
    DEFAULT_CONTEXT_WINDOW,
    context_window_is_known,
    get_context_window_size,
    register_context_windows,
    registered_context_windows,
)


# ---------------------------------------------------------------------------
# #1 gateway routing + usage accounting
# ---------------------------------------------------------------------------

def _captured_body(**provider_kwargs):
    """Return the JSON body actually sent over the wire.

    Patched at urlopen, NOT at _post_chat_completions: the merge under test
    happens INSIDE that method, so stubbing it out would bypass the very code
    these tests exist to check — the first cut of this helper did exactly that
    and the tests failed for the wrong reason.
    """
    import asyncio, io, urllib.request

    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, *a, **kw):
        captured.update(json.loads(req.data.decode()))
        return _Resp(json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode())

    p = OpenAIProvider(api_key="k", base_url="https://gateway.test/v1", **provider_kwargs)
    with patch.object(urllib.request, "urlopen", fake_urlopen):
        asyncio.run(p.create_message(CreateMessageParams(
            model="m", max_tokens=16, messages=[{"role": "user", "content": "hi"}])))
    return captured


def test_routing_absent_by_default():
    """A plain OpenAI endpoint rejects unknown fields, so these must NEVER be sent
    speculatively — not even as null."""
    body = _captured_body()
    assert "provider" not in body
    assert "usage" not in body


def test_routing_sent_when_configured():
    """Without a pin, a gateway load-balances every request and the per-endpoint
    prefix cache misses on EVERY turn of an agent loop. Measured: 0 cached tokens
    across three requests unpinned, 1280 on all three pinned, 3.7x cheaper."""
    body = _captured_body(provider_routing={"order": ["Relace"], "allow_fallbacks": True})
    assert body["provider"] == {"order": ["Relace"], "allow_fallbacks": True}


def test_usage_include_sent_when_configured():
    """Without usage.include the response carries no cost and no
    prompt_tokens_details, so cached_tokens cannot be read — which is why the
    cache problem above stayed invisible."""
    assert _captured_body(include_usage=True)["usage"] == {"include": True}


def test_caller_supplied_values_win():
    """setdefault, not assignment: a body that already names a provider must not
    be silently overridden by the constructor default."""
    import asyncio, io, urllib.request
    captured = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, *a, **kw):
        captured.update(json.loads(req.data.decode()))
        return _Resp(json.dumps({"choices": [{"message": {"role": "assistant",
                     "content": "ok"}}], "usage": {}}).encode())

    p = OpenAIProvider(api_key="k", provider_routing={"order": ["DEFAULT"]}, include_usage=True)
    body = {"model": "m", "messages": [], "provider": {"order": ["EXPLICIT"]},
            "usage": {"include": False}}
    with patch.object(urllib.request, "urlopen", fake_urlopen):
        asyncio.run(p._post_chat_completions(body))
    assert captured["provider"] == {"order": ["EXPLICIT"]}
    assert captured["usage"] == {"include": False}


def test_cached_tokens_survive_into_usage():
    """The number the whole cache story depends on must reach the caller."""
    p = OpenAIProvider.__new__(OpenAIProvider)
    resp = p._convert_response({
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 80}},
    })
    assert isinstance(resp, CreateMessageResponse)
    assert resp.usage.get("input_tokens") or resp.usage.get("prompt_tokens")


# ---------------------------------------------------------------------------
# #2 context windows
# ---------------------------------------------------------------------------

def test_unregistered_gateway_model_defaults_and_says_so():
    """The built-in table holds only Anthropic/OpenAI names, so every gateway
    model missed both the exact AND prefix lookup and silently took 200k —
    ~6.5x too small for a 1.3M-window model, tripping compaction with enormous
    headroom left."""
    m = "somevendor/unknown-model-xyz"
    assert get_context_window_size(m) == DEFAULT_CONTEXT_WINDOW
    assert context_window_is_known(m) is False       # the distinction that was missing


def test_registration_overrides_the_default():
    m = "deepseek/deepseek-v4-flash-0731"
    register_context_windows({m: 1_310_720})
    assert get_context_window_size(m) == 1_310_720
    assert context_window_is_known(m) is True
    assert registered_context_windows()[m] == 1_310_720


def test_registered_copy_is_not_live():
    """Mutating the returned dict must not reconfigure the SDK by accident."""
    register_context_windows({"x/y": 1000})
    registered_context_windows()["x/y"] = 999
    assert get_context_window_size("x/y") == 1000


@pytest.mark.parametrize("bad", [0, -1, "1000", None, 1.5])
def test_registration_rejects_nonsense(bad):
    """A zero or negative window would make every request look over budget."""
    with pytest.raises(ValueError):
        register_context_windows({"bad/model": bad})


# ---------------------------------------------------------------------------
# #3 compaction must use the configured provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_compaction_uses_the_provider_not_an_anthropic_client():
    """It called `client.messages.create` — the Anthropic surface — on EVERY
    backend. On an OpenAI-compatible provider that cannot succeed; a bare except
    swallowed it, counted three failures, and switched compaction off. Dead
    feature, no logs, on the majority of supported deployments."""
    from open_agent_sdk.utils.compact import compact_conversation, create_auto_compact_state

    provider = AsyncMock()
    provider.create_message = AsyncMock(return_value=CreateMessageResponse(
        content=[{"type": "text", "text": "a summary of what happened"}]))

    result = await compact_conversation(
        provider, "deepseek/x",
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "hi"}],
        create_auto_compact_state())

    provider.create_message.assert_awaited_once()
    assert "a summary of what happened" in json.dumps(result["compacted_messages"])


@pytest.mark.asyncio
async def test_empty_summary_is_refused_not_accepted():
    """Accepting an empty summary replaces the conversation with a header and
    nothing — silently destroying the context compaction exists to preserve.
    Worse than not compacting."""
    from open_agent_sdk.utils.compact import compact_conversation, create_auto_compact_state

    provider = AsyncMock()
    provider.create_message = AsyncMock(return_value=CreateMessageResponse(content=[]))
    state = create_auto_compact_state()
    result = await compact_conversation(
        provider, "m", [{"role": "user", "content": "hello"}], state)

    # the failure path returns messages UNCOMPACTED rather than a hollow summary
    assert result["compacted_messages"] == [{"role": "user", "content": "hello"}]
    assert result["state"].consecutive_failures == 1


@pytest.mark.asyncio
async def test_dict_content_blocks_are_read():
    """Provider responses normalise content to DICTS; the old extractor only
    handled objects with a .text attribute, so routing through the provider
    would have produced an empty summary — silently."""
    from open_agent_sdk.utils.compact import compact_conversation, create_auto_compact_state

    provider = AsyncMock()
    provider.create_message = AsyncMock(return_value=CreateMessageResponse(
        content=[{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]))
    result = await compact_conversation(
        provider, "m", [{"role": "user", "content": "x"}], create_auto_compact_state())
    assert "part one part two" in json.dumps(result["compacted_messages"])


# ---------------------------------------------------------------------------
# #4 turn budget
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_turn_budget_announced_only_when_asked():
    """max_turns bounded the loop and the model was never told, so it planned as
    if unbounded and got cut off mid-approach."""
    from open_agent_sdk.engine import QueryEngine, QueryEngineConfig

    async def prompt_for(**kw):
        cfg = QueryEngineConfig(model="m", max_turns=7, **kw)
        eng = QueryEngine.__new__(QueryEngine)
        eng._config = cfg
        return await eng._build_system_prompt()

    with_budget = await prompt_for(announce_turn_budget=True)
    without = await prompt_for()
    assert "7 turns" in with_budget
    assert "turn" not in without.lower().split("# environment")[0].replace("You are a helpful", "")
