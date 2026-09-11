# Vendored from `codeany-ai/open-agent-sdk-python`

This repository is a vendored copy of an **abandoned upstream**, taken at the
commit Velostream already runs in production.

| | |
|---|---|
| upstream | https://github.com/codeany-ai/open-agent-sdk-python |
| commit | `ab068156bf7db79cd23a883e8264a12803e23966` |
| upstream created | 2026-04-01 |
| upstream last push | 2026-04-03 — **two days later** |
| licence | MIT, © 2025 CodeAny — unchanged, see `LICENSE` |

Full upstream history is preserved in this repository. Nothing has been rewritten
or squashed; `git log` shows CodeAny's commits as they were.

## Why vendor rather than depend

**The upstream is abandoned.** Eight commits over two days, then nothing for five
months. There is no issue tracker activity, no releases, and no one to accept a
patch.

**There is no upgrade path, and the obvious one is a trap.** The PyPI name
`open-agent-sdk` belongs to a *different, unrelated project*
(`slb350/open-agent-sdk`). Same name, same `0.1.0` version string, entirely
different code — ours ships `open_agent_sdk/` with `engine.py`, `providers/`,
`skills/`, `mcp/`; theirs ships `any_agent/` with five files. A routine
`pip install -U open-agent-sdk` would silently replace a full agent runtime with
an unrelated thin client.

Velostream installs from git (`direct_url.json` pins the commit), so it was never
exposed to that swap — but nothing recorded *why*, and the next person to
"fix the dependency" would have found out the hard way.

**The code is worth keeping.** Measured A/B against `openai-agents 0.22.2` on two
real Velostream code-discovery tasks, same model
(`deepseek/deepseek-v4-flash-0731`), same working tree, matched tools:

| | this SDK | openai-agents |
|---|---|---|
| task 1 | correct, **3 turns**, 8.7 s | correct, 12 turns, 17.9 s |
| task 2 | correct, **4 turns** | correct, 9 turns |

The gap is parallel tool dispatch — this SDK issues several tool calls in one
turn (`tool start names=Grep,Grep`) where the alternative went serial. On an
agent loop that re-sends its transcript every turn, turns are the cost driver.

## What Velostream patches today, and why it belongs here instead

`.agents/bin/velo_sdk_patches.py` monkey-patches this SDK at import for five
defects. Monkey-patching an abandoned dependency is strictly worse than fixing
the source now that we own it — one patch (`install_tool_input_normalisation`)
*returns False and continues* if it cannot apply, so a change underneath it would
silently restore the crash it prevents.

The defects, all reproduced against this commit:

1. **Crash on double-encoded tool arguments.** `json.loads` succeeds and returns
   a `str`; the SDK assumes `dict` and dies with
   `AttributeError: 'str' object has no attribute 'get'`. Killed a run after
   9.9M input tokens (velostream#2010).
2. **Cache fields hardcoded to zero** in `providers/openai_provider.py`, so
   cache hits were invisible and every cost record understated the cache.
3. **`DEFAULT_CONTEXT_WINDOW = 200_000`** regardless of model — roughly 5×
   too small for the models in use, tripping auto-compaction early.
4. **No OpenRouter provider pinning.** OpenRouter load-balances per request and
   its prefix cache is per-endpoint, so an agent loop missed the cache on every
   turn. Measured: unpinned 0 cached tokens across 3 calls on 3 endpoints;
   pinned, 1,280 cached on all 3 and a 3.7× lower cost.
5. **Turn budget never communicated to the model.**

## Ground rules for this repository

- **Upstream attribution stays.** `LICENSE` and the original history are not to
  be rewritten.
- **Every local change gets a commit message saying what it fixes and how it was
  reproduced** — the same bar as the rest of Velostream's tooling.
- **Patches move here, they do not accumulate in both places.** A defect fixed in
  this repository must be deleted from `velo_sdk_patches.py` in the same change,
  or the two will disagree and the monkey-patch will win silently.
