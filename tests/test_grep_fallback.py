"""The grep fallback must speak the regex dialect the tool advertises.

GrepTool prefers ripgrep and falls back to POSIX grep. The fallback is not a rare
path: `shutil.which("rg")` finds only a real BINARY, so any machine where
ripgrep is absent — or installed as a shell function or alias, which is how it
was configured on the machine that surfaced this — takes it on every call.

Two defects met there, and both were silent:

  1. grep without -E is BASIC regex, where `|` `?` `+` `(` `)` are literal
     characters. `a|b` searched for the string "a|b". Exit 0, no output.
  2. stderr and the exit code were discarded, so a search that FAILED and a
     search that found nothing returned the identical "No matches found."

Measured over a 5-round agent benchmark with ripgrep absent: 63% of the model's
patterns used those metacharacters, and 68% of Grep calls returned no matches.
The agent concluded the code was not there and spent the run rewording queries.
"""
import asyncio
import shutil

import pytest

from open_agent_sdk.tools.grep import GrepTool
from open_agent_sdk.types import ToolContext


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "hit.rs").write_text(
        "fn merge_left_with_nulls_batch() {}\nfn merge_right_with_nulls_batch() {}\n"
    )
    (tmp_path / "miss.rs").write_text("fn unrelated() {}\n")
    return tmp_path


def run(tool, inp, cwd):
    return asyncio.run(tool.call(inp, ToolContext(cwd=str(cwd))))


@pytest.fixture
def no_rg(monkeypatch):
    """Force the fallback — the whole point is the path taken WITHOUT ripgrep."""
    monkeypatch.setattr(shutil, "which", lambda name: None)


@pytest.mark.parametrize("pattern", [
    "merge_left_with_nulls|merge_right_with_nulls",   # alternation
    "merge_left_with_nulls_batch|nothing_here",       # one side matches
    "merge_(left|right)_with_nulls",                  # group + alternation
    "merge_left.?with_nulls",                         # optional
    "merge_left_+with_nulls",                         # repetition
])
def test_extended_regex_works_in_the_fallback(tree, no_rg, pattern):
    """Each of these silently matched NOTHING before -E was added."""
    r = run(GrepTool(), {"pattern": pattern, "path": ".",
                         "output_mode": "files_with_matches"}, tree)
    assert not r.is_error, r.content
    assert "hit.rs" in r.content, f"{pattern!r} found nothing: {r.content!r}"


def test_a_genuine_miss_still_reports_no_matches(tree, no_rg):
    """The fix must not turn every empty result into an error."""
    r = run(GrepTool(), {"pattern": "absolutely_not_present|also_not",
                         "path": ".", "output_mode": "files_with_matches"}, tree)
    assert not r.is_error
    assert "No matches found" in r.content


def test_non_matching_file_is_not_returned(tree, no_rg):
    r = run(GrepTool(), {"pattern": "merge_left|merge_right", "path": ".",
                         "output_mode": "files_with_matches"}, tree)
    assert "miss.rs" not in r.content


def test_content_mode_returns_the_line(tree, no_rg):
    r = run(GrepTool(), {"pattern": "merge_(left|right)_with_nulls_batch",
                         "path": ".", "output_mode": "content"}, tree)
    assert "merge_left_with_nulls_batch" in r.content


def test_a_pattern_starting_with_a_dash_is_not_read_as_a_flag(tree, no_rg):
    """`--` before the pattern: without it, a model searching for "-i" or
    "--force" hands grep a flag instead of a pattern."""
    (tree / "dash.rs").write_text("let x = --force;\n")
    r = run(GrepTool(), {"pattern": "--force", "path": ".",
                         "output_mode": "files_with_matches"}, tree)
    assert not r.is_error, r.content
    assert "dash.rs" in r.content


def test_a_failed_search_is_an_error_not_an_empty_result(tree, no_rg):
    """Exit >=2 means the search never ran. Reporting that as 'No matches found'
    lets an agent conclude the code does not exist because the tool broke."""
    r = run(GrepTool(), {"pattern": "x", "path": "no/such/directory",
                         "output_mode": "files_with_matches"}, tree)
    assert r.is_error, f"expected an error, got: {r.content!r}"
    assert "No matches found" not in r.content
    assert "Search failed" in r.content


def test_unparseable_regex_surfaces_as_an_error(tree, no_rg):
    """An unbalanced group is a bug in the QUERY; the model can only fix it if
    it is told, rather than being shown an empty result."""
    r = run(GrepTool(), {"pattern": "merge_(left", "path": ".",
                         "output_mode": "content"}, tree)
    assert r.is_error, f"expected an error, got: {r.content!r}"
    assert "Search failed" in r.content


@pytest.mark.skipif(not shutil.which("rg"), reason="ripgrep binary not installed")
def test_ripgrep_and_the_fallback_agree(tree, monkeypatch):
    """The fallback exists to be equivalent, so pin that it is."""
    q = {"pattern": "merge_(left|right)_with_nulls_batch", "path": ".",
         "output_mode": "files_with_matches"}
    with_rg = run(GrepTool(), q, tree)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    without = run(GrepTool(), q, tree)
    assert ("hit.rs" in with_rg.content) == ("hit.rs" in without.content) is True
