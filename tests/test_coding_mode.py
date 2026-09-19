"""Two modes, one Leti - and the coding half that only exists in one of them.

The thing these protect above everything else is the promise in the brief:
Default Leti does not become a coding agent. That is not a matter of taste here,
it is measurable - Default Mode's tool list, its schemas and its prompt have to
be what they were, and the coding tools have to be genuinely absent from it
rather than merely unlikely to be chosen.

After that, the safety properties, which are the reason a coding agent is worth
being careful about at all:

  a rollback never touches work the user was already doing;
  a push never lands on a remote that has moved;
  nothing force-pushes, resets hard or deletes a branch, whoever asks;
  a token is never in a result, a log or an error;
  and nothing is reported as verified that was not actually checked.

GitHub is faked at the HTTP boundary, so what runs is the real request building,
the real error mapping and the real refusals. No test touches a real account.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("chromadb", types.ModuleType("chromadb"))

from core import coding, git_ops, github_client, modes  # noqa: E402


@pytest.fixture(autouse=True)
def clean_modes():
    modes.reset_for_tests()
    coding.release()
    git_ops.clear_checkpoints()
    yield
    modes.reset_for_tests()
    coding.release()
    git_ops.clear_checkpoints()


@pytest.fixture(scope="module")
def registry():
    from unittest.mock import MagicMock

    import main

    return main.build_tool_registry(MagicMock(), MagicMock(), MagicMock())


@pytest.fixture
def repo(tmp_path):
    """A real git repository with a commit in it."""
    def run(*args):
        subprocess.run(args, cwd=tmp_path, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "main")
    run("git", "config", "user.email", "t@example.com")
    run("git", "config", "user.name", "T")
    (tmp_path / "auth.py").write_text(
        "import database\n\n\nclass AuthService:\n"
        "    def login(self, name):\n        return database.get_user(name)\n")
    (tmp_path / "database.py").write_text("def get_user(name):\n    return None\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_auth.py").write_text(
        "from auth import AuthService\n\n\ndef test_login():\n    assert AuthService()\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "first")
    return tmp_path


# --- Default Mode is still Default Mode ---------------------------------------------

def test_leti_starts_as_the_general_assistant():
    assert modes.current() == modes.DEFAULT
    assert modes.is_coding() is False
    assert modes.system_note() == "", "Default Mode's prompt gained a line"


def test_default_mode_cannot_see_the_coding_tools(registry):
    """Not "unlikely to pick them" - absent. This is what stops Default Leti
    turning into a coding agent, and what stops four more schemas landing in a
    context window with a hundred-odd tokens spare."""
    visible = modes.visible_tools(registry, modes.DEFAULT)
    for name in ("code_map", "git_workspace", "github"):
        assert name not in visible, f"{name} is visible to the general assistant"
        assert registry.get(name) is not None, f"{name} should still be registered"


def test_the_model_has_no_way_to_change_mode_from_default(registry):
    """Stronger than "it should not": in Default Mode there is no mode tool at all,
    so the model cannot switch however a request is phrased. Entering a mode is an
    explicit command, matched deterministically before any model call
    (core/intent.py), or the selector in the interface."""
    assert modes.SWITCH_TOOL not in modes.visible_tools(registry, modes.DEFAULT)
    assert registry.get(modes.SWITCH_TOOL) is not None, "the switch should still exist"


def test_a_specialised_mode_can_always_be_left_from_inside_it(registry):
    for name in (modes.CODING, modes.BUSINESS):
        assert modes.SWITCH_TOOL in modes.visible_tools(registry, name)


def test_coding_mode_is_smaller_than_default_not_larger(registry):
    default = modes.visible_tools(registry, modes.DEFAULT)
    coding_tools = modes.visible_tools(registry, modes.CODING)
    default_tokens = len(json.dumps(registry.schemas_for(default)))
    coding_tokens = len(json.dumps(registry.schemas_for(coding_tools)))
    assert coding_tokens < default_tokens, (
        "Coding Mode should send fewer schemas, not more - it is a focused workspace")
    assert {"code_map", "git_workspace", "github", "run_tests", "read_file"} <= coding_tools


def test_no_general_assistant_tools_leak_into_coding_mode(registry):
    visible = modes.visible_tools(registry, modes.CODING)
    for irrelevant in ("get_weather", "place_paper_order", "add_social_watch",
                       "send_email", "create_watch"):
        assert irrelevant not in visible, f"{irrelevant} has no business in a coding turn"


def test_every_registered_tool_is_reachable_in_some_mode(registry):
    """Including the switch, which lives in the specialised modes."""
    covered = set()
    for name in modes.MODES:
        covered |= modes.visible_tools(registry, name)
    assert covered == set(registry.names()), sorted(set(registry.names()) - covered)


# --- Switching ----------------------------------------------------------------------

def test_entering_and_leaving():
    assert modes.enter(modes.CODING)["changed"] is True
    assert modes.is_coding() is True
    assert "CODING MODE" in modes.system_note()

    assert modes.leave()["mode"] == modes.DEFAULT
    assert modes.is_coding() is False
    assert modes.system_note() == ""


def test_entering_the_mode_you_are_in_changes_nothing():
    result = modes.enter(modes.DEFAULT)
    assert result["ok"] is True and result["changed"] is False


def test_an_unknown_mode_is_refused_rather_than_guessed():
    result = modes.enter("wizard")
    assert result["ok"] is False and modes.current() == modes.DEFAULT


def test_switching_touches_nothing_it_should_not():
    """The list from the brief: configuration, permissions, memory, projects,
    tasks, and above all the loaded model."""
    import ast

    source = open("core/modes.py").read()
    tree = ast.parse(source)
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for forbidden in ("reload_settings", "update_section", "set_unattended", "authorize",
                      "clear_buffer", "cancel", "pull", "chat", "generate"):
        assert forbidden not in called, f"switching mode calls {forbidden}()"
    # Names, not words: the module's docstring says it does not restart Ollama,
    # and failing it for saying so would be the wrong kind of strict.
    referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    referenced |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ("llm_client", "num_ctx", "OllamaClient", "session_memory"):
        assert forbidden not in referenced, f"core/modes.py touches {forbidden}"


def test_leaving_releases_the_coding_workspace(repo):
    modes.enter(modes.CODING)
    coding.open_workspace(root=str(repo), repository="owner/name")
    asyncio.run(git_ops.checkpoint(repo, "x"))
    assert coding.workspace() is not None

    modes.leave()
    assert coding.workspace() is None, "the coding workspace outlived the mode"
    assert git_ops.last_checkpoint(repo) is None


# --- Codebase and symbol intelligence -----------------------------------------------

def test_the_files_a_request_is_about_are_found_without_reading_the_repository(repo):
    found = coding.find_files(repo, "fix the authentication bug")
    names = [f["path"] for f in found]
    assert "auth.py" in names
    assert names.index("auth.py") < names.index("database.py") if "database.py" in names else True
    assert all("score" in f for f in found)


def test_python_symbols_are_parsed_rather_than_guessed(repo):
    found = coding.symbols_in(repo / "auth.py")
    assert found["how"] == "parsed"
    classes = [s for s in found["symbols"] if s["kind"] == "class"]
    assert classes[0]["name"] == "AuthService"
    assert "login" in classes[0]["methods"]
    assert "database" in found["imports"]


def test_a_file_that_does_not_parse_says_so_rather_than_inventing_symbols(tmp_path):
    broken = tmp_path / "broken.py"
    broken.write_text("def oops(:\n    pass\n")
    found = coding.symbols_in(broken)
    assert found["symbols"] == []
    assert "syntax error" in found["error"]


def test_other_languages_are_matched_with_patterns_and_admit_it(tmp_path):
    js = tmp_path / "server.js"
    js.write_text("import express from 'express'\n"
                  "class Server {}\n"
                  "export function listen(port) {}\n"
                  "app.get('/health', (req, res) => {})\n")
    found = coding.symbols_in(js)
    assert found["how"] == "pattern"
    assert "read the file before relying on it" in found["note"]
    assert {"Server", "listen"} <= {s["name"] for s in found["symbols"]}
    assert "/health" in found["routes"]


def test_references_are_whole_words(repo):
    (repo / "other.py").write_text("AuthServiceFactory = 1\nx = AuthService()\n")
    found = coding.references_to(repo, "AuthService")
    lines = {(f["path"], f["line"]) for f in found}
    assert ("other.py", 2) in lines
    assert ("other.py", 1) not in lines, "a substring match was counted as a reference"


def test_nothing_is_indexed_scanned_or_watched_in_the_background():
    import ast

    for module in ("core/coding.py", "core/modes.py", "core/git_ops.py",
                   "core/github_client.py", "tools/coding_agent.py"):
        source = open(module).read()
        tree = ast.parse(source)
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.While)], module
        for forbidden in ("Thread(", "watchdog", "inotify", "setInterval", "schedule("):
            assert forbidden not in source, f"{module} uses {forbidden}"


# --- Choosing the tests to run ------------------------------------------------------

def test_the_tests_for_a_change_are_the_ones_that_touch_it(repo):
    chosen = coding.tests_for(repo, ["auth.py"])
    assert chosen["whole_suite"] is False
    assert any("test_auth.py" in t["path"] for t in chosen["tests"])


def test_when_nothing_relates_the_whole_suite_is_the_honest_answer(repo):
    (repo / "unrelated_thing.py").write_text("x = 1\n")
    chosen = coding.tests_for(repo, ["unrelated_thing.py"])
    assert chosen["whole_suite"] is True
    assert "guess dressed up as a decision" in chosen["why"]


def test_naming_nothing_as_changed_does_not_narrow_anything(repo):
    assert coding.tests_for(repo, [])["whole_suite"] is True


# --- Classifying a failure ----------------------------------------------------------

@pytest.mark.parametrize("output,changed,before,kind", [
    ("SyntaxError: invalid syntax", ["auth.py"], [], "syntax"),
    ("ModuleNotFoundError: No module named 'flask'", [], [], "dependency"),
    ("OSError: [Errno 98] Address already in use", [], [], "environment"),
    ("FAILED tests/test_auth.py::test_login - AssertionError in auth.py",
     ["auth.py"], [], "caused_by_change"),
    ("FAILED tests/test_billing.py::test_invoice", ["auth.py"],
     ["tests/test_billing.py::test_invoice"], "pre_existing"),
    ("FAILED tests/test_billing.py::test_invoice", ["auth.py"], [], "unrelated"),
    ("everything exploded", [], [], "ambiguous"),
])
def test_a_failure_is_classified_before_anything_is_edited(output, changed, before, kind):
    assert coding.classify_failure(output, changed, before)["kind"] == kind


def test_a_failure_that_was_already_failing_is_not_adopted():
    verdict = coding.classify_failure(
        "FAILED tests/test_old.py::test_thing", ["auth.py"], ["tests/test_old.py::test_thing"])
    assert verdict["kind"] == "pre_existing"
    assert "do not adopt them" in verdict["what_to_do"]
    assert "Do not edit code you have not read" in verdict["never"]


# --- Git: the user's work is not Leti's to touch ------------------------------------

def test_a_checkpoint_changes_nothing_in_the_repository(repo):
    before = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                            capture_output=True, text=True).stdout
    mark = asyncio.run(git_ops.checkpoint(repo, "before"))
    after = subprocess.run(["git", "status", "--porcelain"], cwd=repo,
                           capture_output=True, text=True).stdout
    assert before == after, "taking a checkpoint moved the user's work"
    assert mark["head"] and mark["already_dirty"] == []


def test_leti_and_the_user_stay_distinguishable(repo):
    (repo / "database.py").write_text("# the user was here\n")      # user's edit
    asyncio.run(git_ops.checkpoint(repo))
    (repo / "auth.py").write_text("# leti was here\n")              # Leti's edit
    (repo / "new.py").write_text("# leti made this\n")

    attribution = asyncio.run(git_ops.changes_since_checkpoint(repo))
    assert attribution["leti_changed"] == ["auth.py", "new.py"]
    assert attribution["user_changed"] == ["database.py"]


def test_a_rollback_undoes_letis_work_and_only_letis_work(repo):
    (repo / "database.py").write_text("# the user was here\n")
    asyncio.run(git_ops.checkpoint(repo))
    (repo / "auth.py").write_text("# leti was here\n")
    (repo / "new.py").write_text("# leti made this\n")

    result = asyncio.run(git_ops.rollback(repo))
    assert result["reverted"] == ["auth.py"]
    assert result["removed"] == ["new.py"]
    assert "class AuthService" in (repo / "auth.py").read_text()
    assert (repo / "database.py").read_text() == "# the user was here\n"
    assert not (repo / "new.py").exists()


def test_a_rollback_refuses_the_users_file_even_when_asked_for_it_by_name(repo):
    (repo / "database.py").write_text("# the user was here\n")
    asyncio.run(git_ops.checkpoint(repo))

    result = asyncio.run(git_ops.rollback(repo, paths=["database.py"]))
    assert result["refused"] == ["database.py"]
    assert result["reverted"] == []
    assert (repo / "database.py").read_text() == "# the user was here\n"


def test_no_checkpoint_means_no_rollback(repo):
    (repo / "auth.py").write_text("changed\n")
    with pytest.raises(git_ops.GitError) as refused:
        asyncio.run(git_ops.rollback(repo))
    assert "will not guess" in str(refused.value)


def test_a_commit_since_the_checkpoint_stops_a_rollback(repo):
    asyncio.run(git_ops.checkpoint(repo))
    (repo / "auth.py").write_text("changed\n")
    subprocess.run(["git", "commit", "-aqm", "second"], cwd=repo, check=True,
                   capture_output=True)
    (repo / "auth.py").write_text("changed again\n")
    with pytest.raises(git_ops.GitError) as refused:
        asyncio.run(git_ops.rollback(repo))
    assert "undo more than Leti's edits" in str(refused.value)


@pytest.mark.parametrize("argument", ["--force", "-f", "--hard", "--delete", "-D",
                                      "--force-with-lease"])
def test_history_rewriting_arguments_are_refused_outright(argument):
    with pytest.raises(git_ops.GitError) as refused:
        git_ops._check_arguments(["push", argument])
    assert "whoever asks" in str(refused.value)


def test_a_commit_names_its_files_rather_than_sweeping_everything_up(repo):
    with pytest.raises(git_ops.GitError) as refused:
        asyncio.run(git_ops.commit(repo, "a message", []))
    assert "may not be its work" in str(refused.value)


def test_a_commit_without_a_message_is_refused(repo):
    with pytest.raises(git_ops.GitError):
        asyncio.run(git_ops.commit(repo, "   ", ["auth.py"]))


def test_a_commit_commits_what_it_named_and_nothing_else(repo):
    (repo / "auth.py").write_text("# leti\n")
    (repo / "database.py").write_text("# the user\n")
    result = asyncio.run(git_ops.commit(repo, "fix(auth): tidy", ["auth.py"]))

    assert result["committed"] == ["auth.py"]
    state = asyncio.run(git_ops.status(repo))
    assert "database.py" in state["dirty_paths"], "the user's file was swept into the commit"


def test_a_directory_that_is_not_a_repository_says_so(tmp_path):
    assert asyncio.run(git_ops.is_repository(tmp_path)) is False


# --- GitHub: faked at the HTTP boundary, so the real code runs -----------------------

class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)
        self.content = self.text.encode()

    def json(self):
        return self._payload


class FakeGitHub:
    """Records what was asked for, including the headers - which is how the token
    tests can prove it went into the header and nowhere else."""

    def __init__(self):
        self.calls = []
        self.responses = {}
        self.default = FakeResponse(200, {})

    def reply(self, path, status=200, payload=None, text=""):
        self.responses[path] = FakeResponse(status, payload, text)

    async def request(self, method, url, headers=None, params=None, json=None):
        path = url.replace("https://api.github.com", "")
        self.calls.append({"method": method, "path": path, "headers": dict(headers or {}),
                           "params": params, "body": json})
        return self.responses.get(path, self.default)


@pytest.fixture
def api(monkeypatch):
    fake = FakeGitHub()

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, **kwargs):
            return await fake.request(method, url, **kwargs)

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(github_client, "_settings",
                        lambda: {"token": "ghp_" + "a" * 36, "repository": "owner/repo"})
    return fake


def test_github_says_it_is_not_connected_rather_than_failing_oddly(monkeypatch):
    monkeypatch.setattr(github_client, "_settings", lambda: {})
    assert github_client.is_connected() is False
    with pytest.raises(github_client.GitHubError) as refused:
        asyncio.run(github_client.account())
    assert refused.value.kind == "unconfigured"
    assert "Connections" in str(refused.value)


def test_the_token_goes_in_the_header_and_nowhere_else(api):
    api.reply("/user", 200, {"login": "per1kleus", "type": "User"})
    result = asyncio.run(github_client.account())

    assert result["login"] == "per1kleus"
    assert "ghp_" not in json.dumps(result), "the token came back in a result"
    call = api.calls[0]
    assert call["headers"]["Authorization"].startswith("Bearer ghp_")
    assert "ghp_" not in json.dumps({k: v for k, v in call.items() if k != "headers"})


@pytest.mark.parametrize("status,kind,fragment", [
    (401, "auth", "rejected the token"),
    (403, "forbidden", "not permitted"),
    (404, "not_found", "not visible to this token"),
    (409, "conflict", "state has moved on"),
    (422, "invalid", "refused that as invalid"),
])
def test_each_github_failure_is_named(api, status, kind, fragment):
    api.reply("/repos/owner/repo", status, {"message": "nope"})
    with pytest.raises(github_client.GitHubError) as raised:
        asyncio.run(github_client.repository("owner/repo"))
    assert raised.value.kind == kind
    assert fragment in str(raised.value).lower()


def test_a_rate_limit_is_not_a_permission_problem(api):
    api.reply("/repos/owner/repo", 403, {"message": "API rate limit exceeded"})
    with pytest.raises(github_client.GitHubError) as raised:
        asyncio.run(github_client.repository("owner/repo"))
    assert raised.value.kind == "rate_limit"


def test_an_error_carrying_a_token_is_scrubbed(api):
    leaked = "failed for token ghp_" + "b" * 36
    api.reply("/repos/owner/repo", 500, {"message": leaked})
    with pytest.raises(github_client.GitHubError) as raised:
        asyncio.run(github_client.repository("owner/repo"))
    assert "ghp_" not in str(raised.value)
    assert "[token removed]" in str(raised.value)


def test_only_the_repositories_the_token_was_granted(api):
    api.reply("/user/repos", 200, [
        {"full_name": "owner/repo", "private": True, "default_branch": "main",
         "permissions": {"push": True, "pull": True, "admin": False}}])
    found = asyncio.run(github_client.repositories())
    assert found[0]["full_name"] == "owner/repo"
    assert found[0]["permissions"] == {"push": True, "pull": True, "admin": False}


def test_a_directory_is_browsed_and_a_file_is_read(api):
    import base64

    api.reply("/repos/owner/repo/contents/core", 200, [
        {"name": "modes.py", "path": "core/modes.py", "type": "file", "size": 120}])
    listing = asyncio.run(github_client.contents("owner/repo", "core"))
    assert listing["kind"] == "directory"
    assert listing["entries"][0]["path"] == "core/modes.py"

    api.reply("/repos/owner/repo/contents/core/modes.py", 200, {
        "path": "core/modes.py", "sha": "abc", "size": 12,
        "content": base64.b64encode(b"MODE = 'coding'\n").decode()})
    read = asyncio.run(github_client.read_file("owner/repo", "core/modes.py"))
    assert read["text"] == "MODE = 'coding'\n"


def test_a_huge_file_is_refused_rather_than_pulled_into_context(api):
    api.reply("/repos/owner/repo/contents/big.bin", 200,
              {"path": "big.bin", "size": github_client.MAX_FILE_BYTES + 1, "content": ""})
    with pytest.raises(github_client.GitHubError) as refused:
        asyncio.run(github_client.read_file("owner/repo", "big.bin"))
    assert refused.value.kind == "too_large"


def test_a_long_file_is_truncated_and_says_so(api):
    import base64

    body = ("x" * 50_000).encode()
    api.reply("/repos/owner/repo/contents/long.py", 200,
              {"path": "long.py", "size": len(body), "sha": "d",
               "content": base64.b64encode(body).decode()})
    read = asyncio.run(github_client.read_file("owner/repo", "long.py", max_chars=1000))
    assert len(read["text"]) == 1000 and read["truncated"] is True


@pytest.mark.parametrize("bad", ["notarepo", "owner/repo/extra", "", "../../etc"])
def test_a_repository_name_that_is_not_one_is_refused(api, bad):
    with pytest.raises(github_client.GitHubError):
        asyncio.run(github_client.repository(bad))


def test_a_protected_branch_is_known_to_be_protected(api):
    api.reply("/repos/owner/repo/branches/main", 200, {"name": "main", "protected": True})
    assert asyncio.run(github_client.is_protected("owner/repo", "main")) is True


def test_a_pull_request_is_opened_never_merged(api):
    api.reply("/repos/owner/repo", 200, {"full_name": "owner/repo", "default_branch": "main"})
    api.reply("/repos/owner/repo/pulls", 201,
              {"number": 7, "html_url": "https://github.com/owner/repo/pull/7", "state": "open"})
    created = asyncio.run(github_client.create_pull_request(
        "owner/repo", "fix: routing", "feature/fix", "main", "body"))

    assert created["number"] == 7
    assert all("merge" not in c["path"].lower() for c in api.calls)
    source = open("core/github_client.py").read()
    assert "/merge" not in source, "there is a merge call in the GitHub client"


def test_a_pull_request_needs_two_different_branches(api):
    api.reply("/repos/owner/repo", 200, {"full_name": "owner/repo", "default_branch": "main"})
    with pytest.raises(github_client.GitHubError):
        asyncio.run(github_client.create_pull_request("owner/repo", "t", "main", "main"))


def test_nothing_in_the_github_client_can_delete_or_force(api):
    source = open("core/github_client.py").read()
    for forbidden in ('"DELETE"', "force", "/merge", "delete_branch"):
        assert forbidden not in source, f"the GitHub client contains {forbidden}"


# --- Verification: never claims what it did not check --------------------------------

def test_syntax_is_verified_only_for_what_can_be_parsed(repo):
    result = coding.check_syntax(repo, ["auth.py", "database.py"])
    assert result["result"] == coding.VERIFIED

    (repo / "bad.py").write_text("def x(:\n")
    failed = coding.check_syntax(repo, ["bad.py"])
    assert failed["result"] == coding.FAILED
    assert failed["problems"][0]["path"] == "bad.py"


def test_a_language_leti_cannot_parse_is_not_applicable_rather_than_fine(repo):
    (repo / "thing.rs").write_text("fn main() { let x = ; }\n")
    result = coding.check_syntax(repo, ["thing.rs"])
    assert result["result"] == coding.NOT_APPLICABLE
    assert "thing.rs" in result["not_checked"]


@pytest.mark.parametrize("content,looks_like", [
    ("TOKEN = 'ghp_" + "c" * 36 + "'", "a GitHub token"),
    ("key = 'sk-" + "d" * 30 + "'", "an API key"),
    ('password = "hunter2hunter2"', "a hard-coded credential"),
    ("-----BEGIN PRIVATE KEY-----", "a private key"),
])
def test_a_credential_in_the_changes_fails_the_check(repo, content, looks_like):
    (repo / "config.py").write_text(content + "\n")
    result = coding.check_secrets(repo, ["config.py"])
    assert result["result"] == coding.FAILED
    assert result["found"][0]["looks_like"] == looks_like


def test_clean_changes_pass_the_secret_check_and_say_what_that_means(repo):
    result = coding.check_secrets(repo, ["auth.py"])
    assert result["result"] == coding.VERIFIED
    assert "not every possible secret" in result["limit"]


def test_debug_leftovers_are_not_verified(repo):
    (repo / "auth.py").write_text("def f():\n    breakpoint()\n")
    assert coding.check_leftovers(repo, ["auth.py"])["result"] == coding.NOT_VERIFIED


def test_scope_notices_a_file_the_task_never_meant_to_touch():
    result = coding.check_scope(intended=["auth.py"],
                                actually_changed=["auth.py", "secrets.env"])
    assert result["result"] == coding.NOT_VERIFIED
    assert result["unexpected"] == ["secrets.env"]


def test_scope_does_not_count_the_users_own_edits_against_the_task():
    result = coding.check_scope(intended=["auth.py"],
                                actually_changed=["auth.py", "notes.md"],
                                user_changed=["notes.md"])
    assert result["result"] == coding.VERIFIED


def test_with_nothing_to_check_against_scope_is_not_applicable():
    assert coding.check_scope([], ["a.py"])["result"] == coding.NOT_APPLICABLE


def test_the_overall_verdict_is_never_better_than_its_worst_check():
    assert coding.summarise([{"result": coding.VERIFIED},
                             {"result": coding.FAILED}])["overall"] == coding.FAILED
    assert coding.summarise([{"result": coding.VERIFIED},
                             {"result": coding.NOT_VERIFIED}])["overall"] == coding.NOT_VERIFIED
    assert coding.summarise([{"result": coding.NOT_APPLICABLE}])["overall"] == coding.NOT_APPLICABLE
    assert coding.summarise([{"result": coding.VERIFIED}])["overall"] == coding.VERIFIED


def test_not_applicable_is_never_reported_as_fine():
    summary = coding.summarise([{"result": coding.NOT_APPLICABLE}])
    assert "not the same as it being fine" in summary["how_to_report"]
    assert "a test that was not run has not passed" in summary["how_to_report"]


# --- Safety: the coding tools are tools, not a side door -----------------------------

def test_the_coding_tools_authorise_nothing_themselves():
    import ast

    for module in ("tools/coding_agent.py", "core/coding.py", "core/git_ops.py",
                   "core/github_client.py", "core/modes.py"):
        tree = ast.parse(open(module).read())
        called = {n.func.attr for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        for forbidden in ("authorize", "set_unattended", "execute_tool", "check_hard_block"):
            assert forbidden not in called, f"{module} calls {forbidden}()"


def test_reading_a_repository_and_writing_to_one_are_classified_differently():
    from core.config_loader import get_permissions
    from tools.coding_agent import GitHubTool, GitWorkspaceTool

    entries = get_permissions()["tools"]
    for name in ("switch_mode", "code_map", "git_workspace", "github"):
        assert name in entries, f"{name} has no action class"

    for tool, read, write in ((GitWorkspaceTool(), "status", "push"),
                              (GitHubTool(), "read", "create_pull_request")):
        assert tool.action_case({"action": read}) == "read"
        assert tool.action_case({"action": write}) == "write"
    assert entries["git_workspace"]["action_by_case"]["write"] == "external"
    assert entries["github"]["action_by_case"]["write"] == "external"


@pytest.mark.asyncio
async def test_pushing_is_an_external_action_and_asks():
    from core.safety_guard import ConfirmationDenied, RiskTier, SafetyGuard

    guard = SafetyGuard()
    assert guard.get_tier("git_workspace", case="write") == RiskTier.EXTERNAL
    assert guard.get_tier("git_workspace", case="read") == RiskTier.READ

    guard.confirmation_callback = lambda prompt: False
    with pytest.raises(ConfirmationDenied):
        await guard.authorize("git_workspace", {"action": "push"}, case="write")


@pytest.mark.asyncio
async def test_an_unattended_run_cannot_push_or_open_a_pull_request():
    from core.safety_guard import ConfirmationDenied, SafetyGuard

    guard = SafetyGuard()
    guard.set_unattended(True)
    for tool, arguments in (("git_workspace", {"action": "push"}),
                            ("github", {"action": "create_pull_request"})):
        with pytest.raises(ConfirmationDenied):
            await guard.authorize(tool, arguments, case="write")


@pytest.mark.asyncio
async def test_coding_mode_does_not_start_itself(registry):
    """The user chooses the mode. A request full of programming words does not."""
    from core.tool_router import select_tools_for

    for request in ("fix the bug in my python script", "why does this function return None",
                    "run the tests"):
        routing = select_tools_for(request, registry,
                                   allowed=modes.visible_tools(registry, modes.DEFAULT))
        assert "code_map" not in routing.tool_names
        assert modes.current() == modes.DEFAULT


@pytest.mark.asyncio
async def test_entering_coding_mode_reports_the_workspace_without_a_token(repo, monkeypatch):
    from tools.control_center import SwitchModeTool

    monkeypatch.setattr(github_client, "_settings", lambda: {})
    result = await SwitchModeTool().run("coding", path=str(repo))

    assert result.success and modes.is_coding()
    assert result.output["git"]["branch"] == "main"
    assert "not connected" in result.output["github"]
    assert "ghp_" not in json.dumps(result.output)


# --- The remote is not where you left it ---------------------------------------------
#
# Found by mutation: removing the divergence check from push() broke nothing, which
# meant the most important remote guarantee in the whole feature was untested.

@pytest.fixture
def cloned(tmp_path):
    """A repository with a real origin, and a second clone to move it from."""
    def run(*args, cwd):
        subprocess.run(args, cwd=cwd, check=True, capture_output=True)

    origin = tmp_path / "origin.git"
    origin.mkdir()
    run("git", "init", "-q", "--bare", "-b", "main", cwd=origin)

    work = tmp_path / "work"
    run("git", "clone", "-q", str(origin), str(work), cwd=tmp_path)
    run("git", "config", "user.email", "t@example.com", cwd=work)
    run("git", "config", "user.name", "T", cwd=work)
    (work / "a.py").write_text("x = 1\n")
    run("git", "add", "-A", cwd=work)
    run("git", "commit", "-qm", "first", cwd=work)
    run("git", "push", "-q", "-u", "origin", "main", cwd=work)

    other = tmp_path / "other"
    run("git", "clone", "-q", str(origin), str(other), cwd=tmp_path)
    run("git", "config", "user.email", "o@example.com", cwd=other)
    run("git", "config", "user.name", "O", cwd=other)
    return {"work": work, "other": other, "run": run}


def test_a_push_lands_when_the_remote_has_not_moved(cloned):
    work, run = cloned["work"], cloned["run"]
    (work / "a.py").write_text("x = 2\n")
    asyncio.run(git_ops.commit(work, "change a", ["a.py"]))

    result = asyncio.run(git_ops.push(work))
    assert result["pushed"] == "main"


def test_a_push_refuses_when_somebody_else_pushed_first(cloned):
    """Somebody else's commit is on the remote. Pushing over it is how their work
    disappears, so this stops and says so - it does not force, and it does not
    quietly merge on their behalf."""
    work, other, run = cloned["work"], cloned["other"], cloned["run"]

    (other / "theirs.py").write_text("theirs = True\n")
    run("git", "add", "-A", cwd=other)
    run("git", "commit", "-qm", "their work", cwd=other)
    run("git", "push", "-q", "origin", "main", cwd=other)

    (work / "a.py").write_text("x = 3\n")
    asyncio.run(git_ops.commit(work, "mine", ["a.py"]))

    with pytest.raises(git_ops.GitError) as refused:
        asyncio.run(git_ops.push(work))
    assert "commit(s) this checkout does not" in str(refused.value)
    assert "forcing" in str(refused.value)

    # And their commit is still on the remote, untouched.
    log = subprocess.run(["git", "log", "--oneline", "main"], cwd=cloned["work"].parent / "origin.git",
                         capture_output=True, text=True).stdout
    assert "their work" in log


def test_the_remote_is_fetched_rather_than_remembered(cloned):
    """A local view of a remote branch can be hours old. remote_state fetches, so
    the decision is made against what is there now."""
    work, other, run = cloned["work"], cloned["other"], cloned["run"]
    before = asyncio.run(git_ops.remote_state(work))
    assert before["fetched"] is True and not before.get("diverged")

    (other / "theirs.py").write_text("theirs = True\n")
    run("git", "add", "-A", cwd=other)
    run("git", "commit", "-qm", "their work", cwd=other)
    run("git", "push", "-q", "origin", "main", cwd=other)

    after = asyncio.run(git_ops.remote_state(work))
    assert after["diverged"] is True and after["behind"] >= 1


def test_an_unreachable_remote_stops_a_push_rather_than_guessing(repo):
    """No origin at all: the answer is "could not check", never "probably fine"."""
    with pytest.raises(git_ops.GitError) as refused:
        asyncio.run(git_ops.push(repo))
    # git's own words - "'origin' does not appear to be a git repository" - which
    # is more useful than anything this could paraphrase it into.
    assert "origin" in str(refused.value)
