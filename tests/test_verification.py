"""What Leti is allowed to claim about what it just did.

The one guarantee behind all of these: "the call returned" and "the intended
thing happened" are different facts, and the second is never inferred from the
first. A verifier that cannot confirm something says NOT VERIFIED, and NOT
VERIFIED is never quietly rendered as fine.
"""
from __future__ import annotations

import json

import pytest

from core import coding, verification
from core.verification import FAILED, NOT_APPLICABLE, NOT_VERIFIED, VERIFIED


# --- One vocabulary, not two --------------------------------------------------------

def test_coding_mode_uses_the_shared_vocabulary_rather_than_its_own():
    assert coding.VERIFIED is verification.VERIFIED
    assert coding.NOT_VERIFIED is verification.NOT_VERIFIED
    assert coding.FAILED is verification.FAILED
    assert coding.NOT_APPLICABLE is verification.NOT_APPLICABLE
    assert coding.summarise is verification.summarise


def test_there_is_exactly_one_definition_of_summarise():
    """A second framework for Coding Mode is the thing this must not become."""
    import pathlib

    source = pathlib.Path("core/coding.py").read_text()
    assert "def summarise(" not in source


@pytest.mark.parametrize("checks,expected", [
    ([{"result": VERIFIED}, {"result": VERIFIED}], VERIFIED),
    ([{"result": VERIFIED}, {"result": NOT_VERIFIED}], NOT_VERIFIED),
    ([{"result": NOT_VERIFIED}, {"result": FAILED}], FAILED),
    ([{"result": NOT_APPLICABLE}], NOT_APPLICABLE),
    ([{"result": VERIFIED}, {"result": NOT_APPLICABLE}], VERIFIED),
])
def test_a_summary_is_never_better_than_its_worst_check(checks, expected):
    assert verification.summarise(checks)["overall"] == expected


# --- Files: local, so genuinely checkable --------------------------------------------

def test_a_file_that_exists_with_the_right_contents_is_verified(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello world")
    found = verification.verify("write_file", {"path": str(path), "content": "hello world"},
                                {"success": True})
    assert found["result"] == VERIFIED


def test_a_file_that_does_not_hold_what_was_written_fails(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("something else entirely")
    found = verification.verify("write_file", {"path": str(path), "content": "hello world"},
                                {"success": True})
    assert found["result"] == FAILED


def test_a_write_that_produced_no_file_fails_however_cheerful_the_result(tmp_path):
    found = verification.verify("write_file", {"path": str(tmp_path / "missing.txt")},
                                {"success": True, "message": "Saved!"})
    assert found["result"] == FAILED


def test_a_deleted_file_is_verified_only_when_it_is_gone(tmp_path):
    path = tmp_path / "gone.txt"
    assert verification.verify("delete_file", {"path": str(path)}, {})["result"] == VERIFIED
    path.write_text("still here")
    assert verification.verify("delete_file", {"path": str(path)}, {})["result"] == FAILED


def test_a_move_that_left_the_original_behind_is_not_a_move(tmp_path):
    source, destination = tmp_path / "a.txt", tmp_path / "b.txt"
    source.write_text("x")
    destination.write_text("x")
    found = verification.verify("move_file", {"source_path": str(source),
                                              "destination_path": str(destination)}, {})
    assert found["result"] == NOT_VERIFIED and "copy" in found["detail"]


# --- Things that happen on somebody else's machine ------------------------------------

def test_a_sent_email_is_never_reported_as_delivered():
    found = verification.verify("send_email", {"to": "someone@example.com"},
                                {"success": True, "message": "Email sent"})
    assert found["result"] == NOT_VERIFIED
    assert "delivered" in found["detail"] or "delivery" in found["detail"].lower()


def test_a_calendar_write_without_an_identifier_is_not_confirmed():
    found = verification.verify("schedule_meeting", {"title": "standup"}, {"success": True})
    assert found["result"] == NOT_VERIFIED and found["to_confirm"]


def test_a_calendar_write_the_server_acknowledged_is_confirmed():
    found = verification.verify("schedule_meeting", {"title": "standup"},
                                {"success": True, "uid": "abc-123"})
    assert found["result"] == VERIFIED


def test_github_is_believed_only_when_github_returns_an_identifier():
    assert verification.verify("github", {"action": "create_pull_request"},
                               {"ok": True})["result"] == NOT_VERIFIED
    assert verification.verify("github", {"action": "create_pull_request"},
                               {"number": 7, "html_url": "https://example/pull/7"}
                               )["result"] == VERIFIED


def test_a_zero_exit_code_is_not_a_claim_about_the_intended_effect():
    found = verification.verify("run_shell_command", {"command": "make build"},
                                {"exit_code": 0})
    assert found["result"] == NOT_VERIFIED
    assert "not that" in found["detail"]


def test_a_nonzero_exit_code_is_a_failure():
    found = verification.verify("run_shell_command", {"command": "make build"},
                                {"exit_code": 2})
    assert found["result"] == FAILED


# --- Not applicable is not fine -------------------------------------------------------

def test_a_read_only_tool_is_not_applicable_rather_than_verified():
    found = verification.verify("read_file", {"path": "x"}, {"success": True})
    assert found["result"] == NOT_APPLICABLE


def test_a_tool_with_no_verifier_says_so_rather_than_passing():
    found = verification.verify("some_tool_nobody_wrote_a_verifier_for", {}, {"success": True})
    assert found["result"] == NOT_APPLICABLE
    assert "no way to check" in found["detail"]


def test_a_tool_that_reported_failure_is_failed_whatever_else_is_true():
    found = verification.verify("write_file", {"path": "/x"},
                                {"success": False, "error": "disk full"})
    assert found["result"] == FAILED and "disk full" in found["detail"]


# --- What the model is told -----------------------------------------------------------

def test_a_confirmed_action_adds_no_note_at_all(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("x")
    found = verification.verify("write_file", {"path": str(path), "content": "x"}, {})
    assert verification.note_for(found) == ""


def test_not_applicable_adds_no_note_either():
    assert verification.note_for(verification.verify("read_file", {}, {})) == ""


def test_an_unconfirmed_action_tells_the_model_not_to_claim_it():
    note = verification.note_for(verification.verify("send_email", {"to": "a@b.com"}, {}))
    assert "NOT VERIFIED" in note and "do not report it as confirmed" in note


def test_a_failed_action_tells_the_model_not_to_report_it_as_done(tmp_path):
    note = verification.note_for(
        verification.verify("write_file", {"path": str(tmp_path / "nope")}, {}))
    assert "FAILED" in note and "Do not report this as done" in note


# --- Robustness -----------------------------------------------------------------------

def test_a_verifier_that_raises_reports_not_verified_rather_than_exploding(monkeypatch):
    def explode(arguments, result):
        raise RuntimeError("boom")

    monkeypatch.setitem(verification.VERIFIERS, "write_file", explode)
    found = verification.verify("write_file", {"path": "x"}, {})
    assert found["result"] == NOT_VERIFIED and "boom" in found["detail"]


@pytest.mark.parametrize("result", [None, "", 0, [], "a string result", {"weird": object()}])
def test_any_shape_of_tool_result_is_survivable(result):
    found = verification.verify("write_file", {"path": "/definitely/not/here"}, result)
    assert found["result"] in verification.RESULTS


def test_every_verifier_produces_a_valid_state():
    for name in verification.VERIFIERS:
        found = verification.verify(name, {}, {})
        assert found["result"] in verification.RESULTS, name
        assert found["detail"]


def test_verification_never_contacts_anything(monkeypatch):
    """A check that costs a round trip is a tax on every turn that uses a tool."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("verification made a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in verification.VERIFIERS:
        verification.verify(name, {"to": "a@b.com", "path": "/tmp/x", "command": "ls"},
                            {"success": True})


def test_a_check_is_json_serialisable_because_it_goes_into_a_tool_result():
    for name in list(verification.VERIFIERS) + ["read_file", "unknown_tool"]:
        json.dumps(verification.verify(name, {}, {}))
