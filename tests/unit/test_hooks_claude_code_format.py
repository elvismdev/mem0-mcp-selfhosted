"""Regression tests for the Claude Code transcript schema in hooks.py.

Claude Code writes JSONL transcript entries as
``{"type": "user"|"assistant", "message": {"role": ..., "content": ...}}`` —
role/content are nested under "message". The rest of the suite uses a flat
``{"role", "content"}`` shape, which is how the bug in #11 slipped through:
the Stop hook silently saved nothing because ``_read_recent_messages`` only
looked at the top level. These tests exercise the real nested format.
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import MagicMock, patch

from mem0_mcp_selfhosted import hooks


def _capture_output(func, stdin_data: str) -> dict:
    """Run a hook entry point with mocked stdin and capture its stdout JSON."""
    captured = StringIO()
    with patch("sys.stdin", StringIO(stdin_data)), patch("sys.stdout", captured):
        func()
    return json.loads(captured.getvalue())


def _line(role: str, content) -> str:
    """One Claude Code JSONL line (role/content nested under 'message')."""
    return json.dumps({
        "type": role,
        "message": {"role": role, "content": content},
        "uuid": f"{role}-1",
    })


def test_read_recent_messages_parses_nested_message(tmp_path):
    """role/content nested under 'message' are extracted (string + content blocks)."""
    p = tmp_path / "transcript.jsonl"
    p.write_text(
        _line("user", "first user message that is long enough") + "\n"
        + _line("assistant", [{"type": "text", "text": "assistant response text"}]) + "\n"
    )

    result = hooks._read_recent_messages(str(p))

    assert result == [
        ("user", "first user message that is long enough"),
        ("assistant", "assistant response text"),
    ]


def test_stop_hook_saves_with_nested_message_format(tmp_path):
    """Stop hook saves to mem0 for a real Claude Code (nested) transcript."""
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        _line("user", "Please refactor the auth module to use JWT tokens instead of sessions") + "\n"
        + _line(
            "assistant",
            [{"type": "text", "text": "Refactored auth to JWT: replaced sessions with jsonwebtoken and added a refresh endpoint."}],
        )
        + "\n"
    )
    stdin_data = json.dumps({
        "session_id": "sess-1",
        "cwd": "/home/user/myproject",
        "transcript_path": str(transcript),
    })

    mock_mem = MagicMock()
    with patch.object(hooks, "_get_memory", return_value=mock_mem):
        result = _capture_output(hooks.stop_main, stdin_data)

    assert result["continue"] is True
    mock_mem.add.assert_called_once()
    summary = mock_mem.add.call_args.kwargs["messages"][0]["content"]
    assert "[User]:" in summary
    assert "[Assistant]:" in summary
    assert "JWT" in summary
