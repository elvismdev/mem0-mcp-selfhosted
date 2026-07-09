"""Tests for env.py — centralized env var helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

from mem0_mcp_selfhosted.env import bool_env, env, opt_env


class TestEnv:
    """Tests for env() — required env vars with defaults."""

    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"TEST_KEY": "hello"}):
            assert env("TEST_KEY") == "hello"

    def test_returns_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert env("NONEXISTENT", "fallback") == "fallback"

    def test_empty_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert env("NONEXISTENT") == ""

    def test_strips_whitespace_from_value(self):
        with patch.dict(os.environ, {"TEST_KEY": "  val \n"}):
            assert env("TEST_KEY") == "val"

    def test_strips_whitespace_from_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert env("NONEXISTENT", " padded ") == "padded"

    def test_whitespace_only_value_becomes_empty(self):
        with patch.dict(os.environ, {"TEST_KEY": "  \n"}):
            assert env("TEST_KEY") == ""

    def test_empty_value_ignores_default(self):
        """Explicitly set empty value returns empty, not the default."""
        with patch.dict(os.environ, {"TEST_KEY": ""}):
            assert env("TEST_KEY", "fallback") == ""


class TestOptEnv:
    """Tests for opt_env() — optional env vars (None when absent)."""

    def test_returns_none_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            assert opt_env("NONEXISTENT") is None

    def test_returns_value_when_set(self):
        with patch.dict(os.environ, {"TEST_KEY": "secret"}):
            assert opt_env("TEST_KEY") == "secret"

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"TEST_KEY": "  secret \n"}):
            assert opt_env("TEST_KEY") == "secret"

    def test_whitespace_only_returns_empty_string(self):
        """Whitespace-only value becomes empty string (falsy), not None."""
        with patch.dict(os.environ, {"TEST_KEY": "  \n"}):
            result = opt_env("TEST_KEY")
            assert result == ""
            assert result is not None

    def test_empty_string_returns_empty_string(self):
        """Explicitly empty value returns empty string, not None."""
        with patch.dict(os.environ, {"TEST_KEY": ""}):
            result = opt_env("TEST_KEY")
            assert result == ""
            assert result is not None


class TestBoolEnv:
    """Tests for bool_env() — boolean env vars."""

    def test_true_values(self):
        for val in ("true", "True", "TRUE", "1", "yes", "YES"):
            with patch.dict(os.environ, {"TEST_KEY": val}):
                assert bool_env("TEST_KEY") is True, f"Expected True for {val!r}"

    def test_false_values(self):
        for val in ("false", "0", "no", "", "anything_else"):
            with patch.dict(os.environ, {"TEST_KEY": val}):
                assert bool_env("TEST_KEY") is False, f"Expected False for {val!r}"

    def test_default_is_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert bool_env("NONEXISTENT") is False

    def test_custom_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            assert bool_env("NONEXISTENT", "true") is True

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"TEST_KEY": " true \n"}):
            assert bool_env("TEST_KEY") is True

    def test_whitespace_only_is_false(self):
        with patch.dict(os.environ, {"TEST_KEY": "  \n"}):
            assert bool_env("TEST_KEY") is False


class TestPromptEnv:
    """Tests for prompt_env() — inline prompt vars with _FILE companions."""

    def test_inline_value(self):
        with patch.dict(os.environ, {"TEST_PROMPT": "Extract technical facts."}):
            from mem0_mcp_selfhosted.env import prompt_env

            assert prompt_env("TEST_PROMPT") == "Extract technical facts."

    def test_absent_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            from mem0_mcp_selfhosted.env import prompt_env

            assert prompt_env("TEST_PROMPT") is None

    def test_file_variant(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Line one.\nLine two.\n", encoding="utf-8")
        with patch.dict(os.environ, {"TEST_PROMPT_FILE": str(prompt_file)}, clear=True):
            from mem0_mcp_selfhosted.env import prompt_env

            assert prompt_env("TEST_PROMPT") == "Line one.\nLine two."

    def test_inline_wins_over_file(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("from file", encoding="utf-8")
        env_vars = {"TEST_PROMPT": "inline", "TEST_PROMPT_FILE": str(prompt_file)}
        with patch.dict(os.environ, env_vars, clear=True):
            from mem0_mcp_selfhosted.env import prompt_env

            assert prompt_env("TEST_PROMPT") == "inline"

    def test_missing_file_raises(self):
        with patch.dict(os.environ, {"TEST_PROMPT_FILE": "/nonexistent/prompt.txt"}, clear=True):
            from mem0_mcp_selfhosted.env import prompt_env

            try:
                prompt_env("TEST_PROMPT")
                raise AssertionError("expected OSError")
            except OSError:
                pass
