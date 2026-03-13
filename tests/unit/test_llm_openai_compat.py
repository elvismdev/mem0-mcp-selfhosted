"""Tests for llm_openai_compat.py — OpenAICompatLLM response_format stripping."""

from __future__ import annotations

from unittest.mock import patch


class TestOpenAICompatLLM:
    """Test that OpenAICompatLLM strips json_object response_format."""

    def _make_llm(self):
        """Instantiate OpenAICompatLLM with a minimal mocked config."""
        with patch("mem0.llms.openai.OpenAI"):
            from mem0_mcp_selfhosted.llm_openai_compat import OpenAICompatLLM
            from mem0.configs.llms.openai import OpenAIConfig

            config = OpenAIConfig(model="test-model", api_key="not-needed")
            return OpenAICompatLLM(config)

    def test_json_object_is_replaced_with_text(self):
        """response_format={'type': 'json_object'} is replaced with {'type': 'text'} before super()."""
        llm = self._make_llm()

        with patch("mem0.llms.openai.OpenAILLM.generate_response") as mock_super:
            mock_super.return_value = {"content": "ok", "tool_calls": None}
            llm.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

        _, kwargs = mock_super.call_args
        assert kwargs["response_format"] == {"type": "text"}

    def test_json_schema_is_passed_through_unchanged(self):
        """response_format={'type': 'json_schema', ...} is forwarded as-is."""
        llm = self._make_llm()
        schema_format = {"type": "json_schema", "json_schema": {"name": "Foo", "schema": {}}}

        with patch("mem0.llms.openai.OpenAILLM.generate_response") as mock_super:
            mock_super.return_value = {"content": "ok", "tool_calls": None}
            llm.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                response_format=schema_format,
            )

        _, kwargs = mock_super.call_args
        assert kwargs["response_format"] is schema_format

    def test_none_response_format_is_passed_through(self):
        """response_format=None is forwarded as-is."""
        llm = self._make_llm()

        with patch("mem0.llms.openai.OpenAILLM.generate_response") as mock_super:
            mock_super.return_value = {"content": "ok", "tool_calls": None}
            llm.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                response_format=None,
            )

        _, kwargs = mock_super.call_args
        assert kwargs["response_format"] is None
