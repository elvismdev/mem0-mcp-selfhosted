"""Tests for llm_router.py — SplitModelGraphLLM tool-name-based routing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_tool(name: str) -> dict:
    """Create a minimal OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Test tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


EXTRACTION_RESPONSE = {
    "content": None,
    "tool_calls": [{"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}]}}],
}

CONTRADICTION_RESPONSE = {
    "content": None,
    "tool_calls": [{"name": "delete_graph_memory", "arguments": {"source": "Alice"}}],
}


class TestSplitModelGraphLLM:
    """Test routing logic with mocked backing LLMs."""

    @pytest.fixture
    def router(self):
        """Create a SplitModelGraphLLM with mocked backing LLMs."""
        with patch("mem0_mcp_selfhosted.llm_router.LlmFactory") as mock_factory:
            # Make LlmFactory.create() return distinct mocks for each provider
            extraction_llm = MagicMock(name="extraction_llm")
            extraction_llm.generate_response.return_value = EXTRACTION_RESPONSE

            contradiction_llm = MagicMock(name="contradiction_llm")
            contradiction_llm.generate_response.return_value = CONTRADICTION_RESPONSE

            def create_side_effect(provider, config):
                if provider == "gemini":
                    return extraction_llm
                return contradiction_llm

            mock_factory.create.side_effect = create_side_effect

            from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig

            config = SplitModelGraphLLMConfig(
                extraction_provider="gemini",
                extraction_model="gemini-2.5-flash-lite",
                contradiction_provider="anthropic",
                contradiction_model="claude-opus-4-6",
            )
            router = SplitModelGraphLLM(config)
            router._extraction_mock = extraction_llm
            router._contradiction_mock = contradiction_llm
            return router

    def test_extract_entities_routes_to_extraction(self, router):
        """extract_entities tool routes to extraction LLM."""
        tools = [_make_tool("extract_entities")]
        result = router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._extraction_mock.generate_response.assert_called_once()
        router._contradiction_mock.generate_response.assert_not_called()

    def test_establish_relationships_routes_to_extraction(self, router):
        """establish_relationships tool routes to extraction LLM."""
        tools = [_make_tool("establish_relationships")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._extraction_mock.generate_response.assert_called_once()
        router._contradiction_mock.generate_response.assert_not_called()

    def test_establish_relations_routes_to_extraction(self, router):
        """establish_relations (alternate name) routes to extraction LLM."""
        tools = [_make_tool("establish_relations")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._extraction_mock.generate_response.assert_called_once()
        router._contradiction_mock.generate_response.assert_not_called()

    def test_delete_graph_memory_routes_to_contradiction(self, router):
        """delete_graph_memory tool routes to contradiction LLM."""
        tools = [_make_tool("delete_graph_memory")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._contradiction_mock.generate_response.assert_called_once()
        router._extraction_mock.generate_response.assert_not_called()

    def test_update_graph_memory_routes_to_contradiction(self, router):
        """update_graph_memory tool routes to contradiction LLM."""
        tools = [_make_tool("update_graph_memory")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._contradiction_mock.generate_response.assert_called_once()
        router._extraction_mock.generate_response.assert_not_called()

    def test_add_graph_memory_routes_to_contradiction(self, router):
        """add_graph_memory tool routes to contradiction LLM."""
        tools = [_make_tool("add_graph_memory")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._contradiction_mock.generate_response.assert_called_once()
        router._extraction_mock.generate_response.assert_not_called()

    def test_noop_routes_to_contradiction(self, router):
        """noop tool routes to contradiction LLM."""
        tools = [_make_tool("noop")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._contradiction_mock.generate_response.assert_called_once()
        router._extraction_mock.generate_response.assert_not_called()

    def test_no_tools_routes_to_extraction(self, router):
        """No tools defaults to extraction LLM."""
        router.generate_response(messages=[{"role": "user", "content": "test"}])

        router._extraction_mock.generate_response.assert_called_once()
        router._contradiction_mock.generate_response.assert_not_called()

    def test_unknown_tool_routes_to_extraction(self, router):
        """Unknown tool name defaults to extraction LLM."""
        tools = [_make_tool("some_unknown_tool")]
        router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        router._extraction_mock.generate_response.assert_called_once()
        router._contradiction_mock.generate_response.assert_not_called()

    def test_response_passthrough_extraction(self, router):
        """Router returns exact response from extraction LLM."""
        tools = [_make_tool("extract_entities")]
        result = router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        assert result is EXTRACTION_RESPONSE

    def test_response_passthrough_contradiction(self, router):
        """Router returns exact response from contradiction LLM."""
        tools = [_make_tool("delete_graph_memory")]
        result = router.generate_response(messages=[{"role": "user", "content": "test"}], tools=tools)

        assert result is CONTRADICTION_RESPONSE

    def test_contradiction_openai_base_url_passed_to_factory(self):
        """contradiction_openai_base_url is forwarded to LlmFactory.create() for the contradiction LLM."""
        with patch("mem0_mcp_selfhosted.llm_router.LlmFactory") as mock_factory:
            mock_factory.create.return_value = MagicMock()

            from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig

            config = SplitModelGraphLLMConfig(
                extraction_provider="gemini",
                extraction_model="gemini-2.5-flash-lite",
                contradiction_provider="openai",
                contradiction_model="qwen3-14b",
                contradiction_openai_base_url="http://192.168.200.83:1234/v1",
            )
            SplitModelGraphLLM(config)

            # Find the call for the contradiction LLM (provider="openai")
            openai_calls = [
                call for call in mock_factory.create.call_args_list
                if call.args[0] == "openai"
            ]
            assert len(openai_calls) == 1
            _, contradiction_config = openai_calls[0].args
            assert contradiction_config["openai_base_url"] == "http://192.168.200.83:1234/v1"
