"""Unit tests for server.py — MCP tool functions, _init_memory(), _create_server().

Tests tool orchestration logic with mocked Memory: kwargs assembly, user_id
defaults, scope validation, error handling, and delegation to helpers.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import mem0_mcp_selfhosted.server as server_mod


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    """Set default env vars for all tests."""
    monkeypatch.setenv("MEM0_USER_ID", "test-user")


@pytest.fixture
def mock_memory():
    """Create a mock Memory object and patch server globals."""
    mem = MagicMock()
    mem.graph = None
    mem.enable_graph = False
    mem.add.return_value = {"results": [{"id": "mem-1", "memory": "test fact"}]}
    mem.search.return_value = {"results": [{"id": "mem-1", "score": 0.95}]}
    mem.get_all.return_value = {"results": [{"id": "mem-1"}]}
    mem.get.return_value = {"id": "mem-1", "memory": "test fact"}
    mem.update.return_value = None
    mem.delete.return_value = None
    return mem


@pytest.fixture
def server_with_mock(mock_memory):
    """Create a FastMCP server with mocked Memory and helpers."""
    original_memory = server_mod.memory
    original_graph_default = server_mod._enable_graph_default
    server_mod.memory = mock_memory
    server_mod._enable_graph_default = False

    srv = server_mod._create_server()

    yield srv, mock_memory

    server_mod.memory = original_memory
    server_mod._enable_graph_default = original_graph_default


def _get_tool_fn(srv, name: str):
    """Extract a tool function from the FastMCP server by name."""
    tool = srv._tool_manager._tools.get(name)
    assert tool is not None, f"Tool {name!r} not found in server"
    return tool.fn


# ============================================================
# 2. Tool Function Tests
# ============================================================


class TestAddMemory:
    def test_kwargs_assembly(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        result = fn(
            text="I prefer Python",
            user_id="alice",
            metadata={"source": "chat"},
            infer=False,
        )
        mem.add.assert_called_once()
        args, kwargs = mem.add.call_args
        assert args[0] == [{"role": "user", "content": "I prefer Python"}]
        assert kwargs["user_id"] == "alice"
        assert kwargs["metadata"] == {"source": "chat"}
        assert kwargs["infer"] is False
        parsed = json.loads(result)
        assert "results" in parsed

    def test_messages_precedence(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        custom_msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        fn(text="ignored text", messages=custom_msgs)
        args, _ = mem.add.call_args
        assert args[0] == custom_msgs

    def test_default_user_id(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "add_memory")
        fn(text="some fact")
        _, kwargs = mem.add.call_args
        assert kwargs["user_id"] == "test-user"


class TestSearchMemories:
    def test_all_kwargs(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "search_memories")
        fn(
            query="python preferences",
            user_id="bob",
            agent_id="agent-1",
            run_id="run-1",
            filters={"key": {"eq": "val"}},
            limit=5,
            threshold=0.8,
            rerank=True,
        )
        _, kwargs = mem.search.call_args
        assert kwargs["query"] == "python preferences"
        assert kwargs["user_id"] == "bob"
        assert kwargs["agent_id"] == "agent-1"
        assert kwargs["run_id"] == "run-1"
        assert kwargs["filters"] == {"key": {"eq": "val"}}
        assert kwargs["limit"] == 5
        assert kwargs["threshold"] == 0.8
        assert kwargs["rerank"] is True


class TestGetMemories:
    def test_scope_filters(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memories")
        fn(user_id="alice", agent_id="agent-1", run_id="run-1", limit=10)
        mem.get_all.assert_called_once_with(
            user_id="alice", agent_id="agent-1", run_id="run-1", limit=10
        )


class TestGetMemory:
    def test_by_id(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "get_memory")
        fn(memory_id="uuid-123")
        mem.get.assert_called_once_with("uuid-123")


class TestUpdateMemory:
    def test_uses_data_param(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "update_memory")
        result = fn(memory_id="uuid-123", text="updated fact")
        mem.update.assert_called_once_with("uuid-123", data="updated fact")
        parsed = json.loads(result)
        assert parsed["message"] == "Memory updated successfully!"


class TestDeleteMemory:
    def test_delegation(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_memory")
        result = fn(memory_id="uuid-123")
        mem.delete.assert_called_once_with("uuid-123")
        parsed = json.loads(result)
        assert parsed["message"] == "Memory deleted successfully!"


class TestDeleteAllMemories:
    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=0)
    def test_scope_defaults_to_user_id(self, mock_sbd, server_with_mock):
        """delete_all_memories always falls back to get_default_user_id(),
        so uid is always truthy.  Verify the default user scope is used."""
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_all_memories")
        result = fn()
        mock_sbd.assert_called_once_with(mem, {"user_id": "test-user"}, graph_enabled=False)
        parsed = json.loads(result)
        assert parsed["count"] == 0

    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=3)
    def test_delegates_safe_bulk_delete(self, mock_sbd, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_all_memories")
        result = fn(user_id="alice")
        mock_sbd.assert_called_once_with(mem, {"user_id": "alice"}, graph_enabled=False)
        parsed = json.loads(result)
        assert parsed["count"] == 3


class TestListEntities:
    @patch("mem0_mcp_selfhosted.server.list_entities_facet")
    def test_delegation(self, mock_facet, server_with_mock):
        mock_facet.return_value = {"users": [], "agents": [], "runs": []}
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "list_entities")
        result = fn()
        mock_facet.assert_called_once_with(mem)
        parsed = json.loads(result)
        assert "users" in parsed


class TestDeleteEntities:
    def test_scope_validation(self, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_entities")
        result = fn()
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("mem0_mcp_selfhosted.server.safe_bulk_delete", return_value=5)
    def test_delegates_safe_bulk_delete(self, mock_sbd, server_with_mock):
        srv, mem = server_with_mock
        fn = _get_tool_fn(srv, "delete_entities")
        result = fn(user_id="alice")
        mock_sbd.assert_called_once_with(mem, {"user_id": "alice"}, graph_enabled=False)
        parsed = json.loads(result)
        assert parsed["count"] == 5


class TestGraphTools:
    @patch("mem0_mcp_selfhosted.server.search_graph")
    def test_search_graph_delegation(self, mock_sg, server_with_mock):
        mock_sg.return_value = '{"entities": []}'
        srv, _ = server_with_mock
        fn = _get_tool_fn(srv, "mcp_search_graph")
        fn(query="Python")
        mock_sg.assert_called_once_with("Python")

    @patch("mem0_mcp_selfhosted.server.get_entity")
    def test_get_entity_delegation(self, mock_ge, server_with_mock):
        mock_ge.return_value = '{"relationships": []}'
        srv, _ = server_with_mock
        fn = _get_tool_fn(srv, "mcp_get_entity")
        fn(name="TypeScript")
        mock_ge.assert_called_once_with("TypeScript")


# ============================================================
# 3. Error Handling and Initialization Tests
# ============================================================


class TestToolErrorHandling:
    def test_exception_returns_json_error(self, server_with_mock):
        srv, mem = server_with_mock
        mem.get.side_effect = RuntimeError("connection lost")
        fn = _get_tool_fn(srv, "get_memory")
        result = fn(memory_id="uuid-123")
        parsed = json.loads(result)
        assert "error" in parsed
        assert "connection lost" in parsed.get("detail", "")


class TestInitMemory:
    @patch("mem0_mcp_selfhosted.server.patch_graph_sanitizer")
    @patch("mem0.Memory.from_config")
    @patch("mem0.utils.factory.LlmFactory.register_provider")
    @patch("mem0_mcp_selfhosted.server.build_config")
    def test_registers_both_providers_anthropic(
        self, mock_bc, mock_reg, mock_from_config, mock_patch
    ):
        mock_memory = MagicMock()
        mock_memory.graph = None
        mock_from_config.return_value = mock_memory
        mock_bc.return_value = (
            {"llm": {"provider": "anthropic", "config": {}}},
            [
                {
                    "name": "ollama",
                    "class_path": "mem0_mcp_selfhosted.llm_ollama.OllamaToolLLM",
                },
                {
                    "name": "anthropic",
                    "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
                },
            ],
            None,
        )
        server_mod._init_memory()
        assert mock_reg.call_count == 2  # Both Ollama and Anthropic registered
        mock_patch.assert_called_once()

    @patch("mem0_mcp_selfhosted.server.patch_graph_sanitizer")
    @patch("mem0.Memory.from_config")
    @patch("mem0.utils.factory.LlmFactory.register_provider")
    @patch("mem0_mcp_selfhosted.server.build_config")
    def test_registers_ollama_only(self, mock_bc, mock_reg, mock_from_config, mock_patch):
        """When LLM is ollama, only ollama provider is registered (always included)."""
        mock_memory = MagicMock()
        mock_memory.graph = None
        mock_from_config.return_value = mock_memory
        mock_bc.return_value = (
            {"llm": {"provider": "ollama", "config": {}}},
            [
                {
                    "name": "ollama",
                    "class_path": "mem0_mcp_selfhosted.llm_ollama.OllamaToolLLM",
                },
            ],
            None,
        )
        server_mod._init_memory()
        mock_reg.assert_called_once()
        call_kwargs = mock_reg.call_args
        assert call_kwargs.kwargs["name"] == "ollama"

    @patch("mem0_mcp_selfhosted.server.patch_graph_sanitizer")
    @patch("mem0.Memory.from_config")
    @patch("mem0_mcp_selfhosted.server.build_config")
    def test_patches_graph_sanitizer(self, mock_bc, mock_from_config, mock_patch):
        call_order = []
        mock_patch.side_effect = lambda: call_order.append("patch_graph_sanitizer")
        mock_memory = MagicMock()
        mock_memory.graph = None
        mock_from_config.side_effect = lambda cfg: (
            call_order.append("from_config") or mock_memory
        )
        mock_bc.return_value = (
            {"llm": {"provider": "ollama", "config": {}}},
            [],
            None,
        )
        server_mod._init_memory()
        mock_patch.assert_called_once()
        assert call_order.index("patch_graph_sanitizer") < call_order.index("from_config"), (
            f"patch_graph_sanitizer must be called before Memory.from_config, got: {call_order}"
        )


class TestResolveConfigClass:
    def test_anthropic_oat_returns_config(self):
        """anthropic_oat resolves to AnthropicOATConfig (same as anthropic)."""
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        result = server_mod._resolve_config_class("anthropic_oat")
        assert result is AnthropicOATConfig

    def test_anthropic_returns_config(self):
        """anthropic still resolves to AnthropicOATConfig."""
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        result = server_mod._resolve_config_class("anthropic")
        assert result is AnthropicOATConfig

    def test_openai_returns_config(self):
        """openai resolves to OpenAIConfig."""
        from mem0.configs.llms.openai import OpenAIConfig

        result = server_mod._resolve_config_class("openai")
        assert result is OpenAIConfig

    def test_unknown_returns_none(self):
        """Unknown provider returns None."""
        assert server_mod._resolve_config_class("unknown") is None


class TestRegisterProviders:
    @patch("mem0.utils.factory.LlmFactory.register_provider")
    def test_anthropic_oat_registers_without_error(self, mock_reg):
        """anthropic_oat provider registers successfully with LlmFactory."""
        server_mod.register_providers([
            {"name": "anthropic_oat", "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM"},
        ])
        mock_reg.assert_called_once()
        assert mock_reg.call_args[1]["name"] == "anthropic_oat"

    @patch("mem0.utils.factory.LlmFactory.register_provider")
    def test_unknown_provider_logs_warning(self, mock_reg):
        """Unknown provider name logs a warning and is skipped."""
        with patch("mem0_mcp_selfhosted.server.logger") as mock_logger:
            server_mod.register_providers([
                {"name": "unknown_provider", "class_path": "some.module.SomeClass"},
            ])
        mock_logger.warning.assert_called_once()
        assert "unknown_provider" in mock_logger.warning.call_args[0][1]
        mock_reg.assert_not_called()


class TestCreateServer:
    def test_registers_11_tools(self):
        srv = server_mod._create_server()
        tools = srv._tool_manager._tools
        assert len(tools) == 11, f"Expected 11 tools, got {len(tools)}: {list(tools.keys())}"

    def test_registers_prompt(self):
        srv = server_mod._create_server()
        prompts = srv._prompt_manager._prompts
        assert "memory_assistant" in prompts

    def test_tools_register_without_memory(self):
        """Server creates tools even when memory is None (lazy init)."""
        original = server_mod.memory
        server_mod.memory = None
        try:
            srv = server_mod._create_server()
            tools = srv._tool_manager._tools
            assert len(tools) == 11
        finally:
            server_mod.memory = original


# ============================================================
# 4. Lazy Memory Initialization Tests
# ============================================================


class TestEnsureMemory:
    @pytest.fixture(autouse=True)
    def _reset_lazy_state(self):
        """Reset lazy init state before each test."""
        original_memory = server_mod.memory
        original_failure = server_mod._last_init_failure
        server_mod.memory = None
        server_mod._last_init_failure = 0.0
        yield
        server_mod.memory = original_memory
        server_mod._last_init_failure = original_failure

    @patch("mem0_mcp_selfhosted.server._init_memory")
    def test_lazy_init_on_first_call(self, mock_init):
        """_ensure_memory() triggers _init_memory() on first call."""
        mock_mem = MagicMock()
        mock_mem.graph = None

        def set_memory():
            server_mod.memory = mock_mem

        mock_init.side_effect = set_memory
        result = server_mod._ensure_memory()
        mock_init.assert_called_once()
        assert result is mock_mem

    def test_returns_cached_memory(self, mock_memory):
        """_ensure_memory() returns cached memory without re-init."""
        server_mod.memory = mock_memory
        result = server_mod._ensure_memory()
        assert result is mock_memory

    @patch("mem0_mcp_selfhosted.server._init_memory")
    def test_returns_none_on_failure(self, mock_init):
        """_ensure_memory() returns None when init fails."""
        mock_init.side_effect = ConnectionError("Qdrant unreachable")
        result = server_mod._ensure_memory()
        assert result is None
        mock_init.assert_called_once()

    @patch("mem0_mcp_selfhosted.server._init_memory")
    def test_no_retry_during_cooldown(self, mock_init):
        """_ensure_memory() skips retry within cooldown period."""
        mock_init.side_effect = ConnectionError("Qdrant unreachable")
        # First call fails
        server_mod._ensure_memory()
        mock_init.reset_mock()
        # Second call within cooldown — should NOT retry
        result = server_mod._ensure_memory()
        assert result is None
        mock_init.assert_not_called()

    @patch("mem0_mcp_selfhosted.server._init_memory")
    @patch("mem0_mcp_selfhosted.server.time")
    def test_retries_after_cooldown(self, mock_time, mock_init):
        """_ensure_memory() retries after cooldown expires."""
        mock_init.side_effect = ConnectionError("Qdrant unreachable")
        mock_time.monotonic.return_value = 100.0
        # First call fails at t=100
        server_mod._ensure_memory()
        mock_init.reset_mock()
        # Second call at t=131 (past 30s cooldown)
        mock_time.monotonic.return_value = 131.0
        mock_init.side_effect = ConnectionError("still down")
        server_mod._ensure_memory()
        mock_init.assert_called_once()


class TestToolsWithNoMemory:
    """Test that tools return structured errors when memory is unavailable."""

    @pytest.fixture(autouse=True)
    def _no_memory(self):
        """Ensure memory is None and lazy init fails."""
        original = server_mod.memory
        original_failure = server_mod._last_init_failure
        server_mod.memory = None
        server_mod._last_init_failure = 0.0
        yield
        server_mod.memory = original
        server_mod._last_init_failure = original_failure

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_add_memory_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "add_memory")
        result = fn(text="test")
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_search_memories_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "search_memories")
        result = fn(query="test")
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_get_memories_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "get_memories")
        result = fn()
        parsed = json.loads(result)
        assert parsed["error"] == "Memory not initialized"

    @patch("mem0_mcp_selfhosted.server._init_memory", side_effect=ConnectionError("no qdrant"))
    def test_get_memory_returns_error(self, mock_init):
        srv = server_mod._create_server()
        fn = _get_tool_fn(srv, "get_memory")
        result = fn(memory_id="uuid-123")
        parsed = json.loads(result)
        assert parsed["error"] == "Memory not initialized"
