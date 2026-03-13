"""Config matrix tests — parametrized provider combination tests.

Verifies that build_config() produces valid config dicts for all meaningful
combinations of (llm_provider, graph_llm_provider, enable_graph).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mem0_mcp_selfhosted.config import build_config

# Base env vars required for all tests
_BASE_ENV = {
    "MEM0_QDRANT_URL": "http://localhost:6333",
    "MEM0_EMBED_URL": "http://localhost:11434",
    "MEM0_EMBED_MODEL": "bge-m3",
    "MEM0_EMBED_DIMS": "1024",
    "MEM0_NEO4J_URL": "bolt://127.0.0.1:7687",
    "MEM0_NEO4J_USER": "neo4j",
    "MEM0_NEO4J_PASSWORD": "testpass",
}


def _make_env(llm_provider: str, graph_provider: str | None, enable_graph: bool) -> dict:
    """Build env dict for a provider combination."""
    env = dict(_BASE_ENV)
    env["MEM0_LLM_PROVIDER"] = llm_provider
    env["MEM0_ENABLE_GRAPH"] = "true" if enable_graph else "false"
    if graph_provider:
        env["MEM0_GRAPH_LLM_PROVIDER"] = graph_provider
    if graph_provider == "gemini" or graph_provider == "gemini_split":
        env["GOOGLE_API_KEY"] = "test-google-key"
    return env


# (llm_provider, graph_provider, enable_graph, test_id)
MATRIX = [
    ("anthropic", None, False, "anthropic-no-graph"),
    ("ollama", None, False, "ollama-no-graph"),
    ("anthropic", "anthropic", True, "anthropic-anthropic-graph"),
    ("anthropic", "ollama", True, "anthropic-ollama-graph"),
    ("ollama", "ollama", True, "ollama-ollama-graph"),
    ("ollama", "anthropic", True, "ollama-anthropic-graph"),
    ("anthropic", "gemini", True, "anthropic-gemini-graph"),
    ("anthropic", "gemini_split", True, "anthropic-gemini-split"),
    ("ollama", "gemini_split", True, "ollama-gemini-split"),
]


class TestProviderMatrix:
    @pytest.mark.parametrize(
        "llm_provider,graph_provider,enable_graph,test_id",
        MATRIX,
        ids=[m[3] for m in MATRIX],
    )
    @patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-ant-api-fake")
    def test_provider_combination(
        self, mock_token, llm_provider, graph_provider, enable_graph, test_id, monkeypatch
    ):
        env = _make_env(llm_provider, graph_provider, enable_graph)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        # Clean up env vars not in this combination
        for key in ("MEM0_LLM_MODEL", "MEM0_GRAPH_LLM_MODEL", "MEM0_LLM_URL",
                     "MEM0_GRAPH_LLM_URL", "MEM0_EMBED_PROVIDER", "MEM0_OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        config_dict, providers_info, split_config = build_config()

        # Basic shape validation
        assert config_dict["llm"]["provider"] == llm_provider
        assert config_dict["embedder"]["provider"] == "ollama"
        assert config_dict["vector_store"]["provider"] == "qdrant"

        # Provider info — always includes Ollama, plus Anthropic when applicable
        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names  # Always registered
        needs_anthropic = (
            llm_provider == "anthropic"
            or (enable_graph and graph_provider in ("anthropic", "anthropic_oat"))
        )
        if needs_anthropic:
            assert "anthropic" in provider_names

        # Graph store
        if enable_graph:
            assert "graph_store" in config_dict
            gs = config_dict["graph_store"]
            assert gs["provider"] == "neo4j"
            if graph_provider == "gemini_split":
                # gemini_split overrides provider to "gemini" for pydantic
                assert gs["llm"]["provider"] == "gemini"
                assert split_config is not None
            elif graph_provider:
                assert gs["llm"]["provider"] == graph_provider
        else:
            assert "graph_store" not in config_dict
            assert split_config is None

        # Provider-specific config keys
        if llm_provider == "ollama":
            assert "ollama_base_url" in config_dict["llm"]["config"]
            assert "api_key" not in config_dict["llm"]["config"]
        elif llm_provider == "anthropic":
            assert "api_key" in config_dict["llm"]["config"]
            assert "max_tokens" in config_dict["llm"]["config"]


class TestDefaultModelAcrossMatrix:
    @pytest.mark.parametrize(
        "llm_provider,expected_model",
        [
            ("anthropic", "claude-opus-4-6"),
            ("ollama", "qwen3:14b"),
        ],
    )
    @patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-ant-api-fake")
    def test_default_model(self, mock_token, llm_provider, expected_model, monkeypatch):
        env = _make_env(llm_provider, None, False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)

        config_dict, _, _ = build_config()
        assert config_dict["llm"]["config"]["model"] == expected_model
