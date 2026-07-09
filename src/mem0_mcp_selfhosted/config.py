"""Environment-driven configuration for mem0-mcp-selfhosted.

Reads all config from env vars with sensible defaults, constructs a
mem0ai MemoryConfig dict, and returns provider registration info.
"""

from __future__ import annotations

import os
from typing import Any, TypedDict

from mem0_mcp_selfhosted.auth import resolve_token
from mem0_mcp_selfhosted.env import bool_env, env, opt_env, prompt_env


class ProviderInfo(TypedDict):
    """Custom LLM provider registration info for LlmFactory."""

    name: str
    class_path: str


def _resolve_ollama_url(*env_keys: str) -> str:
    """Resolve the Ollama base URL from a priority chain of env vars.

    Checks each key in *env_keys* first, then falls back to
    ``MEM0_OLLAMA_URL``, then ``"http://localhost:11434"``.
    """
    for key in env_keys:
        val = env(key)
        if val:
            return val
    return env("MEM0_OLLAMA_URL") or "http://localhost:11434"


def build_config() -> tuple[dict[str, Any], list[ProviderInfo], dict[str, Any] | None]:
    """Build mem0ai MemoryConfig dict and provider registration info.

    Returns:
        (config_dict, providers_info, split_config) where:
        - providers_info: list of ProviderInfo dicts (name + class_path)
        - split_config: if gemini_split was requested, config for the SplitModelGraphLLM
    """
    token = resolve_token()

    # --- Top-level provider default (cascades to LLM and graph LLM) ---
    _provider_default = env("MEM0_PROVIDER", "anthropic")
    _supported_llm_providers = ("anthropic", "ollama")
    if _provider_default not in _supported_llm_providers:
        raise ValueError(
            f"Unsupported MEM0_PROVIDER={_provider_default!r}. "
            f"Supported: {list(_supported_llm_providers)}"
        )

    # --- LLM ---
    llm_provider = env("MEM0_LLM_PROVIDER", _provider_default)
    if llm_provider not in _supported_llm_providers:
        raise ValueError(
            f"Unsupported MEM0_LLM_PROVIDER={llm_provider!r}. "
            f"Supported: {list(_supported_llm_providers)}"
        )

    _llm_model_defaults = {"anthropic": "claude-opus-4-6", "ollama": "qwen3:14b"}
    llm_model = env("MEM0_LLM_MODEL", _llm_model_defaults[llm_provider])
    llm_max_tokens = int(env("MEM0_LLM_MAX_TOKENS", "16384"))

    llm_config: dict[str, Any] = {"model": llm_model}
    if llm_provider == "anthropic":
        llm_config["max_tokens"] = llm_max_tokens
        if token:
            llm_config["api_key"] = token
    elif llm_provider == "ollama":
        llm_config["ollama_base_url"] = _resolve_ollama_url("MEM0_LLM_URL")

    # --- Embedder ---
    embed_provider = env("MEM0_EMBED_PROVIDER", "ollama")
    embed_model = env("MEM0_EMBED_MODEL", "bge-m3")
    embed_url = _resolve_ollama_url("MEM0_EMBED_URL")
    embed_dims = int(env("MEM0_EMBED_DIMS", "1024"))

    embedder_config: dict[str, Any] = {
        "model": embed_model,
    }
    if embed_provider == "ollama":
        embedder_config["ollama_base_url"] = embed_url

    # --- Vector Store ---
    qdrant_url = env("MEM0_QDRANT_URL", "http://localhost:6333")
    collection = env("MEM0_COLLECTION", "mem0_mcp_selfhosted")
    qdrant_api_key = opt_env("MEM0_QDRANT_API_KEY")
    qdrant_on_disk = bool_env("MEM0_QDRANT_ON_DISK")

    vector_config: dict[str, Any] = {
        "collection_name": collection,
        "url": qdrant_url,
        "embedding_model_dims": embed_dims,
    }
    if qdrant_api_key:
        vector_config["api_key"] = qdrant_api_key
    if qdrant_on_disk:
        vector_config["on_disk"] = True
    qdrant_timeout = opt_env("MEM0_QDRANT_TIMEOUT")
    if qdrant_timeout:
        # QdrantConfig's Pydantic model does not accept "timeout" directly.
        # Create a pre-configured QdrantClient with the timeout and pass it
        # via the "client" field, which mem0ai uses as-is.
        from qdrant_client import QdrantClient

        client_kwargs: dict[str, Any] = {
            "url": qdrant_url,
            "timeout": int(qdrant_timeout),
        }
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key
        vector_config["client"] = QdrantClient(**client_kwargs)

    # --- History ---
    history_db_path = opt_env("MEM0_HISTORY_DB_PATH")

    # --- Custom prompts ---
    # mem0ai's default fact-extraction prompt ("Personal Information
    # Organizer") returns {"facts": []} for technical/project content, so
    # infer=true silently stores nothing. These overrides let deployments
    # tailor extraction to their domain. Each var has a _FILE companion
    # (see prompt_env) for multi-line prompts.
    custom_fact_extraction_prompt = prompt_env("MEM0_CUSTOM_FACT_EXTRACTION_PROMPT")
    custom_update_memory_prompt = prompt_env("MEM0_CUSTOM_UPDATE_MEMORY_PROMPT")

    # --- Build config dict ---
    config_dict: dict[str, Any] = {
        "llm": {
            "provider": llm_provider,
            "config": llm_config,
        },
        "embedder": {
            "provider": embed_provider,  # Explicit — never rely on mem0ai's openai default
            "config": embedder_config,
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_config,
        },
        "version": "v1.1",
    }

    if history_db_path:
        config_dict["history_db_path"] = history_db_path

    if custom_fact_extraction_prompt:
        config_dict["custom_fact_extraction_prompt"] = custom_fact_extraction_prompt
    if custom_update_memory_prompt:
        config_dict["custom_update_memory_prompt"] = custom_update_memory_prompt

    # --- Graph Store (conditional) ---
    enable_graph = bool_env("MEM0_ENABLE_GRAPH")
    graph_llm_provider_raw: str | None = None  # set inside block, used for provider registration
    if enable_graph:
        neo4j_url = env("MEM0_NEO4J_URL", "bolt://127.0.0.1:7687")
        neo4j_user = env("MEM0_NEO4J_USER", "neo4j")
        neo4j_password = env("MEM0_NEO4J_PASSWORD", "mem0graph")
        neo4j_database = opt_env("MEM0_NEO4J_DATABASE")
        neo4j_base_label = opt_env("MEM0_NEO4J_BASE_LABEL")
        graph_threshold = float(env("MEM0_GRAPH_THRESHOLD", "0.7"))

        graph_neo4j_config: dict[str, Any] = {
            "url": neo4j_url,
            "username": neo4j_user,
            "password": neo4j_password,
        }
        if neo4j_database:
            # WORKAROUND: mem0ai's graph_memory.py passes config values as
            # positional args to Neo4jGraph(url, username, password, ...) where
            # the 4th param is `token`, NOT `database`. Putting database in the
            # config dict causes it to land in token → AuthError.
            # Set NEO4J_DATABASE env var instead — langchain_neo4j reads it via
            # get_from_dict_or_env(). Upstream: mem0ai #3906, #3981, #4085.
            # NOTE: Intentional process-global mutation — Neo4jGraph reads this
            # env var at init time, which happens after build_config() returns.
            os.environ["NEO4J_DATABASE"] = neo4j_database
        if neo4j_base_label:
            graph_neo4j_config["base_label"] = neo4j_base_label

        # Graph LLM — MUST be explicit (mem0ai defaults to "openai" if omitted)
        graph_llm_provider_raw = env("MEM0_GRAPH_LLM_PROVIDER", _provider_default)
        graph_llm_provider = graph_llm_provider_raw
        graph_llm_model = env("MEM0_GRAPH_LLM_MODEL", llm_model)

        graph_llm_config: dict[str, Any] = {
            "model": graph_llm_model,
        }

        if graph_llm_provider == "ollama":
            graph_llm_config["ollama_base_url"] = _resolve_ollama_url(
                "MEM0_GRAPH_LLM_URL", "MEM0_LLM_URL"
            )
        elif graph_llm_provider in ("anthropic", "anthropic_oat"):
            if token:
                graph_llm_config["api_key"] = token
            graph_llm_config["max_tokens"] = llm_max_tokens
        elif graph_llm_provider == "gemini":
            # Use mem0ai's built-in GeminiLLM provider
            # Default to flash-lite (not the main Claude model) when no explicit model set
            graph_llm_config["model"] = env(
                "MEM0_GRAPH_LLM_MODEL", "gemini-2.5-flash-lite"
            )
            google_api_key = opt_env("GOOGLE_API_KEY")
            if google_api_key:
                graph_llm_config["api_key"] = google_api_key
        elif graph_llm_provider == "gemini_split":
            # Split-model router: Gemini for extraction, separate LLM for contradiction.
            # Use "gemini" as config provider (passes pydantic validation), then
            # server.py swaps the graph LLM to the SplitModelGraphLLM after creation.
            graph_llm_config["model"] = env(
                "MEM0_GRAPH_LLM_MODEL", "gemini-2.5-flash-lite"
            )
            google_api_key = opt_env("GOOGLE_API_KEY")
            if google_api_key:
                graph_llm_config["api_key"] = google_api_key
            # Override provider to "gemini" for pydantic validation
            graph_llm_provider = "gemini"

        config_dict["graph_store"] = {
            "provider": "neo4j",
            "config": graph_neo4j_config,
            "threshold": graph_threshold,
            "llm": {
                "provider": graph_llm_provider,
                "config": graph_llm_config,
            },
        }

        # Custom entity-extraction prompt — mem0ai injects it as an extra
        # numbered instruction into the graph extraction system prompt
        # (GraphStoreConfig.custom_prompt).
        graph_custom_prompt = prompt_env("MEM0_GRAPH_CUSTOM_PROMPT")
        if graph_custom_prompt:
            config_dict["graph_store"]["custom_prompt"] = graph_custom_prompt

    # --- Provider registration info ---
    # Always register custom Ollama provider — strict superset of upstream
    # OllamaLLM (restores tool-calling removed in mem0ai PR #3241).
    # Registering even when not used has no side effects.
    providers_info: list[ProviderInfo] = [
        {
            "name": "ollama",
            "class_path": "mem0_mcp_selfhosted.llm_ollama.OllamaToolLLM",
        },
    ]
    # Register Anthropic when used as main LLM, graph LLM, or contradiction LLM
    contradiction_provider = env(
        "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER", "anthropic"
    )
    _needs_anthropic = (
        llm_provider == "anthropic"
        or (enable_graph and graph_llm_provider_raw in ("anthropic", "anthropic_oat"))
        or (enable_graph and graph_llm_provider_raw == "gemini_split"
            and contradiction_provider in ("anthropic", "anthropic_oat"))
    )
    if _needs_anthropic:
        providers_info.append({
            "name": "anthropic",
            "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
        })

    # Split-model config: if gemini_split was requested, provide the config
    # for server.py to swap the graph LLM after Memory creation.
    split_config: dict[str, Any] | None = None
    if enable_graph and graph_llm_provider_raw == "gemini_split":
        extraction_model = env("MEM0_GRAPH_LLM_MODEL", "gemini-2.5-flash-lite")
        google_api_key = opt_env("GOOGLE_API_KEY")
        contradiction_provider = env(
            "MEM0_GRAPH_CONTRADICTION_LLM_PROVIDER", "anthropic"
        )
        # Provider-aware default: when contradiction provider is anthropic,
        # default to a Claude model (not the main LLM model which may be Ollama).
        _contradiction_model_defaults = {
            "anthropic": "claude-opus-4-6",
            "anthropic_oat": "claude-opus-4-6",
        }
        contradiction_model = env(
            "MEM0_GRAPH_CONTRADICTION_LLM_MODEL",
            _contradiction_model_defaults.get(contradiction_provider, llm_model),
        )
        split_config = {
            "extraction_provider": "gemini",
            "extraction_model": extraction_model,
            "contradiction_provider": contradiction_provider,
            "contradiction_model": contradiction_model,
            "contradiction_max_tokens": llm_max_tokens,
        }
        if google_api_key:
            split_config["extraction_api_key"] = google_api_key
        if contradiction_provider in ("anthropic", "anthropic_oat") and token:
            split_config["contradiction_api_key"] = token
        elif contradiction_provider == "ollama":
            split_config["contradiction_ollama_base_url"] = _resolve_ollama_url(
                "MEM0_GRAPH_LLM_URL", "MEM0_LLM_URL"
            )

    return config_dict, providers_info, split_config
