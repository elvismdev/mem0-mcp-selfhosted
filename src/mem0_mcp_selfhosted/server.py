"""FastMCP server for mem0-mcp-selfhosted.

Orchestrates: tool registration → transport → lazy Memory init on first call.
Memory initialization is deferred to the first tool invocation via _ensure_memory(),
allowing the server to respond to MCP initialize/tools/list without live infrastructure.
All 11 MCP tools + memory_assistant prompt.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Annotated, Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from mem0_mcp_selfhosted.config import ProviderInfo, build_config
from mem0_mcp_selfhosted.env import bool_env, env
from mem0_mcp_selfhosted.graph_tools import get_entity, search_graph
from mem0_mcp_selfhosted.helpers import (
    _mem0_call,
    call_with_graph,
    get_default_user_id,
    list_entities_facet,
    patch_gemini_parse_response,
    patch_graph_sanitizer,
    safe_bulk_delete,
)

logger = logging.getLogger(__name__)

# --- Globals set during startup ---
memory = None
mcp: FastMCP | None = None
_enable_graph_default = False

# --- Lazy init state ---
_memory_init_lock = threading.Lock()
_last_init_failure: float = 0.0
_INIT_RETRY_COOLDOWN = 30.0  # seconds before retrying after a failed init


def register_providers(providers_info: list[ProviderInfo]) -> None:
    """Register custom LLM providers with mem0ai's LlmFactory.

    Maps provider names to their config classes and registers each.
    Config classes are lazy-imported to avoid pulling in unnecessary
    dependencies (e.g. ``anthropic`` package in Ollama-only mode).
    Safe to call multiple times (LlmFactory.register_provider is idempotent).
    """
    if not providers_info:
        return

    from mem0.utils.factory import LlmFactory

    for pi in providers_info:
        config_class = _resolve_config_class(pi["name"])
        if config_class is None:
            logger.warning("No config class for provider %r, skipping", pi["name"])
            continue
        LlmFactory.register_provider(
            name=pi["name"],
            class_path=pi["class_path"],
            config_class=config_class,
        )


def _resolve_config_class(provider_name: str) -> type | None:
    """Lazy-resolve the config class for a provider name.

    Imports are deferred so that unnecessary packages (e.g. ``anthropic``)
    are never loaded in a pure-Ollama setup.
    """
    if provider_name == "ollama":
        from mem0.configs.llms.ollama import OllamaConfig

        return OllamaConfig
    if provider_name in ("anthropic", "anthropic_oat"):
        from mem0_mcp_selfhosted.llm_anthropic import AnthropicOATConfig

        return AnthropicOATConfig
    if provider_name == "openai":
        from mem0.configs.llms.openai import OpenAIConfig

        return OpenAIConfig
    return None


def _init_memory() -> Any:
    """Initialize mem0ai Memory with config and registered providers."""
    global memory, _enable_graph_default

    config_dict, providers_info, split_config = build_config()

    register_providers(providers_info)

    # Patch mem0ai's relationship sanitizer before Memory init
    patch_graph_sanitizer()
    patch_gemini_parse_response()

    # Initialize Memory
    from mem0 import Memory

    memory = Memory.from_config(config_dict)

    # If split-model was requested, swap the graph LLM with the router
    if split_config and memory.graph is not None:
        from mem0_mcp_selfhosted.llm_router import SplitModelGraphLLM, SplitModelGraphLLMConfig

        router_config = SplitModelGraphLLMConfig(**split_config)
        memory.graph.llm = SplitModelGraphLLM(router_config)

    _enable_graph_default = bool_env("MEM0_ENABLE_GRAPH")
    return memory


def _ensure_memory() -> Any:
    """Lazy-initialize Memory on first tool call. Thread-safe with retry-after-delay.

    Returns the Memory instance, or None if initialization failed.
    After a failure, waits ``_INIT_RETRY_COOLDOWN`` seconds before retrying.
    Matches the lazy-init pattern used by ``graph_tools._get_driver()``.
    """
    global memory, _last_init_failure

    if memory is not None:
        return memory

    now = time.monotonic()
    if _last_init_failure and (now - _last_init_failure < _INIT_RETRY_COOLDOWN):
        return None  # Too soon to retry

    with _memory_init_lock:
        # Double-check after acquiring lock
        if memory is not None:
            return memory

        try:
            _init_memory()
            logger.info("mem0ai Memory initialized successfully (lazy)")
        except Exception as exc:
            _last_init_failure = time.monotonic()
            logger.error("Lazy Memory init failed: %s", exc)
            return None

    return memory


def _create_server() -> FastMCP:
    """Create and configure the FastMCP server with all tools and prompts."""
    global mcp

    host = env("MEM0_HOST", "0.0.0.0")
    port = int(env("MEM0_PORT", "8081"))

    mcp = FastMCP(
        "mem0",
        host=host,
        port=port,
        instructions=(
            "Memory tools for persistent cross-session memory. "
            "Use search_memories to find relevant context before starting work. "
            "Use add_memory to store important facts, preferences, and decisions. "
            "Use get_memories to browse stored memories with filters. "
            "Use search_graph to find relationships between entities. "
            "Use get_memory to retrieve a specific memory by ID. "
            "Use update_memory to modify existing memories. "
            "Use list_entities to see who/what has stored memories."
        ),
    )

    _register_tools(mcp)
    _register_prompts(mcp)

    return mcp


# ============================================================
# Memory Tools (7 tools)
# ============================================================


def _register_tools(mcp: FastMCP) -> None:
    """Register all 11 MCP tools on the server."""

    @mcp.tool()
    def add_memory(
        text: Annotated[str, Field(description="Text to store as a memory. Converted to messages format internally.")],
        messages: Annotated[list[dict] | None, Field(description="Structured conversation history (role/content dicts). When provided, takes precedence over text.")] = None,
        user_id: Annotated[str | None, Field(description="User scope identifier. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope identifier.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope identifier.")] = None,
        metadata: Annotated[dict | None, Field(description="Arbitrary metadata JSON to store alongside the memory.")] = None,
        infer: Annotated[bool | None, Field(description="If true (default), LLM extracts key facts. If false, stores raw text.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Override default graph toggle for this call.")] = None,
    ) -> str:
        """Store a new memory. Requires at least one of user_id, agent_id, or run_id."""
        uid = user_id or get_default_user_id()

        # Build messages for mem0ai
        if messages:
            msgs = messages
        else:
            msgs = [{"role": "user", "content": text}]

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if metadata:
            kwargs["metadata"] = metadata
        if infer is not None:
            kwargs["infer"] = infer

        mem = _ensure_memory()

        def _do_add():
            return mem.add(msgs, **kwargs)

        return _mem0_call(call_with_graph, mem, enable_graph, _enable_graph_default, _do_add)

    @mcp.tool()
    def search_memories(
        query: Annotated[str, Field(description="Natural language description of what to find.")],
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        filters: Annotated[dict | None, Field(description="Additional structured filter clauses.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of results.")] = None,
        threshold: Annotated[float | None, Field(description="Minimum relevance score (0.0-1.0).")] = None,
        rerank: Annotated[bool | None, Field(description="Whether to apply reranking.")] = None,
        enable_graph: Annotated[bool | None, Field(description="Override default graph toggle.")] = None,
    ) -> str:
        """Semantic search across existing memories."""
        uid = user_id or get_default_user_id()

        kwargs: dict[str, Any] = {"user_id": uid, "query": query}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if filters:
            kwargs["filters"] = filters
        if limit is not None:
            kwargs["limit"] = limit
        if threshold is not None:
            kwargs["threshold"] = threshold
        if rerank is not None:
            kwargs["rerank"] = rerank

        mem = _ensure_memory()

        def _do_search():
            return mem.search(**kwargs)

        return _mem0_call(call_with_graph, mem, enable_graph, _enable_graph_default, _do_search)

    @mcp.tool()
    def get_memories(
        user_id: Annotated[str | None, Field(description="User scope. Defaults to MEM0_USER_ID.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope.")] = None,
        limit: Annotated[int | None, Field(description="Maximum number of memories to return.")] = None,
    ) -> str:
        """Page through memories using filters instead of search."""
        uid = user_id or get_default_user_id()

        kwargs: dict[str, Any] = {"user_id": uid}
        if agent_id:
            kwargs["agent_id"] = agent_id
        if run_id:
            kwargs["run_id"] = run_id
        if limit is not None:
            kwargs["limit"] = limit

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)
        return _mem0_call(mem.get_all, **kwargs)

    @mcp.tool()
    def get_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to fetch.")],
    ) -> str:
        """Fetch a single memory by its ID."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)
        return _mem0_call(mem.get, memory_id)

    @mcp.tool()
    def update_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to update.")],
        text: Annotated[str, Field(description="Replacement text for the memory.")],
    ) -> str:
        """Overwrite an existing memory's text. Re-embeds and re-indexes."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_update():
            mem.update(memory_id, data=text)
            return {"message": "Memory updated successfully!"}

        return _mem0_call(_do_update)

    @mcp.tool()
    def delete_memory(
        memory_id: Annotated[str, Field(description="Exact memory UUID to delete.")],
    ) -> str:
        """Delete a single memory."""
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_delete():
            mem.delete(memory_id)
            return {"message": "Memory deleted successfully!"}

        return _mem0_call(_do_delete)

    @mcp.tool()
    def delete_all_memories(
        user_id: Annotated[str | None, Field(description="User scope to delete.")] = None,
        agent_id: Annotated[str | None, Field(description="Agent scope to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run scope to delete.")] = None,
    ) -> str:
        """Bulk-delete all memories in the given scope. Requires at least one filter.

        NEVER calls memory.delete_all() — uses safe bulk-delete instead.
        """
        uid = user_id or get_default_user_id()
        if not any([uid, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        filters: dict[str, Any] = {}
        if uid:
            filters["user_id"] = uid
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_bulk_delete():
            count = safe_bulk_delete(mem, filters, graph_enabled=_enable_graph_default)
            return {"message": f"Deleted {count} memories.", "count": count}

        return _mem0_call(_do_bulk_delete)

    # ============================================================
    # Entity Tools (2 tools)
    # ============================================================

    @mcp.tool()
    def list_entities() -> str:
        """List which users/agents/runs currently hold memories.

        Uses Qdrant Facet API (v1.12+) for server-side aggregation,
        with scroll+dedupe fallback for older versions.
        """
        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_list():
            return list_entities_facet(mem)

        return _mem0_call(_do_list)

    @mcp.tool()
    def delete_entities(
        user_id: Annotated[str | None, Field(description="User entity to delete (cascades to all memories).")] = None,
        agent_id: Annotated[str | None, Field(description="Agent entity to delete.")] = None,
        run_id: Annotated[str | None, Field(description="Run entity to delete.")] = None,
    ) -> str:
        """Delete an entity and cascade-delete all its memories.

        Functionally equivalent to delete_all_memories in self-hosted mode.
        """
        if not any([user_id, agent_id, run_id]):
            return json.dumps(
                {"error": "At least one scope (user_id, agent_id, or run_id) is required."},
                ensure_ascii=False,
            )

        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        mem = _ensure_memory()
        if mem is None:
            return json.dumps({"error": "Memory not initialized", "detail": "Infrastructure may be unavailable."}, ensure_ascii=False)

        def _do_delete_entity():
            count = safe_bulk_delete(mem, filters, graph_enabled=_enable_graph_default)
            return {"message": f"Entity deleted. Removed {count} memories.", "count": count}

        return _mem0_call(_do_delete_entity)

    # ============================================================
    # Direct Neo4j Graph Tools
    # ============================================================

    @mcp.tool()
    def mcp_search_graph(
        query: Annotated[str, Field(description="Entity or topic to search for (e.g., 'Python', 'TypeScript').")],
    ) -> str:
        """Search entities by name/id substring matching in Neo4j knowledge graph."""
        return search_graph(query)

    @mcp.tool()
    def mcp_get_entity(
        name: Annotated[str, Field(description="Exact entity name to look up.")],
    ) -> str:
        """Get all relationships for a specific entity (bidirectional)."""
        return get_entity(name)


# ============================================================
# MCP Prompt
# ============================================================


def _register_prompts(mcp: FastMCP) -> None:
    """Register MCP prompts."""

    @mcp.prompt()
    def memory_assistant() -> str:
        """Quick-start guide for using the mem0 memory server."""
        return (
            "You are using the mem0 MCP server for long-term memory management.\n\n"
            "Quick Start:\n"
            "1. Store memories: Use add_memory to save facts, preferences, or conversations\n"
            "2. Search memories: Use search_memories for semantic queries\n"
            "3. Browse memories: Use get_memories for filtered listing\n"
            "4. Update/Delete: Use update_memory and delete_memory for modifications\n"
            "5. Graph exploration: Use search_graph and get_entity for entity relationships\n\n"
            "Tips:\n"
            "- user_id is automatically injected from MEM0_USER_ID default\n"
            "- Set enable_graph=true to include knowledge graph results\n"
            "- Use infer=false to store raw text without LLM extraction\n"
            "- Use threshold on search_memories to filter by relevance score\n"
            "- Use filters for structured queries: {\"key\": {\"eq\": \"value\"}}\n"
        )


# ============================================================
# Server Runner
# ============================================================


def run_server() -> None:
    """Entry point: create server and run.

    Memory initialization is deferred to the first tool call via
    ``_ensure_memory()``, allowing the server to respond to MCP
    ``initialize`` and ``tools/list`` without live infrastructure.
    """
    # Configure logging
    log_level = env("MEM0_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s | %(message)s",
    )

    # Load .env file
    load_dotenv()

    # Create and run server (Memory init deferred to first tool call)
    server = _create_server()
    transport = env("MEM0_TRANSPORT", "stdio").lower()

    if transport == "sse":
        server.run(transport="sse")
    elif transport == "streamable-http":
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")
