"""Split-model LLM router for the graph pipeline.

Routes graph pipeline calls to different backing LLMs based on tool names:
- Extraction tools (extract_entities, establish_relationships/relations) -> extraction LLM
- Contradiction tools (delete/update/add_graph_memory, noop) -> contradiction LLM

Registered as "gemini_split" via LlmFactory.register_provider().
"""

from __future__ import annotations

import logging
from typing import Any

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase
from mem0.utils.factory import LlmFactory

logger = logging.getLogger(__name__)

# Tool names that indicate extraction pipeline stages (Calls 1 & 2)
_EXTRACTION_TOOLS = frozenset({
    "extract_entities",
    "establish_relationships",
    "establish_relations",
})

# Tool names that indicate contradiction detection stage (Call 3)
_CONTRADICTION_TOOLS = frozenset({
    "delete_graph_memory",
    "update_graph_memory",
    "add_graph_memory",
    "noop",
})


class SplitModelGraphLLMConfig(BaseLlmConfig):
    """Config for the split-model graph LLM router.

    Carries settings for both backing LLMs. The inherited `model` field
    is unused by the router itself but satisfies BaseLlmConfig.
    """

    def __init__(
        self,
        extraction_provider: str = "gemini",
        extraction_model: str = "gemini-2.5-flash-lite",
        extraction_api_key: str | None = None,
        contradiction_provider: str = "anthropic",
        contradiction_model: str = "claude-opus-4-6",
        contradiction_api_key: str | None = None,
        contradiction_max_tokens: int = 16384,
        contradiction_ollama_base_url: str | None = None,
        contradiction_openai_base_url: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.extraction_provider = extraction_provider
        self.extraction_model = extraction_model
        self.extraction_api_key = extraction_api_key
        self.contradiction_provider = contradiction_provider
        self.contradiction_model = contradiction_model
        self.contradiction_api_key = contradiction_api_key
        self.contradiction_max_tokens = contradiction_max_tokens
        self.contradiction_ollama_base_url = contradiction_ollama_base_url
        self.contradiction_openai_base_url = contradiction_openai_base_url


class SplitModelGraphLLM(LLMBase):
    """LLM router that dispatches graph pipeline calls to different backing LLMs.

    Inspects the tool function name in each generate_response() call to decide
    which backing LLM should handle it. Both LLMs are created via LlmFactory.
    """

    def __init__(self, config: SplitModelGraphLLMConfig | None = None):
        super().__init__(config)
        if not isinstance(self.config, SplitModelGraphLLMConfig):
            base = self.config
            self.config = SplitModelGraphLLMConfig(
                model=base.model,
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                top_p=base.top_p,
                top_k=base.top_k,
            )

        # Build extraction LLM config
        extraction_config = {
            "model": self.config.extraction_model,
        }
        if self.config.extraction_api_key:
            extraction_config["api_key"] = self.config.extraction_api_key

        # Build contradiction LLM config
        contradiction_config = {
            "model": self.config.contradiction_model,
            "max_tokens": self.config.contradiction_max_tokens,
        }
        if self.config.contradiction_api_key:
            contradiction_config["api_key"] = self.config.contradiction_api_key
        if self.config.contradiction_ollama_base_url:
            contradiction_config["ollama_base_url"] = self.config.contradiction_ollama_base_url
        if self.config.contradiction_openai_base_url:
            contradiction_config["openai_base_url"] = self.config.contradiction_openai_base_url

        self.extraction_llm = LlmFactory.create(
            self.config.extraction_provider, extraction_config
        )
        self.contradiction_llm = LlmFactory.create(
            self.config.contradiction_provider, contradiction_config
        )

        logger.info(
            "SplitModelGraphLLM initialized: extraction=%s/%s, contradiction=%s/%s",
            self.config.extraction_provider,
            self.config.extraction_model,
            self.config.contradiction_provider,
            self.config.contradiction_model,
        )

    def _get_tool_name(self, tools: list[dict] | None) -> str | None:
        """Extract the first tool function name from the tools list."""
        if not tools:
            return None
        tool = tools[0]
        if "function" in tool:
            return tool["function"].get("name")
        return tool.get("name")

    def generate_response(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = "auto",
    ):
        """Route to the appropriate backing LLM based on tool name."""
        tool_name = self._get_tool_name(tools)

        if tool_name and tool_name in _CONTRADICTION_TOOLS:
            logger.debug("Routing to contradiction LLM (tool: %s)", tool_name)
            return self.contradiction_llm.generate_response(
                messages=messages,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )

        # Default: extraction LLM (extraction tools, unknown tools, or no tools)
        logger.debug("Routing to extraction LLM (tool: %s)", tool_name)
        return self.extraction_llm.generate_response(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
