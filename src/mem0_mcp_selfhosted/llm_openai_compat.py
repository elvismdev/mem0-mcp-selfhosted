"""OpenAI-compatible LLM provider for local servers (LM Studio, vLLM, llama.cpp).

Subclasses mem0ai's OpenAILLM and transforms unsupported response_format values.
LM Studio (and many local servers) reject {"type": "json_object"} — they only
accept {"type": "json_schema"} or {"type": "text"}.  mem0ai's prompts already
request JSON explicitly, so falling back to "text" is safe: the model still
returns JSON and mem0ai's JSON extraction logic handles it.
"""

from __future__ import annotations

import logging
from typing import Any

from mem0.llms.openai import OpenAILLM

logger = logging.getLogger(__name__)


class OpenAICompatLLM(OpenAILLM):
    """Drop-in replacement for OpenAILLM that strips json_object response_format.

    Registered as the "openai" provider so it transparently replaces the
    built-in when MEM0_LLM_PROVIDER=openai is used with a local server.
    """

    def generate_response(
        self,
        messages: list[dict[str, str]],
        response_format: Any = None,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ):
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            logger.debug(
                "OpenAICompatLLM: replacing unsupported response_format={'type': 'json_object'} "
                "with {'type': 'text'} for local server compatibility"
            )
            response_format = {"type": "text"}

        return super().generate_response(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
