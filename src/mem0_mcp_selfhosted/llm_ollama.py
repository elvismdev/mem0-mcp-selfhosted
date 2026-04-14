"""Custom Ollama LLM provider with tool-calling and defense-in-depth.

Restores tool-call support removed in mem0ai PR #3241 and adds six
defensive layers to prevent silent data loss when Ollama returns empty
or malformed JSON (caused by the documented <think> + format:"json"
incompatibility — Ollama issues #10538, #10929, #10976).

Defense layers (applied in order):
  1. /no_think injection — suppresses qwen3 thinking tokens before API call
  2. temperature=0, repeat_penalty=1.0 — deterministic structured output
  3. keep_alive — prevents model unload between sequential graph pipeline calls
  4. Think-tag stripping — removes leaked <think> blocks from response content
  5. extract_json() — strips code fences / text prefixes from JSON responses
  6. Single retry — retries once on empty or invalid JSON
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List

from mem0_mcp_selfhosted.env import env

from mem0.llms.ollama import OllamaLLM

logger = logging.getLogger(__name__)


def extract_json(text: str) -> str:
    """Extract JSON from potentially wrapped text.

    Handles code-fenced JSON, text-prefixed JSON, and clean JSON.
    Identical to llm_anthropic.py's version — kept as a peer copy to
    avoid cross-module coupling between providers.
    """
    text = text.strip()
    if not text:
        return text

    # Try code-fenced JSON (closed fence)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)

    # Try unclosed code fence
    match = re.search(r"```(?:json)?\s*([\s\S]*)", text)
    if match:
        return match.group(1).strip()

    # Try text-prefixed JSON
    if text[0] not in ("{", "["):
        obj_idx = text.find("{")
        arr_idx = text.find("[")
        candidates = [i for i in (obj_idx, arr_idx) if i >= 0]
        if candidates:
            return text[min(candidates):]

    return text


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and unclosed <think> tags."""
    # Strip closed think blocks
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Strip unclosed think tag (everything from <think> to end)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip()


class OllamaToolLLM(OllamaLLM):
    """Ollama LLM with tool-calling support and defense-in-depth layers."""

    def _parse_response(self, response, tools):
        """Parse response with think-tag stripping and tool_calls extraction.

        Handles both modern Ollama SDK ToolCall objects and legacy dict format.
        Think tags are stripped from content for ALL response types.
        """
        # Extract content from response (handles both dict and object)
        if isinstance(response, dict):
            content = response["message"]["content"]
        else:
            content = response.message.content

        # Layer 4: Strip think tags from content (applies to all responses)
        if content:
            content = _strip_think_tags(content)

        if tools:
            processed_response: Dict = {
                "content": content,
                "tool_calls": [],
            }

            # Extract tool_calls from response
            tool_calls_data = None
            if isinstance(response, dict):
                tool_calls_data = response.get("message", {}).get("tool_calls")
            elif hasattr(response, "message") and hasattr(response.message, "tool_calls"):
                tool_calls_data = response.message.tool_calls

            if tool_calls_data:
                for tc in tool_calls_data:
                    if isinstance(tc, dict):
                        processed_response["tool_calls"].append({
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        })
                    else:
                        processed_response["tool_calls"].append({
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        })

            return processed_response
        else:
            return content

    def _is_json_valid(self, text: str) -> bool:
        """Check if text is non-empty, parseable JSON, and not just {}."""
        if not text or not text.strip():
            return False
        try:
            parsed = json.loads(text)
            if parsed == {}:
                return False
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: List[Dict] | None = None,
        tool_choice: str = "auto",  # Accepted for API compat; Ollama has no equivalent
        **kwargs,
    ):
        """Generate a response with defense-in-depth layers.

        Layers applied:
          1. /no_think injection (before API call)
          2. temperature=0, repeat_penalty=1.0 for structured requests (in options)
          3. keep_alive parameter (in API call)
         3b. Explicit ``think`` API parameter (Ollama ≥0.9.0) — definitive
              control; silently ignored by non-thinking models
          4. Think-tag stripping (in _parse_response)
          5. extract_json() (after _parse_response, JSON-mode only)
          6. Single retry on empty/invalid JSON (wraps the pipeline)
        """
        # Copy messages to avoid mutating the caller's list
        messages = [dict(m) for m in messages]

        is_json = bool(
            response_format and response_format.get("type") == "json_object"
        )
        has_tools = bool(tools)

        # Layer 1: /no_think injection
        think_enabled = env("MEM0_OLLAMA_THINK").lower() in (
            "true", "1", "yes",
        )
        if not think_enabled:
            # Find last user message and inject /no_think
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if not re.search(r"/(no_)?think\b", content):
                        msg["content"] = content + " /no_think"
                    break

        params = {
            "model": self.config.model,
            "messages": messages,
        }

        # Handle JSON response format
        if is_json:
            params["format"] = "json"
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nPlease respond with valid JSON only."
            else:
                messages.append({"role": "user", "content": "Please respond with valid JSON only."})

        # Layer 2: Deterministic options for structured requests
        options = {
            "num_predict": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if is_json or has_tools:
            options["temperature"] = 0
            options["repeat_penalty"] = 1.0
        else:
            options["temperature"] = self.config.temperature
        params["options"] = options

        # Layer 3: keep_alive
        keep_alive = env("MEM0_OLLAMA_KEEP_ALIVE", "30m")
        # Env vars are always strings, but Ollama's Go duration parser
        # rejects numeric strings like "-1" or "300" — they must be sent as
        # ints. Coerce numeric strings; pass duration strings ("30m", "1h")
        # through unchanged. Without this, setting MEM0_OLLAMA_KEEP_ALIVE=-1
        # (to keep a model resident forever) breaks every call with
        # `ResponseError: time: missing unit in duration "-1"`.
        try:
            keep_alive = int(keep_alive)
        except (ValueError, TypeError):
            pass
        params["keep_alive"] = keep_alive

        # Layer 3b: Explicit thinking control via Ollama API (requires v0.9.0+).
        # The /no_think text injection (Layer 1) is unreliable as a sole
        # mechanism — thinking models may still produce thinking tokens that
        # consume the num_predict budget and return empty content.  The API
        # parameter is definitive.  For non-thinking models (llama3, mistral,
        # gemma, …) the parameter is silently ignored by Ollama.
        params["think"] = think_enabled

        # Pass tools to Ollama (restored from upstream PR #3241)
        if has_tools:
            params["tools"] = tools

        # Execute API call
        response = self.client.chat(**params)
        result = self._parse_response(response, tools)

        # Layer 5: extract_json() for JSON-mode (not tool-calling)
        if is_json and not has_tools and isinstance(result, str):
            result = extract_json(result)

        # Layer 6: Single retry for JSON-mode (not tool-calling)
        if is_json and not has_tools and isinstance(result, str):
            if not self._is_json_valid(result):
                logger.warning("Empty or invalid JSON from Ollama, retrying once")
                response = self.client.chat(**params)
                result = self._parse_response(response, tools)
                if isinstance(result, str):
                    result = extract_json(result)
                if isinstance(result, str) and not self._is_json_valid(result):
                    logger.error(
                        "Retry also returned empty/invalid JSON — returning as-is"
                    )

        return result
