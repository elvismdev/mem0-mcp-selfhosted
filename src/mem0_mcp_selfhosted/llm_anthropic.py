"""Custom Anthropic LLM provider for mem0ai with OAT support.

Fixes upstream tool-call parsing bug, adds structured outputs via
output_config, and injects Claude Code identity headers for OAT tokens.

Registered as "anthropic" via LlmFactory.register_provider(), overriding
the built-in provider to fix tool-call parsing and add OAT support.
"""

from __future__ import annotations

import logging
import re
import time

from mem0_mcp_selfhosted.env import env
from typing import Any

import anthropic
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.base import LLMBase

from mem0_mcp_selfhosted.auth import (
    is_oat_token,
    is_token_expiring_soon,
    read_credentials_full,
    refresh_oat_token,
    resolve_token,
)

logger = logging.getLogger(__name__)

# --- OAT Identity Headers ---
# Required for OAT token requests, matching what Claude Code sends.
OAT_HEADERS = {
    "accept": "application/json",
    "anthropic-dangerous-direct-browser-access": "true",
    "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
    "user-agent": "claude-cli/1.0.0 (external, cli)",
    "x-app": "cli",
}

# --- Structured Output Schemas ---
# Two schemas for the two call types in mem0ai's pipeline.

FACT_RETRIEVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memory"],
    "additionalProperties": False,
}

MEMORY_UPDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "event": {
                        "type": "string",
                        "enum": ["ADD", "UPDATE", "DELETE", "NONE"],
                    },
                    "old_memory": {"type": "string"},
                },
                "required": ["id", "text", "event"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["memory"],
    "additionalProperties": False,
}

# Model prefixes that support structured outputs (output_config).
_STRUCTURED_OUTPUT_PREFIXES = ("claude-opus-4", "claude-sonnet-4", "claude-haiku-4")


def extract_json(text: str) -> str:
    """Extract JSON from potentially wrapped text.

    Handles code-fenced JSON, text-prefixed JSON, and clean JSON.
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
            return text[min(candidates) :]

    return text


class AnthropicOATConfig(BaseLlmConfig):
    """Config for the custom Anthropic OAT LLM provider.

    Extends BaseLlmConfig (plain ABC, not Pydantic) with OAT-specific fields.
    """

    def __init__(self, auth_token: str | None = None, anthropic_base_url: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.auth_token = auth_token
        self.anthropic_base_url = anthropic_base_url


class AnthropicOATLLM(LLMBase):
    """Custom Anthropic LLM provider with OAT support and tool-call parsing.

    Two response paths:
    - Path 1 (response_format, no tools): Structured outputs via output_config
    - Path 2 (tools): Standard tool calling with _parse_response()
    """

    def __init__(self, config: AnthropicOATConfig | None = None):
        super().__init__(config)
        if not isinstance(self.config, AnthropicOATConfig):
            # Convert BaseLlmConfig → AnthropicOATConfig preserving attributes
            base = self.config
            self.config = AnthropicOATConfig(
                model=base.model,
                api_key=base.api_key,
                max_tokens=base.max_tokens,
                temperature=base.temperature,
                top_p=base.top_p,
                top_k=base.top_k,
            )

        # Resolve token: config field → fallback chain
        token = self.config.auth_token or self.config.api_key or resolve_token()
        self._build_client(token)

        # In-memory OAuth state for OAT token self-refresh
        self._refresh_token: str | None = None
        self._expires_at: int | None = None
        self._refresh_threshold: int = int(
            env("MEM0_OAT_REFRESH_THRESHOLD_SECONDS", "1800")
        )

        if token and is_oat_token(token):
            creds = read_credentials_full()
            if creds:
                self._refresh_token = creds["refresh_token"]
                self._expires_at = creds["expires_at"]

    def _build_client(self, token: str | None) -> None:
        """Build (or rebuild) the Anthropic client from a token.

        Handles OAT vs API key auth, OAT identity headers, base URL config.
        Updates self._current_token and self.client.
        """
        self._current_token = token

        client_kwargs: dict[str, Any] = {}
        if self.config.anthropic_base_url:
            client_kwargs["base_url"] = self.config.anthropic_base_url

        if token and is_oat_token(token):
            client_kwargs["auth_token"] = token
            # Inject OAT identity headers unless disabled
            oat_headers_mode = env("MEM0_OAT_HEADERS", "auto").lower()
            if oat_headers_mode != "none":
                client_kwargs["default_headers"] = OAT_HEADERS
        elif token:
            client_kwargs["api_key"] = token

        self.client = anthropic.Anthropic(**client_kwargs)

    # Transient HTTP status codes that warrant automatic retry.
    _RETRYABLE_STATUS_CODES = (500, 502, 503, 529)
    _MAX_RETRIES = 2
    _BACKOFF_SECONDS = (1, 2)

    def _try_piggyback_refresh(self) -> str | None:
        """Step 1: Re-read credentials file for a new token (piggyback on Claude Code).

        Returns the new token if different from current, or None.
        """
        new_token = resolve_token()
        if new_token and new_token != self._current_token:
            # Also update refresh token and expiry from the file
            creds = read_credentials_full()
            if creds:
                self._refresh_token = creds["refresh_token"]
                self._expires_at = creds["expires_at"]
            return new_token
        return None

    def _try_self_refresh(self) -> str | None:
        """Step 2: Mint a new token via OAuth endpoint using stored refresh token.

        Returns the new access token on success, or None.
        """
        if not self._refresh_token:
            return None

        result = refresh_oat_token(self._refresh_token)
        if result:
            self._refresh_token = result["refresh_token"]
            expires_in = result.get("expires_in")
            if expires_in:
                self._expires_at = int(time.time() * 1000) + (expires_in * 1000)
            return result["access_token"]
        return None

    def _try_wait_and_retry(self) -> str | None:
        """Step 3: Wait 2s for Claude Code to finish refreshing, then re-read.

        Returns the new token if different from current, or None.
        """
        time.sleep(2)
        new_token = resolve_token()
        if new_token and new_token != self._current_token:
            creds = read_credentials_full()
            if creds:
                self._refresh_token = creds["refresh_token"]
                self._expires_at = creds["expires_at"]
            return new_token
        return None

    def _proactive_refresh(self) -> None:
        """Check token expiry before API call and refresh proactively if needed.

        Attempts piggyback then self-refresh (skips wait-and-retry for proactive).
        On failure, proceeds silently — the normal 401 retry flow will handle it.
        """
        if not self._current_token or not is_oat_token(self._current_token):
            return
        if not is_token_expiring_soon(self._expires_at, self._refresh_threshold):
            return

        logger.info("[mem0] OAT token expiring soon, proactively refreshing")

        # Try piggyback first
        new_token = self._try_piggyback_refresh()
        if new_token:
            self._build_client(new_token)
            hours = (self._expires_at - int(time.time() * 1000)) / 3_600_000 if self._expires_at else 0
            logger.info("[mem0] OAT token proactively refreshed, expires in %.1fh", hours)
            return

        # Try self-refresh
        new_token = self._try_self_refresh()
        if new_token:
            self._build_client(new_token)
            hours = (self._expires_at - int(time.time() * 1000)) / 3_600_000 if self._expires_at else 0
            logger.info("[mem0] OAT token proactively refreshed, expires in %.1fh", hours)
            return

        # Proactive refresh failed — proceed with current token, 401 retry will handle it

    def _call_api(self, params: dict) -> anthropic.types.Message:
        """Call the Anthropic API with proactive refresh, auth retry, and transient retry.

        Flow:
        1. Proactive check: if OAT token expiring soon, refresh before calling
        2. Make API call with transient retry (500/502/503/529)
        3. On AuthenticationError with OAT token, 3-step defensive strategy:
           Step 1 (piggyback): re-read credentials file
           Step 2 (self-refresh): mint new token via OAuth endpoint
           Step 3 (wait-and-retry): sleep 2s, re-read credentials file
        4. On any step success, retry API call exactly once
        """
        # Proactive pre-expiry refresh
        self._proactive_refresh()

        try:
            response = self._call_with_transient_retry(params)
        except anthropic.AuthenticationError as auth_err:
            if not is_oat_token(self._current_token):
                raise

            # Step 1: Piggyback on credentials file
            new_token = self._try_piggyback_refresh()
            if new_token:
                logger.info("[mem0] OAT token expired, piggybacked on credentials file refresh")
                self._build_client(new_token)
                response = self._call_with_transient_retry(params)
            else:
                # Step 2: Self-refresh via OAuth
                new_token = self._try_self_refresh()
                if new_token:
                    logger.info("[mem0] OAT token expired, self-refreshed via OAuth endpoint")
                    self._build_client(new_token)
                    response = self._call_with_transient_retry(params)
                else:
                    # Step 3: Wait-and-retry
                    new_token = self._try_wait_and_retry()
                    if new_token:
                        logger.info("[mem0] OAT token expired, recovered after wait-and-retry")
                        self._build_client(new_token)
                        response = self._call_with_transient_retry(params)
                    else:
                        logger.error("[mem0] OAT token expired, all refresh strategies exhausted")
                        raise auth_err

        if response.stop_reason == "max_tokens":
            logger.warning(
                "[mem0] Anthropic response truncated for model %s", self.config.model
            )

        return response

    def _call_with_transient_retry(self, params: dict) -> anthropic.types.Message:
        """Call the API with retry-with-backoff for transient server errors."""
        last_exc: Exception | None = None
        for attempt in range(1 + self._MAX_RETRIES):
            try:
                return self.client.messages.create(**params)
            except anthropic.APIStatusError as exc:
                if exc.status_code not in self._RETRYABLE_STATUS_CODES:
                    raise
                last_exc = exc
                if attempt < self._MAX_RETRIES:
                    delay = self._BACKOFF_SECONDS[attempt]
                    logger.warning(
                        "[mem0] Anthropic %d error (attempt %d/%d), retrying in %ds",
                        exc.status_code, attempt + 1, 1 + self._MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    def _supports_structured_output(self) -> bool:
        """Check if the configured model supports structured outputs."""
        return self.config.model.startswith(_STRUCTURED_OUTPUT_PREFIXES)

    def _select_schema(self, messages: list[dict]) -> dict:
        """Select structured output schema based on call type.

        Detection: fact extraction calls always have a system message
        (the prompt template); memory update calls have only a user message.
        This is an intentional architectural invariant in mem0ai.
        """
        has_system = any(m.get("role") == "system" for m in messages)
        if has_system:
            return FACT_RETRIEVAL_SCHEMA
        return MEMORY_UPDATE_SCHEMA

    @staticmethod
    def _parse_response(response: anthropic.types.Message) -> dict:
        """Convert Anthropic tool_use blocks to dict format for graph_memory.py.

        Returns: {"content": "...", "tool_calls": [{"name": ..., "arguments": ...}]}
        """
        text_parts: list[str] = []
        tool_calls: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "name": block.name,
                    "arguments": block.input,  # Already a dict, no JSON parsing needed
                })

        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
        }

    def generate_response(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ):
        """Generate a response from the Anthropic API.

        Path 1 (tools provided): Standard tool calling with _parse_response().
        Path 2 (response_format, no tools): Structured output via output_config.
        No tools, no response_format: Plain text response.
        """
        # Separate system messages from user/assistant messages
        system_parts: list[str] = []
        api_messages: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(msg["content"])
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build base params
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": api_messages,
            "max_tokens": self.config.max_tokens or 4096,
        }
        # Anthropic rejects temperature + top_p together — only send temperature
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if system_parts:
            params["system"] = "\n\n".join(system_parts)

        # Path 2: Tool calling
        if tools:
            # Convert OpenAI-style tool defs to Anthropic format
            anthropic_tools = []
            for tool in tools:
                if "function" in tool:
                    fn = tool["function"]
                    anthropic_tools.append({
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    })
                else:
                    anthropic_tools.append(tool)
            params["tools"] = anthropic_tools

            if tool_choice:
                if isinstance(tool_choice, str):
                    if tool_choice == "required":
                        params["tool_choice"] = {"type": "any"}
                    elif tool_choice == "auto":
                        params["tool_choice"] = {"type": "auto"}
                    elif tool_choice == "none":
                        pass  # Don't set tool_choice
                    else:
                        params["tool_choice"] = {"type": "tool", "name": tool_choice}
                else:
                    params["tool_choice"] = tool_choice

            response = self._call_api(params)
            return self._parse_response(response)

        # Path 1: Structured output (response_format, no tools)
        if response_format:
            if self._supports_structured_output():
                schema = self._select_schema(messages)
                params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": schema,
                    },
                }
            # else: no output_config — rely on extractJson fallback

            response = self._call_api(params)
            if not response.content:
                logger.warning("Anthropic API returned empty content (structured output path)")
                return ""
            text = response.content[0].text
            return extract_json(text)

        # Plain text response (no tools, no response_format)
        response = self._call_api(params)
        if not response.content:
            logger.warning("Anthropic API returned empty content (plain text path)")
            return ""
        return response.content[0].text
