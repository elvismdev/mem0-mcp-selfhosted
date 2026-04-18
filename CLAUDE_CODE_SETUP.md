# Running mem0-mcp-selfhosted with Claude Code (v2 compatibility fixes)

This branch carries two patches that are required to run `mem0-mcp-selfhosted` v0.3.2 against `mem0ai >= 2.0.0`. Without them, `add_memory` silently writes nothing and `search_memories` returns the wrong results.

## What works on this branch

- `add_memory` extracts facts from input text and writes them to Qdrant
- `search_memories` returns semantically relevant memories with similarity scores
- `get_memories` pages through all memories for the current user
- Memories persist across Claude Code sessions

## Prerequisites

- Qdrant on `localhost:6333`
- Ollama on `localhost:11434` with `bge-m3` pulled (1024-dim embeddings)
- An Anthropic API key exported as `ANTHROPIC_API_KEY`
- Python 3.12 and `uv` installed
- This repo cloned locally

Quick health check:
```bash
curl -s http://localhost:6333/readyz
curl -s http://localhost:11434/api/tags | jq '.models[].name'
echo "API key length: ${#ANTHROPIC_API_KEY}"   # ~108
```

## One-time project setup

```bash
cd /path/to/mem0-mcp-selfhosted
git checkout fix/claude-code-v2-compat
uv sync   # creates .venv with editable install of the patched source
```

## MCP registration

Register the server pointing at the local venv's entry point. Do **not** use `uvx --from <repo>`; `uv` caches builds and won't pick up source changes without an explicit `--reinstall` or version bump.

```bash
claude mcp add --scope user --transport stdio mem0 \
  --env MEM0_ANTHROPIC_TOKEN="$ANTHROPIC_API_KEY" \
  --env MEM0_USER_ID=your-handle \
  -- /path/to/mem0-mcp-selfhosted/.venv/bin/mem0-mcp-selfhosted
```

Confirm:
```bash
claude mcp list | grep mem0
# mem0: /path/.../mem0-mcp-selfhosted/.venv/bin/mem0-mcp-selfhosted  - ✓ Connected
```

## Why `MEM0_ANTHROPIC_TOKEN` and not `ANTHROPIC_API_KEY`

The custom Anthropic provider resolves auth in this order:

1. `MEM0_ANTHROPIC_TOKEN` env var
2. Claude Code's OAT token at `~/.claude/.credentials.json`
3. `ANTHROPIC_API_KEY` env var

Inside a Claude Code MCP subprocess the OAT is always present at priority 2, so priority 3 is never reached. The OAT works for the API but carries stricter rate limits than a pay-as-you-go key. Setting `MEM0_ANTHROPIC_TOKEN` forces the provider to use your regular API key instead.

## The two patches

### 1. v2 search API shape (`server.py`)

`mem0ai` 2.0 changed `search()` and `get_all()` from accepting `user_id=` as a keyword argument to expecting `filters={"user_id": ...}`. The `search_memories` and `get_memories` tools now build a `filters` dict that folds in `user_id`, `agent_id`, `run_id`, and any caller-supplied `filters`.

### 2. Structured-output schema mismatch (`llm_anthropic.py`)

`llm_anthropic.py` used Anthropic's `output_config` to force Claude into a v1 JSON schema (`{"facts":[...]}` for extraction, `{"memory":[{"id","text","event","old_memory"}]}` for updates). `mem0ai` 2.0 expects a different shape (`{"memory":[{"id","text","attributed_to","linked_memory_ids"}]}`) driven by its v2 system prompt. Forcing the old schema made mem0 silently drop every `add_memory` because its v2 parser found no usable content in the response.

`_supports_structured_output()` now returns `False`, letting the system prompt drive the response shape; `extract_json()` handles parsing.

## Troubleshooting

### `add_memory` returns `{"results": []}`

**Stale history.** mem0 writes the last K messages to `~/.mem0/history.db` and includes them in the fact-extraction prompt. If earlier test messages have already been extracted, Claude treats new facts as already-known context.
```bash
rm -f ~/.mem0/history.db
```

**Wrong code in the MCP process.** If the MCP was registered via `uvx`, uv may have cached a pre-patch build. Re-register against `.venv/bin/mem0-mcp-selfhosted` as shown above.

**Auth not reaching the provider.** Confirm the env is attached to the MCP registration:
```bash
python3 -c "
import json, pathlib
m = json.loads(pathlib.Path.home().joinpath('.claude.json').read_text()).get('mcpServers', {}).get('mem0', {})
for k, v in (m.get('env') or {}).items():
    print(k, '=', ('***set***' if any(s in k for s in ('TOKEN','KEY')) else v))
"
```

### `Unsupported LLM provider: anthropic_custom`

You're calling `Memory.from_config()` directly instead of `server._init_memory()`. The provider name is `"anthropic"` — this project registers a custom implementation under that name, overriding the upstream one.

## End-to-end verification

```bash
rm -f ~/.mem0/history.db
claude   # fresh session
```

Inside that session:
> Please call `add_memory` on the mem0 server with: "Test fact: my name is Example and I ride a bike." Then call `get_memories` and show raw JSON.

Expect:
- `add_memory` result contains `"results": [ { "id": "<uuid>", "memory": "...", "event": "ADD" }, ... ]`
- `get_memories` returns the same items with `user_id` matching `MEM0_USER_ID`

## Maintenance

- `uv.lock` pins `mem0ai` to the version these patches were validated against. Don't run `uv lock --upgrade` unless you're prepared to retest; a minor `mem0ai` bump could change the v2 response shape again.
- If upstream merges the PR that includes these patches, you can drop this branch and return to `main`.
- To re-verify after any change, run a direct library test:
```bash
  cd /path/to/mem0-mcp-selfhosted
  uv run python -c "
  from mem0_mcp_selfhosted import server as s
  m = s._init_memory()
  print(m.add('Ping test.', user_id='healthcheck'))
  "
```
