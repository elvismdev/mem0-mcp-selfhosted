"""Claude Code session hooks for mem0-mcp-selfhosted.

Three entry points registered in pyproject.toml:
- mem0-hook-context  -> context_main()   (SessionStart)
- mem0-hook-stop     -> stop_main()      (Stop)
- mem0-install-hooks -> install_main()   (CLI installer)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

from dotenv import load_dotenv

# Load .env early so _get_user_id() sees MEM0_USER_ID even when it's
# called before _get_memory().  load_dotenv(override=False) is the
# default — it never clobbers values already in os.environ.
load_dotenv()

# Hooks write JSON responses to stdout — logging must go to stderr
# so it never corrupts the hook response channel.
logging.basicConfig(stream=sys.stderr, format="%(levelname)s %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared initialization
# ---------------------------------------------------------------------------

_memory = None

_MAX_MEMORIES = 20
_MIN_USER_LEN = 20
_MIN_ASSISTANT_LEN = 50
_MAX_CONTENT_LEN = 4000
_RECENT_WINDOW = 6  # last ~3 exchanges (user+assistant pairs)


def _get_user_id() -> str:
    """Resolve user ID from MEM0_USER_ID env var, defaulting to ``'user'``."""
    return os.environ.get("MEM0_USER_ID", "user")


def _get_memory():
    """Lazy-initialize and cache a mem0 Memory instance with graph disabled.

    Graph is force-disabled for speed — hooks must complete within the
    Claude Code timeout (15s for context, 30s for stop).  The instance
    is cached in a module global; since each hook invocation is a
    separate process, this only initializes once.
    """
    global _memory
    if _memory is not None:
        return _memory

    # Force graph off — the hard os.environ set overrides any .env value
    # that load_dotenv() loaded at module init.
    os.environ["MEM0_ENABLE_GRAPH"] = "false"

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info, _ = build_config()
    register_providers(providers_info)
    # patch_graph_sanitizer() skipped — graph is force-disabled in hooks,
    # so the relationship sanitizer modules are never invoked.

    from mem0 import Memory

    _memory = Memory.from_config(config_dict)
    return _memory


def _output(data: dict) -> None:
    """Print JSON to stdout (the hook response channel)."""
    print(json.dumps(data))


def _nonfatal() -> dict:
    """Return the standard non-fatal / no-op hook response.

    Must return a **fresh** dict each time — callers may mutate it
    (e.g. adding ``additionalContext``).
    """
    return {"continue": True, "suppressOutput": True}


# ---------------------------------------------------------------------------
# Context Hook  (SessionStart)
# ---------------------------------------------------------------------------


def _extract_results(raw) -> list[dict]:
    """Normalise mem0 search results to a flat list of dicts."""
    if isinstance(raw, dict):
        return raw.get("results", [])
    if isinstance(raw, list):
        return raw
    return []


def context_main() -> None:
    """SessionStart hook: inject cross-session memories as additionalContext."""
    try:
        hook_input = json.loads(sys.stdin.read())
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"
        user_id = _get_user_id()

        mem = _get_memory()

        # --- Multi-query search with deduplication ---
        seen_ids: set[str] = set()
        all_memories: list[dict] = []

        queries = [
            f"project context, architecture, conventions for {project_name}",
            f"recent session summary, decisions, key changes for {project_name}",
        ]

        for query in queries:
            results = _extract_results(
                mem.search(query=query, user_id=user_id, limit=15)
            )
            for r in results:
                mid = r.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_memories.append(r)

        # Cap total injected memories
        all_memories = all_memories[:_MAX_MEMORIES]

        if not all_memories:
            _output(_nonfatal())
            return

        # Format as numbered lines
        lines = ["# mem0 Cross-Session Memory\n"]
        for i, m in enumerate(all_memories, 1):
            text = m.get("memory", m.get("text", ""))
            lines.append(f"{i}. {text}")

        response = _nonfatal()
        response["additionalContext"] = "\n".join(lines)
        _output(response)

    except Exception:
        logger.debug("context_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Stop Hook
# ---------------------------------------------------------------------------


def _extract_content(content) -> str:
    """Extract plain text from a transcript content field.

    Claude Code transcripts use content blocks:
    ``[{"type": "text", "text": "..."}]``
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(parts)
    return ""


def _read_recent_messages(transcript_path: str) -> list[tuple[str, str]]:
    """Read recent user/assistant messages from a JSONL transcript.

    Returns up to ``_RECENT_WINDOW`` ``(role, content)`` tuples in
    chronological order.  Uses a bounded deque so memory stays O(1)
    regardless of transcript length (which can reach ~900 KB).
    Content is truncated during parsing to avoid holding large
    assistant responses (tool results, file reads) in memory.
    """
    messages: deque[tuple[str, str]] = deque(maxlen=_RECENT_WINDOW)

    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Claude Code wraps each transcript entry as
            #   {"type": "user"|"assistant", "message": {"role": ..., "content": ...}, ...}
            # so role/content live under "message", not at the top level.
            # Fall back to the top level for any flat-format transcripts.
            msg = entry["message"] if isinstance(entry.get("message"), dict) else entry
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = _extract_content(msg.get("content", ""))[:_MAX_CONTENT_LEN]
            if content:
                messages.append((role, content))

    return list(messages)


def stop_main() -> None:
    """Stop hook: save session summary to mem0."""
    try:
        hook_input = json.loads(sys.stdin.read())

        # Infinite-loop guard: Claude Code sets this when re-entering
        if hook_input.get("stop_hook_active"):
            _output(_nonfatal())
            return

        session_id = hook_input.get("session_id", "")
        transcript_path = hook_input.get("transcript_path", "")
        cwd = hook_input.get("cwd", "")
        project_name = Path(cwd).name if cwd else "project"
        if not project_name:
            project_name = "project"

        # Missing / invalid transcript
        if not transcript_path or not Path(transcript_path).is_file():
            _output(_nonfatal())
            return

        recent = _read_recent_messages(transcript_path)

        # Skip short sessions — AND means we save when *either* side
        # contributed meaningful content (e.g. short question + long answer).
        user_total = sum(len(c) for r, c in recent if r == "user")
        asst_total = sum(len(c) for r, c in recent if r == "assistant")
        if user_total < _MIN_USER_LEN and asst_total < _MIN_ASSISTANT_LEN:
            _output(_nonfatal())
            return

        # Build summary prompt with recent exchanges
        exchanges = []
        for role, content in recent:
            label = "User" if role == "user" else "Assistant"
            exchanges.append(f"[{label}]: {content}")

        summary = (
            f"Session summary for project '{project_name}':\n\n"
            + "\n\n".join(exchanges)
            + "\n\n"
            "Extract key decisions, solutions found, patterns discovered, "
            "configuration changes, and important context for future sessions."
        )

        mem = _get_memory()
        user_id = _get_user_id()

        mem.add(
            messages=[{"role": "user", "content": summary}],
            user_id=user_id,
            infer=True,
            metadata={
                "source": "session-stop-hook",
                "session_id": session_id,
            },
        )

        _output(_nonfatal())

    except Exception:
        logger.debug("stop_main failed", exc_info=True)
        _output(_nonfatal())


# ---------------------------------------------------------------------------
# Install-Hooks CLI
# ---------------------------------------------------------------------------

_HOOK_CONTEXT_CMD = "mem0-hook-context"
_HOOK_STOP_CMD = "mem0-hook-stop"


def _has_hook(hooks_list: list, command: str) -> bool:
    """Check if a hook with the given command already exists.

    Searches both the current nested format and the legacy flat format::

        Nested:  [{"matcher": "...", "hooks": [{"type": "command", "command": "..."}]}]
        Legacy:  [{"matcher": "...", "command": "..."}]
    """
    for group in hooks_list:
        if not isinstance(group, dict):
            continue
        # Current nested format
        for handler in group.get("hooks") or []:
            if isinstance(handler, dict) and handler.get("command") == command:
                return True
        # Legacy flat format (pre-nested schema)
        if group.get("command") == command:
            return True
    return False


_HANDLER_KEYS = {"command", "timeout"}
_GROUP_KEYS = {"matcher"}


def _migrate_legacy_hooks(hooks_list: list) -> list:
    """Convert legacy flat-format hooks to the nested format.

    Flat entries (``{"command": "...", "timeout": ...}``) are converted to
    nested format (``{"hooks": [{"type": "command", ...}]}``).  Already-nested
    entries are kept as-is.  Non-dict entries are discarded.  Unknown keys are
    forwarded to preserve any extra properties the user may have set.
    """
    migrated = []
    for group in hooks_list:
        if not isinstance(group, dict):
            continue
        if "hooks" in group:
            # Already in nested format
            migrated.append(group)
        elif "command" in group:
            # Legacy flat format — convert, forwarding unknown keys to
            # group level so no user data is silently dropped.
            handler: dict = {"type": "command"}
            new_group: dict = {}
            for k, v in group.items():
                if k in _HANDLER_KEYS:
                    handler[k] = v
                elif k in _GROUP_KEYS:
                    new_group[k] = v
                else:
                    new_group[k] = v
            new_group["hooks"] = [handler]
            migrated.append(new_group)
        else:
            # Unknown format — preserve as-is
            migrated.append(group)
    return migrated


def install_main() -> None:
    """CLI: install mem0 hooks into .claude/settings.json."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mem0-install-hooks",
        description="Install mem0 session hooks for Claude Code",
    )
    parser.add_argument(
        "--global",
        dest="global_install",
        action="store_true",
        help="Install to ~/.claude/settings.json instead of project directory",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Project directory (defaults to CWD)",
    )
    args = parser.parse_args()

    if args.global_install:
        settings_dir = Path.home() / ".claude"
    else:
        project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
        if not project_dir.is_dir():
            print(f"Error: project directory does not exist: {project_dir}", file=sys.stderr)
            sys.exit(1)
        settings_dir = project_dir / ".claude"

    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "settings.json"

    # Read existing settings (preserve everything)
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: {settings_path} contains invalid JSON: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        settings = {}

    if not isinstance(settings.get("hooks"), dict):
        settings["hooks"] = {}

    hooks = settings["hooks"]

    # Migrate any legacy flat-format hooks to nested format
    for event_key in ("SessionStart", "Stop"):
        if isinstance(hooks.get(event_key), list):
            hooks[event_key] = _migrate_legacy_hooks(hooks[event_key])

    installed: list[str] = []
    skipped: list[str] = []

    # --- SessionStart hook ---
    if not isinstance(hooks.get("SessionStart"), list):
        hooks["SessionStart"] = []
    if _has_hook(hooks["SessionStart"], _HOOK_CONTEXT_CMD):
        skipped.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")
    else:
        hooks["SessionStart"].append({
            "matcher": "startup|compact",
            "hooks": [{
                "type": "command",
                "command": _HOOK_CONTEXT_CMD,
                "timeout": 15000,
            }],
        })
        installed.append(f"SessionStart ({_HOOK_CONTEXT_CMD})")

    # --- Stop hook ---
    if not isinstance(hooks.get("Stop"), list):
        hooks["Stop"] = []
    if _has_hook(hooks["Stop"], _HOOK_STOP_CMD):
        skipped.append(f"Stop ({_HOOK_STOP_CMD})")
    else:
        hooks["Stop"].append({
            "hooks": [{
                "type": "command",
                "command": _HOOK_STOP_CMD,
                "timeout": 30000,
            }],
        })
        installed.append(f"Stop ({_HOOK_STOP_CMD})")

    # Atomic write: temp file + rename avoids truncated settings on crash
    fd, tmp_path = tempfile.mkstemp(dir=str(settings_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, str(settings_path))
    except BaseException:
        os.unlink(tmp_path)
        raise

    # Report
    for hook in installed:
        print(f"Installed: {hook}")
    for hook in skipped:
        print(f"Already installed: {hook}")
    print(f"Settings: {settings_path}")
