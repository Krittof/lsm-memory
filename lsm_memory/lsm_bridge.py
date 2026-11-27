"""Upstream-compatible WebUI bridge for local-shell-mcp (LSM v6.0.3+).

Provides:
1. Split-injection provider projection (`project_memory_vault_messages`) wired into
   `AgentRunner._project_model_messages`:
   - Leaves `messages[0]` (static system prompt + LibreChat preamble) byte-identical
     to preserve provider prompt caching.
   - Never mutates `state.messages` or persisted SQLite checkpoints.
   - Re-evaluates live vault state on every model turn (including after `AgentSuspended`
     checkpoint resume) and strips any stale `<lsm-memory-recall ...>` blocks so
     suppressed, forgotten, or paused memories never leak after resume.
   - Obeys LSM's per-conversation memory feature toggle, user memory preference,
     agent scope partitioning, and content-protection filters.
2. Dual backend support:
   - Loopback HTTP daemon mode (`LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN` + token)
   - Zero-daemon embedded SQLite mode (`LOCAL_SHELL_MCP_MEMORY_VAULT_DB`) for 1C1G /
     single-process upstream deployments.
3. Optional built-in WebUI vault tools (`memory_vault_search`, `memory_vault_status`,
   `memory_vault_remember`, `memory_vault_suppress`, `memory_vault_forget`,
   `memory_vault_capture`) that coexist cleanly with LibreChat `set_memory`/`delete_memory`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .policy import assert_content_safe
from .retrieval import AdaptiveRetriever, RECALL_NOTICE, is_small_talk
from .store import KINDS, MemoryStore

VAULT_TOOL_NAMES = frozenset({
    "memory_vault_search",
    "memory_vault_status",
    "memory_vault_remember",
    "memory_vault_suppress",
    "memory_vault_forget",
    "memory_vault_capture",
})
VAULT_READONLY_TOOLS = frozenset({"memory_vault_search", "memory_vault_status"})

_RECALL_BLOCK_RE = re.compile(
    r"\n?<lsm-memory-recall\b[^>]*>.*?</lsm-memory-recall>\n*",
    re.DOTALL,
)
_SAFE_SCOPE_RE = re.compile(r"[^A-Za-z0-9._:-]+")
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def _parse_bool_env(name: str, default: bool | None = None) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    low = raw.strip().lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    return default


def _parse_int_env(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(lo, min(int(raw.strip()), hi))
    except ValueError:
        return default


def _is_loopback_origin(origin: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(origin)
    except Exception:
        return False
    return parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")


@dataclass(frozen=True)
class VaultBridgeConfig:
    enabled: bool = False
    origin: str = ""
    token_file: str = ""
    token: str = ""
    db_path: str = ""
    timeout_ms: int = 750
    max_items: int = 5
    max_chars: int = 1200
    expose_tools: bool = True

    @classmethod
    def from_env(cls) -> "VaultBridgeConfig":
        origin = os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN", "").strip().rstrip("/")
        token_file = os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN_FILE", "").strip()
        token = os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN", "").strip()
        db_path = os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_DB", "").strip()
        explicit = _parse_bool_env("LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED", None)
        if explicit is False:
            enabled = False
        elif explicit is True:
            enabled = bool(origin or db_path)
        else:
            enabled = bool(origin or db_path)
        if origin and not _is_loopback_origin(origin):
            # Fail closed on non-loopback daemon URLs.
            origin = ""
            if not db_path:
                enabled = False
        expose_tools = _parse_bool_env("LOCAL_SHELL_MCP_MEMORY_VAULT_TOOLS", True)
        return cls(
            enabled=enabled,
            origin=origin,
            token_file=token_file,
            token=token,
            db_path=db_path,
            timeout_ms=_parse_int_env("LOCAL_SHELL_MCP_MEMORY_VAULT_TIMEOUT_MS", 750, 50, 3000),
            max_items=_parse_int_env("LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_ITEMS", 5, 1, 10),
            max_chars=_parse_int_env("LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_CHARS", 1200, 200, 4000),
            expose_tools=True if expose_tools is None else expose_tools,
        )

    @property
    def mode(self) -> str:
        if not self.enabled:
            return "disabled"
        if self.origin:
            return "daemon"
        if self.db_path:
            return "embedded"
        return "disabled"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect blocked", headers, fp)


class VaultBridgeClient:
    """Thread-safe dual-mode client (loopback HTTP daemon or embedded SQLite)."""

    def __init__(self, config: VaultBridgeConfig):
        self.config = config
        self._lock = threading.Lock()
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        self._token_cache: str = ""
        self._token_mtime: float = -1.0
        self._store: MemoryStore | None = None
        self._retriever: AdaptiveRetriever | None = None

    def _read_token(self) -> str:
        if self.config.token_file:
            p = Path(self.config.token_file)
            try:
                st = p.stat()
                if st.st_mtime != self._token_mtime or not self._token_cache:
                    self._token_cache = p.read_text(encoding="utf-8").strip()
                    self._token_mtime = st.st_mtime
                return self._token_cache
            except OSError:
                return ""
        return self.config.token

    def _embedded(self) -> tuple[MemoryStore, AdaptiveRetriever]:
        with self._lock:
            if self._store is None or self._retriever is None:
                self._store = MemoryStore(self.config.db_path, max_cached_vectors=128)
                self._retriever = AdaptiveRetriever(self._store)
            return self._store, self._retriever

    def _http(self, method: str, path: str, body: dict[str, Any] | None = None, *, timeout_s: float | None = None) -> dict[str, Any]:
        token = self._read_token()
        if not token:
            raise RuntimeError("Missing LSMMemory daemon bearer token")
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.config.origin + path,
            data=data,
            headers=headers,
            method=method,
        )
        t = timeout_s if timeout_s is not None else (self.config.timeout_ms / 1000.0)
        with self._opener.open(req, timeout=t) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise RuntimeError(str((payload or {}).get("message") or (payload or {}).get("error") or "daemon error"))
        return payload["data"]

    def recall_candidates(
        self,
        query: str,
        *,
        scope: str = "personal",
        limit: int | None = None,
    ) -> dict[str, Any]:
        lim = limit if limit is not None else self.config.max_items
        if self.config.mode == "daemon":
            return self._http(
                "POST",
                "/v1/search",
                {
                    "query": query,
                    "scope": scope,
                    "limit": lim,
                    "expand_relations": True,
                    "skip_small_talk": True,
                    "respect_pause": True,
                },
                timeout_s=self.config.timeout_ms / 1000.0,
            )
        if self.config.mode == "embedded":
            _, retriever = self._embedded()
            return retriever.search(
                query,
                scope=scope,
                limit=lim,
                expand_relations=True,
                skip_small_talk=True,
                respect_pause=True,
            )
        return {"results": [], "global_revision": 0, "reason": "disabled"}

    def search(self, query: str, *, scope: str = "personal", limit: int = 5) -> dict[str, Any]:
        if self.config.mode == "daemon":
            return self._http("POST", "/v1/search", {"query": query, "scope": scope, "limit": limit})
        _, retriever = self._embedded()
        return retriever.search(query, scope=scope, limit=limit)

    def status(self) -> dict[str, Any]:
        if self.config.mode == "daemon":
            return self._http("GET", "/v1/status")
        store, _ = self._embedded()
        return store.meta()

    def add_memory(
        self,
        content: str,
        *,
        kind: str = "episode",
        scope: str = "personal",
        search_terms: list[str] | None = None,
        topic: str = "",
        source_client: str = "lsm-webui",
        machine: str = "",
        session_ref: str = "",
    ) -> dict[str, Any]:
        host = machine or socket.gethostname() or "controller"
        assert_content_safe(content, search_terms)
        if self.config.mode == "daemon":
            return self._http(
                "POST",
                "/v1/memories",
                {
                    "content": content,
                    "kind": kind,
                    "scope": scope,
                    "search_terms": search_terms or [],
                    "topic": topic,
                    "source_client": source_client,
                    "machine": host,
                    "session_ref": session_ref,
                },
                timeout_s=5.0,
            )
        store, _ = self._embedded()
        return store.add_memory(
            content,
            kind=kind,
            scope=scope,
            search_terms=search_terms,
            topic=topic,
            source_client=source_client,
            machine=host,
            session_ref=session_ref,
        )

    def suppress(self, memory_id: int, *, action: str) -> dict[str, Any]:
        if action not in ("suppress", "restore"):
            raise ValueError("action must be 'suppress' or 'restore'")
        if self.config.mode == "daemon":
            return self._http("POST", f"/v1/memories/{int(memory_id)}/{action}", {}, timeout_s=5.0)
        store, _ = self._embedded()
        return store.set_suppressed(int(memory_id), action == "suppress", actor="lsm-webui")

    def forget(self, memory_id: int, *, reason: str = "") -> dict[str, Any]:
        if self.config.mode == "daemon":
            return self._http("POST", f"/v1/memories/{int(memory_id)}/forget", {"reason": reason}, timeout_s=5.0)
        store, _ = self._embedded()
        return store.forget_memory(int(memory_id), reason=reason, actor="lsm-webui")

    def capture(
        self,
        content: str,
        *,
        scope: str = "personal",
        source_client: str = "lsm-webui",
        machine: str = "",
        session_ref: str = "",
    ) -> dict[str, Any]:
        host = machine or socket.gethostname() or "controller"
        assert_content_safe(content)
        if self.config.mode == "daemon":
            return self._http(
                "POST",
                "/v1/inbox",
                {
                    "content": content,
                    "scope": scope,
                    "source_client": source_client,
                    "machine": host,
                    "session_ref": session_ref,
                },
                timeout_s=5.0,
            )
        store, _ = self._embedded()
        return store.enqueue_inbox(
            content,
            scope=scope,
            source_client=source_client,
            machine=host,
            session_ref=session_ref,
        )


_CLIENT_LOCK = threading.Lock()
_CACHED_CONFIG: VaultBridgeConfig | None = None
_CACHED_CLIENT: VaultBridgeClient | None = None


def get_bridge_client(config: VaultBridgeConfig | None = None) -> VaultBridgeClient | None:
    cfg = config if config is not None else VaultBridgeConfig.from_env()
    if not cfg.enabled:
        return None
    if config is not None:
        return VaultBridgeClient(cfg)
    global _CACHED_CONFIG, _CACHED_CLIENT
    with _CLIENT_LOCK:
        if _CACHED_CLIENT is None or _CACHED_CONFIG != cfg:
            _CACHED_CONFIG = cfg
            _CACHED_CLIENT = VaultBridgeClient(cfg)
        return _CACHED_CLIENT


def reset_bridge_cache() -> None:
    """Clear process-level cached bridge client (primarily for tests)."""
    global _CACHED_CONFIG, _CACHED_CLIENT
    with _CLIENT_LOCK:
        _CACHED_CONFIG = None
        _CACHED_CLIENT = None


def vault_scope_for_agent(agent: dict[str, Any] | None) -> str:
    """Mirror LSM's agent_partition(agent): personal pool unless agent-scoped."""
    if isinstance(agent, dict) and agent.get("memory_scope") == "agent":
        raw_id = agent.get("id")
        if isinstance(raw_id, str) and raw_id.strip():
            clean = raw_id.split("____", 1)[0].strip()
            clean = _SAFE_SCOPE_RE.sub("_", clean)[:48]
            if clean:
                return f"agent:{clean}"
    return "personal"


def strip_recall_blocks(text: str) -> str:
    if "<lsm-memory-recall" not in text:
        return text
    return _RECALL_BLOCK_RE.sub("", text).lstrip("\n")


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if isinstance(content, str):
        if "<lsm-memory-recall" not in content:
            return message
        copied = dict(message)
        copied["content"] = strip_recall_blocks(content)
        return copied
    if isinstance(content, list):
        changed = False
        new_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                txt = part["text"]
                if "<lsm-memory-recall" in txt:
                    changed = True
                    stripped = strip_recall_blocks(txt)
                    if stripped:
                        p_copy = dict(part)
                        p_copy["text"] = stripped
                        new_parts.append(p_copy)
                    continue
            new_parts.append(part)
        if not changed:
            return message
        copied = dict(message)
        copied["content"] = new_parts
        return copied
    return message


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str)
        ]
        return "\n".join(x for x in parts if x.strip()).strip()
    return ""


def _lsm_content_blocked(value: str, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return False
    try:
        from local_shell_mcp.webui.memory_protection import inspect_memory_content
    except Exception:
        return False
    try:
        return inspect_memory_content({"value": value}, filters) is not None
    except Exception:
        # Fail closed when an active filter throws.
        return True


def _render_recall_block(
    items: list[dict[str, Any]],
    *,
    global_revision: int,
    scope: str,
    max_chars: int,
) -> str:
    if not items:
        return ""
    header = (
        f'<lsm-memory-recall scope="{scope}" revision="{global_revision}" untrusted="true">\n'
        f"{RECALL_NOTICE}\n"
    )
    footer = "\n</lsm-memory-recall>"
    budget = max(120, max_chars - len(header) - len(footer))
    lines: list[str] = []
    used = 0
    for item in items:
        mid = item.get("id", "?")
        kind = item.get("kind", "episode")
        ver = item.get("version", 1)
        src = item.get("source_client") or item.get("machine") or "vault"
        text = str(item.get("content") or "").strip().replace("\r\n", " ").replace("\n", " ")
        if not text:
            continue
        if len(text) > 320:
            text = text[:317] + "..."
        line = f"- [#{mid} v{ver} {kind} @{src}] {text}"
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return header + "\n".join(lines) + footer


def project_memory_vault_messages(
    messages: list[dict[str, Any]],
    *,
    agent: dict[str, Any] | None = None,
    namespace: str = "",
    memory_feature_enabled: bool = True,
    filters: dict[str, Any] | None = None,
    client: VaultBridgeClient | None = None,
    content_blocker: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Project live LSMMemory recall into ephemeral provider messages.

    Safety invariants:
    - Always returns a list without mutating caller's `messages` or inner dicts.
    - Strips any stale `<lsm-memory-recall ...>` blocks (including on resume or disable).
    - Never modifies `messages[0]` (preserves static system prompt / Prompt Cache prefix).
    - Attaches ephemeral recall adjacent to the latest user turn (provider-universal).
    - Fails closed (returns cleaned messages) on any timeout, pause, or backend error.
    """
    if not messages:
        return messages

    # 1. Always strip stale ephemeral recall blocks (handles resume after suppress/disable).
    cleaned = [_clean_message(m) for m in messages if not m.get("_lsm_memory_recall")]

    if not memory_feature_enabled:
        return cleaned
    if "/subagent:" in (namespace or ""):
        return cleaned

    bridge = client if client is not None else get_bridge_client()
    if bridge is None:
        return cleaned

    # 2. Locate the latest user message (never touch system prefix at index 0).
    last_user_idx = -1
    query_text = ""
    for idx in range(len(cleaned) - 1, -1, -1):
        if cleaned[idx].get("role") == "user":
            txt = _extract_text(cleaned[idx].get("content"))
            if txt:
                last_user_idx = idx
                query_text = txt
                break
    if last_user_idx < 0 or not query_text or is_small_talk(query_text):
        return cleaned

    scope = vault_scope_for_agent(agent)
    try:
        packet = bridge.recall_candidates(query_text, scope=scope)
    except Exception:
        return cleaned

    if not isinstance(packet, dict):
        return cleaned
    raw_items = packet.get("results") or []
    if not isinstance(raw_items, list) or not raw_items:
        return cleaned

    blocker = content_blocker if content_blocker is not None else (lambda val: _lsm_content_blocked(val, filters))
    safe_items = [
        item for item in raw_items
        if isinstance(item, dict) and isinstance(item.get("content"), str) and not blocker(item["content"])
    ]
    if not safe_items:
        return cleaned

    block = _render_recall_block(
        safe_items,
        global_revision=int(packet.get("global_revision") or 0),
        scope=scope,
        max_chars=bridge.config.max_chars,
    )
    if not block:
        return cleaned

    # 3. Attach to a shallow copy of the latest user turn only (prefix 0..last_user_idx-1 byte-identical!).
    result = list(cleaned)
    target = dict(result[last_user_idx])
    orig_content = target.get("content")
    if isinstance(orig_content, str):
        target["content"] = block + "\n\n" + orig_content
    elif isinstance(orig_content, list):
        target["content"] = [{"type": "text", "text": block}, *orig_content]
    else:
        return cleaned
    result[last_user_idx] = target
    return result


def vault_tool_definitions(config: VaultBridgeConfig | None = None) -> list[dict[str, Any]]:
    """OpenAI-compatible function schemas for LSM WebUI built-in memory vault tools."""
    cfg = config if config is not None else VaultBridgeConfig.from_env()
    if not cfg.enabled or not cfg.expose_tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "memory_vault_search",
                "description": (
                    "Search the cross-network persistent memory vault (LSMMemory) for relevant facts, "
                    "decisions, preferences, and operational rules across devices."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query or keyword"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_vault_remember",
                "description": (
                    "Store a durable, self-contained memory into the cross-network vault with optional "
                    "multilingual aliases (search_terms) for strong recall even on 1C1G hosts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Self-contained factual statement (no secrets)."},
                        "kind": {"type": "string", "enum": list(KINDS), "default": "episode"},
                        "topic": {"type": "string", "description": "Short topic label"},
                        "search_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Short aliases or bilingual keywords for fast retrieval.",
                        },
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_vault_suppress",
                "description": "Soft-suppress (hide from recall without deleting) or restore a memory by numeric ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "minimum": 1},
                        "action": {"type": "string", "enum": ["suppress", "restore"]},
                    },
                    "required": ["memory_id", "action"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_vault_forget",
                "description": "Permanently forget a memory by numeric ID with a tombstone that prevents resurrection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "minimum": 1},
                        "reason": {"type": "string", "default": ""},
                    },
                    "required": ["memory_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_vault_capture",
                "description": "Capture raw conversation notes into the offline consolidation inbox for later refinement.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Raw note to queue for offline consolidation."},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "memory_vault_status",
                "description": "Inspect cross-network memory vault status, revision counter, counts, and tier mode.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def execute_vault_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    agent: dict[str, Any] | None = None,
    filters: dict[str, Any] | None = None,
    session_ref: str = "",
    client: VaultBridgeClient | None = None,
) -> str:
    """Execute one built-in WebUI memory vault tool and return compact JSON text."""
    bridge = client if client is not None else get_bridge_client()
    if bridge is None:
        return json.dumps({"ok": False, "error": "memory_vault_disabled"})
    args = arguments if isinstance(arguments, dict) else {}
    scope = vault_scope_for_agent(agent)
    try:
        if name == "memory_vault_search":
            res = bridge.search(
                str(args.get("query") or ""),
                scope=scope,
                limit=max(1, min(int(args.get("limit", 5)), 10)),
            )
            if filters and isinstance(res.get("results"), list):
                res["results"] = [
                    r for r in res["results"]
                    if not _lsm_content_blocked(str(r.get("content") or ""), filters)
                ]
            return json.dumps(res, ensure_ascii=False)
        if name == "memory_vault_status":
            return json.dumps(bridge.status(), ensure_ascii=False)
        if name == "memory_vault_remember":
            content = str(args.get("content") or "")
            if _lsm_content_blocked(content, filters):
                return json.dumps({"ok": False, "error": "content_filter_block"})
            terms = args.get("search_terms")
            if not isinstance(terms, list):
                terms = []
            res = bridge.add_memory(
                content,
                kind=str(args.get("kind") or "episode"),
                scope=scope,
                search_terms=[str(t) for t in terms],
                topic=str(args.get("topic") or ""),
                session_ref=session_ref,
            )
            return json.dumps(res, ensure_ascii=False)
        if name == "memory_vault_suppress":
            res = bridge.suppress(int(args["memory_id"]), action=str(args["action"]))
            return json.dumps(res, ensure_ascii=False)
        if name == "memory_vault_forget":
            res = bridge.forget(int(args["memory_id"]), reason=str(args.get("reason") or ""))
            return json.dumps(res, ensure_ascii=False)
        if name == "memory_vault_capture":
            content = str(args.get("content") or "")
            if _lsm_content_blocked(content, filters):
                return json.dumps({"ok": False, "error": "content_filter_block"})
            res = bridge.capture(content, scope=scope, session_ref=session_ref)
            return json.dumps(res, ensure_ascii=False)
        return json.dumps({"ok": False, "error": f"unknown_tool:{name}"})
    except Exception as exc:
        return json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False)
