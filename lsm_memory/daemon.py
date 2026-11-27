"""Zero-dependency loopback HTTP daemon for LSMMemory (1C1G-friendly).

Binds strictly to loopback (default 127.0.0.1:8775), enforces Bearer token auth,
and serves as the single-writer process shared by LSM WebUI, Nested MCP stdio
bridge, CLI tools, and remote GPU/API vector backfill workers.
"""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import time
import urllib.parse
from typing import Any

from . import __version__
from .consolidator import build_consolidation_pack, commit_consolidation_pack
from .embedder import OpenAICompatibleEmbeddingProvider, provider_from_env
from .vector_sync import DirectStoreTransport, sync_missing_vectors
from .governance import (
    GovernanceDeniedError,
    enforce_operation,
    load_governance,
    save_governance,
)
from .policy import ContentBlockedError, assert_content_safe
from .retrieval import AdaptiveRetriever
from .store import ConflictError, MemoryStore, NotFoundError

MAX_BODY_BYTES = 8 * 1024 * 1024  # Allows batch vector uploads up to 8 MB


def _rss_mb() -> float | None:
    """Best-effort current process RSS in MiB (zero external deps)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return round(kb / 1024.0, 2)
    except OSError:
        pass
    try:
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(pmc)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return None


def load_or_create_token(token_file: Path) -> str:
    token_file = token_file.expanduser().resolve()
    token_file.parent.mkdir(parents=True, exist_ok=True)
    if token_file.is_file():
        existing = token_file.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing
    token = secrets.token_urlsafe(36)
    token_file.write_text(token + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(token_file, 0o600)
    return token


def _embed_config_path(db_path: Path) -> Path:
    return db_path.parent / "embed_config.json"


def _default_worker_profile() -> dict[str, str]:
    return {
        "worker": "",
        "llama_server": "",
        "model_path": "",
        "proxy": "",
        "last_fingerprint": "",
        "last_model_name": "",
        "last_model_sha256": "",
    }


def _clean_worker_profile(raw: Any, fallback: dict[str, str] | None = None) -> dict[str, str]:
    base = dict(fallback or _default_worker_profile())
    if not isinstance(raw, dict):
        return base
    for k in ("worker", "llama_server", "model_path", "proxy", "last_fingerprint", "last_model_name", "last_model_sha256"):
        if k in raw and raw[k] is not None:
            base[k] = str(raw[k]).strip()[:512]
    return base


def _infer_tier_mode(cfg: dict[str, Any]) -> str:
    explicit = str(cfg.get("tier_mode") or "").strip().lower()
    if explicit in ("tier0", "tier1a", "tier1b"):
        return explicit
    if not cfg.get("enabled") or not cfg.get("url"):
        return "tier0"
    url = str(cfg.get("url") or "").lower()
    sha = str(cfg.get("model_sha256") or "api").strip().lower()
    if url.startswith("https://") and sha == "api":
        return "tier1a"
    return "tier1b"


def _load_saved_embed_config(db_path: Path) -> dict[str, Any]:
    default_cfg: dict[str, Any] = {
        "enabled": False,
        "tier_mode": "tier0",
        "url": "",
        "model": "",
        "dim": 0,
        "model_sha256": "api",
        "api_key": "",
        "timeout_s": 0.65,
        "worker_profile": _default_worker_profile(),
    }
    p = _embed_config_path(db_path)
    if not p.is_file():
        return default_cfg
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return default_cfg
        cfg = {
            "enabled": bool(raw.get("enabled", bool(raw.get("url")))),
            "tier_mode": str(raw.get("tier_mode") or "").strip().lower(),
            "url": str(raw.get("url") or "").strip(),
            "model": str(raw.get("model") or "").strip(),
            "dim": int(raw.get("dim") or 0),
            "model_sha256": str(raw.get("model_sha256") or "api").strip() or "api",
            "api_key": str(raw.get("api_key") or "").strip(),
            "timeout_s": max(0.1, min(float(raw.get("timeout_s") or 0.65), 10.0)),
            "worker_profile": _clean_worker_profile(raw.get("worker_profile")),
        }
        cfg["tier_mode"] = _infer_tier_mode(cfg)
        return cfg
    except Exception:
        return default_cfg


def _save_embed_config(db_path: Path, cfg: dict[str, Any]) -> None:
    p = _embed_config_path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if os.name != "nt":
        os.chmod(tmp, 0o600)
    tmp.replace(p)
    if os.name != "nt":
        os.chmod(p, 0o600)


def _provider_from_saved_cfg(cfg: dict[str, Any]) -> OpenAICompatibleEmbeddingProvider | None:
    if not cfg.get("enabled") or not cfg.get("url"):
        return None
    return OpenAICompatibleEmbeddingProvider(
        cfg["url"],
        model=cfg.get("model") or "default",
        dim=int(cfg.get("dim") or 0),
        model_sha256=cfg.get("model_sha256") or "api",
        api_key=cfg.get("api_key") or "",
        timeout_s=float(cfg.get("timeout_s") or 0.65),
    )


class MemoryDaemonServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: MemoryStore,
        retriever: AdaptiveRetriever,
        token: str,
        *,
        embed_config: dict[str, Any] | None = None,
    ):
        self.store = store
        self.retriever = retriever
        self.token = token
        self.embed_config: dict[str, Any] = embed_config if embed_config is not None else _load_saved_embed_config(store.db_path)
        super().__init__(address, _Handler)


class _Handler(BaseHTTPRequestHandler):
    server: MemoryDaemonServer
    server_version = f"LSMMemory/{__version__}"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization") or ""
        if not auth.startswith("Bearer "):
            return False
        provided = auth[7:].strip()
        return bool(provided) and secrets.compare_digest(provided, self.server.token)

    def _channel(self) -> str:
        raw = (self.headers.get("X-LSM-Memory-Channel") or "admin").strip().lower()
        return raw if raw in ("admin", "webui", "mcp") else "admin"

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            raise ValueError("Invalid Content-Length") from None
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _public_embed_config(self) -> dict[str, Any]:
        cfg = self.server.embed_config
        prov = self.server.retriever.embedding_provider
        return {
            "enabled": bool(prov is not None),
            "tier_mode": _infer_tier_mode(cfg),
            "url": cfg.get("url") or (prov.endpoint_url if prov else ""),
            "model": cfg.get("model") or (prov.model if prov else ""),
            "dim": int(cfg.get("dim") or (prov._dim if prov else 0)),
            "model_sha256": cfg.get("model_sha256") or (prov.model_sha256 if prov else "api"),
            "timeout_s": float(cfg.get("timeout_s") or 0.65),
            "has_api_key": bool(cfg.get("api_key") or (prov.api_key if prov else "")),
            "active_tier": "tier1_hybrid_rrf" if prov else "tier0_zero_model",
            "active_fingerprint": (prov._fingerprint if prov else ""),
            "worker_profile": _clean_worker_profile(cfg.get("worker_profile")),
        }

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/healthz" and method == "GET":
            self._send_json(200, {"ok": True, "version": __version__, "rss_mb": _rss_mb()})
            return

        if not self._authorized():
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return

        store = self.server.store
        retriever = self.server.retriever
        channel = self._channel()

        try:
            gov = load_governance(store)

            if path == "/v1/status" and method == "GET":
                info = store.meta()
                info["version"] = __version__
                info["rss_mb"] = _rss_mb()
                info["tier_mode"] = "tier1_or_2" if retriever.embedding_provider else "tier0_zero_model"
                info["governance"] = {
                    "auto_inject": gov["auto_inject"],
                    "auto_consolidate_mode": gov["auto_consolidate_mode"],
                    "mcp_default_role": gov["mcp_default_role"],
                    "default_machine_included": gov["default_machine_included"],
                    "excluded_machines": sorted(
                        m for m, r in (gov.get("machines") or {}).items() if not r.get("included", True)
                    ),
                }
                info["embedding"] = self._public_embed_config()
                self._send_json(200, {"ok": True, "data": info})
                return

            if path == "/v1/governance" and method == "GET":
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "data": {
                            "policy": gov,
                            "known_machines": store.list_known_machines(),
                        },
                    },
                )
                return

            if path == "/v1/governance" and method == "POST":
                if channel == "mcp":
                    raise GovernanceDeniedError("admin_only", "Governance policy can only be modified from admin/WebUI.")
                body = self._read_json()
                updated = save_governance(store, body, replace=bool(body.get("replace", False)))
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "data": {
                            "policy": updated,
                            "known_machines": store.list_known_machines(),
                        },
                    },
                )
                return

            if path == "/v1/config/embedding" and method == "GET":
                self._send_json(200, {"ok": True, "data": self._public_embed_config()})
                return

            if path == "/v1/config/embedding" and method == "POST":
                if channel == "mcp":
                    raise GovernanceDeniedError("admin_only", "Embedding config can only be modified from admin/WebUI.")
                body = self._read_json()
                cur = self.server.embed_config
                worker_profile = _clean_worker_profile(body.get("worker_profile"), cur.get("worker_profile"))
                if body.get("worker_profile_only"):
                    new_cfg = dict(cur)
                    new_cfg["worker_profile"] = worker_profile
                    if body.get("tier_mode") in ("tier0", "tier1a", "tier1b"):
                        new_cfg["tier_mode"] = body["tier_mode"]
                    _save_embed_config(store.db_path, new_cfg)
                    self.server.embed_config = new_cfg
                    self._send_json(200, {"ok": True, "data": self._public_embed_config()})
                    return

                enabled = bool(body.get("enabled", bool(str(body.get("url") or "").strip())))
                url = str(body.get("url") or "").strip()
                model = str(body.get("model") or "default").strip() or "default"
                dim = int(body.get("dim") or 0)
                sha = str(body.get("model_sha256") or "api").strip() or "api"
                timeout_s = max(0.1, min(float(body.get("timeout_s") or 0.65), 10.0))
                # Preserve existing api_key unless explicitly replaced or cleared
                if "api_key" in body and body["api_key"] is not None:
                    api_key = str(body["api_key"]).strip()
                elif body.get("clear_api_key"):
                    api_key = ""
                else:
                    api_key = str(cur.get("api_key") or "")

                new_cfg = {
                    "enabled": enabled and bool(url),
                    "tier_mode": str(body.get("tier_mode") or "").strip().lower(),
                    "url": url,
                    "model": model,
                    "dim": dim,
                    "model_sha256": sha,
                    "api_key": api_key,
                    "timeout_s": timeout_s,
                    "worker_profile": worker_profile,
                }
                new_cfg["tier_mode"] = _infer_tier_mode(new_cfg)
                # Validate URL / build provider BEFORE persisting
                new_provider = _provider_from_saved_cfg(new_cfg)
                _save_embed_config(store.db_path, new_cfg)
                self.server.embed_config = new_cfg
                retriever.embedding_provider = new_provider
                self._send_json(200, {"ok": True, "data": self._public_embed_config()})
                return

            if path == "/v1/config/embedding/backfill" and method == "POST":
                if channel == "mcp":
                    raise GovernanceDeniedError("admin_only", "Embedding backfill is admin/WebUI only.")
                body = self._read_json()
                cur = self.server.embed_config
                url = str(body.get("url") or cur.get("url") or "").strip()
                if not url:
                    raise ValueError("Configure and save an Embedding Endpoint URL before running backfill.")
                model = str(body.get("model") or cur.get("model") or "default").strip() or "default"
                sha = str(body.get("model_sha256") or cur.get("model_sha256") or "api").strip() or "api"
                api_key = str(body["api_key"]).strip() if body.get("api_key") else str(cur.get("api_key") or "")
                prov = OpenAICompatibleEmbeddingProvider(
                    url,
                    model=model,
                    model_sha256=sha,
                    api_key=api_key,
                    timeout_s=min(float(body.get("timeout_s") or 20.0), 30.0),
                )
                report = sync_missing_vectors(
                    DirectStoreTransport(store),
                    prov,
                    scope=body.get("scope") or None,
                    batch_size=max(1, min(int(body.get("batch_size") or 32), 64)),
                    max_items=max(1, min(int(body.get("max_items") or 256), 1000)),
                )
                self._send_json(200, {"ok": True, "data": report})
                return

            if path == "/v1/config/embedding/test" and method == "POST":
                if channel == "mcp":
                    raise GovernanceDeniedError("admin_only", "Embedding test is admin/WebUI only.")
                body = self._read_json()
                url = str(body.get("url") or self.server.embed_config.get("url") or "").strip()
                if not url:
                    raise ValueError("url is required to test embedding connectivity")
                model = str(body.get("model") or self.server.embed_config.get("model") or "default").strip() or "default"
                sha = str(body.get("model_sha256") or self.server.embed_config.get("model_sha256") or "api").strip() or "api"
                api_key = str(body["api_key"]).strip() if body.get("api_key") else str(self.server.embed_config.get("api_key") or "")
                t0 = time.perf_counter()
                probe = OpenAICompatibleEmbeddingProvider(
                    url,
                    model=model,
                    model_sha256=sha,
                    api_key=api_key,
                    timeout_s=min(float(body.get("timeout_s") or 10.0), 15.0),
                )
                dim = probe.probe_dimension()
                dt_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "data": {
                            "reachable": True,
                            "dim": dim,
                            "fingerprint": probe.fingerprint,
                            "model": probe.model,
                            "latency_ms": dt_ms,
                        },
                    },
                )
                return

            if path == "/v1/pause" and method == "POST":
                if channel == "mcp":
                    enforce_operation(gov, "suppress", channel="mcp")
                body = self._read_json()
                data = store.set_global_pause(bool(body.get("paused")))
                self._send_json(200, {"ok": True, "data": data})
                return

            if path == "/v1/graph" and method == "GET":
                scope = qs.get("scope", [None])[0]
                if scope == "*":
                    scope = None
                inc_sup = qs.get("include_suppressed", ["1"])[0] in ("1", "true")
                lim = int(qs.get("limit", ["300"])[0])
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "data": store.graph_snapshot(scope=scope, include_suppressed=inc_sup, limit=lim),
                    },
                )
                return

            if path == "/v1/search" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "search", machine=body.get("machine"), channel=channel)
                data = retriever.search(
                    body.get("query", ""),
                    scope=body.get("scope", "personal"),
                    limit=int(body.get("limit", 5)),
                    kind=body.get("kind"),
                    machine=body.get("machine"),
                    source_client=body.get("source_client"),
                    expand_relations=bool(body.get("expand_relations", True)),
                    skip_small_talk=bool(body.get("skip_small_talk", False)),
                    respect_pause=bool(body.get("respect_pause", False)),
                )
                self._send_json(200, {"ok": True, "data": data})
                return

            if path == "/v1/recall-packet" and method == "POST":
                body = self._read_json()
                data = retriever.build_recall_packet(
                    body.get("query", ""),
                    scope=body.get("scope", "personal"),
                    limit=int(body.get("limit", 5)),
                    max_chars=int(body.get("max_chars", 1200)),
                    include_pinned=bool(body.get("include_pinned", True)),
                )
                self._send_json(200, {"ok": True, "data": data})
                return

            if path == "/v1/memories" and method == "GET":
                scope = qs.get("scope", ["personal"])[0]
                if scope == "*":
                    scope = None
                items = store.list_memories(
                    scope=scope,
                    include_suppressed=qs.get("include_suppressed", ["0"])[0] in ("1", "true"),
                    include_forgotten=qs.get("include_forgotten", ["0"])[0] in ("1", "true"),
                    kind=qs.get("kind", [None])[0],
                    machine=qs.get("machine", [None])[0],
                    source_client=qs.get("source_client", [None])[0],
                    limit=int(qs.get("limit", ["100"])[0]),
                )
                self._send_json(200, {"ok": True, "data": {"memories": items, "global_revision": store.global_revision()}})
                return

            if path == "/v1/memories" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "add", machine=body.get("machine"), channel=channel)
                content = body.get("content", "")
                terms = body.get("search_terms")
                if isinstance(content, str):
                    assert_content_safe(content, terms if isinstance(terms, list) else None)
                item = store.add_memory(
                    content,
                    kind=body.get("kind", "episode"),
                    scope=body.get("scope", "personal"),
                    search_terms=terms,
                    topic=body.get("topic", ""),
                    source_client=body.get("source_client", "manual"),
                    machine=body.get("machine", "local"),
                    session_ref=body.get("session_ref", ""),
                    actor=body.get("actor", "agent"),
                    allow_resurrect=bool(body.get("allow_resurrect", False)),
                )
                self._send_json(201, {"ok": True, "data": item})
                return

            if path.startswith("/v1/memories/"):
                parts = path.split("/")
                mem_id = int(parts[3])
                if len(parts) == 4 and method == "GET":
                    item = store.get_memory(mem_id)
                    if item is None:
                        raise NotFoundError(f"Memory #{mem_id} not found")
                    self._send_json(200, {"ok": True, "data": item})
                    return
                if len(parts) == 5 and parts[4] == "revisions" and method == "GET":
                    revs = store.list_revisions(mem_id, limit=int(qs.get("limit", ["30"])[0]))
                    self._send_json(200, {"ok": True, "data": {"revisions": revs}})
                    return
                if len(parts) == 5 and parts[4] == "correct" and method == "POST":
                    body = self._read_json()
                    enforce_operation(gov, "correct", machine=body.get("machine"), channel=channel)
                    content = body.get("content", "")
                    terms = body.get("search_terms")
                    if isinstance(content, str):
                        assert_content_safe(content, terms if isinstance(terms, list) else None)
                    item = store.correct_memory(
                        mem_id,
                        content,
                        int(body.get("expected_version", 0)),
                        search_terms=terms,
                        topic=body.get("topic"),
                        actor=body.get("actor", "agent"),
                    )
                    self._send_json(200, {"ok": True, "data": item})
                    return
                if len(parts) == 5 and parts[4] == "suppress" and method == "POST":
                    body = self._read_json()
                    enforce_operation(gov, "suppress", machine=body.get("machine"), channel=channel)
                    item = store.set_suppressed(mem_id, True, actor=body.get("actor", "agent"))
                    self._send_json(200, {"ok": True, "data": item})
                    return
                if len(parts) == 5 and parts[4] == "restore" and method == "POST":
                    body = self._read_json()
                    enforce_operation(gov, "restore", machine=body.get("machine"), channel=channel)
                    item = store.set_suppressed(mem_id, False, actor=body.get("actor", "agent"))
                    self._send_json(200, {"ok": True, "data": item})
                    return
                if len(parts) == 5 and parts[4] == "forget" and method == "POST":
                    body = self._read_json()
                    enforce_operation(gov, "forget", machine=body.get("machine"), channel=channel)
                    item = store.forget_memory(mem_id, reason=body.get("reason", ""), actor=body.get("actor", "agent"))
                    self._send_json(200, {"ok": True, "data": item})
                    return

            if path == "/v1/relations" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "relate", machine=body.get("machine"), channel=channel)
                rel = store.add_relation(
                    int(body["source_id"]),
                    int(body["target_id"]),
                    body.get("relation_type", "related"),
                    weight=float(body.get("weight", 1.0)),
                    confirmed=bool(body.get("confirmed", True)),
                )
                self._send_json(201, {"ok": True, "data": rel})
                return

            if path == "/v1/relations/delete" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "delete_relation", channel=channel)
                res = store.delete_relation(
                    int(body["source_id"]),
                    int(body["target_id"]),
                    body.get("relation_type"),
                )
                self._send_json(200, {"ok": True, "data": res})
                return

            if path == "/v1/inbox" and method == "GET":
                items = store.list_inbox(
                    status=qs.get("status", ["pending"])[0],
                    scope=qs.get("scope", [None])[0],
                    limit=int(qs.get("limit", ["100"])[0]),
                )
                self._send_json(200, {"ok": True, "data": {"events": items}})
                return

            if path == "/v1/inbox" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "capture", machine=body.get("machine"), channel=channel)
                content = body.get("content", "")
                if isinstance(content, str):
                    assert_content_safe(content)
                ev = store.enqueue_inbox(
                    content,
                    scope=body.get("scope", "personal"),
                    source_client=body.get("source_client", "manual"),
                    machine=body.get("machine", "local"),
                    session_ref=body.get("session_ref", ""),
                    event_key=body.get("event_key"),
                    metadata=body.get("metadata"),
                )
                self._send_json(201, {"ok": True, "data": ev})
                return

            if path == "/v1/inbox/mark" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "consolidate_commit", channel=channel)
                res = store.mark_inbox(body.get("ids", []), body.get("status", "consolidated"))
                self._send_json(200, {"ok": True, "data": res})
                return

            if path == "/v1/vectors/missing" and method == "GET":
                fp = qs.get("fingerprint", [""])[0]
                scope = qs.get("scope", [None])[0]
                lim = int(qs.get("limit", ["256"])[0])
                items = store.list_missing_vectors(fp, scope=scope, limit=lim)
                self._send_json(200, {"ok": True, "data": {"items": items}})
                return

            if path == "/v1/vectors/batch-upsert" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "vectors_upsert", channel=channel)
                res = store.save_vectors_batch(body.get("items", []))
                self._send_json(200, {"ok": True, "data": res})
                return

            if path == "/v1/consolidate/pack" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "consolidate_pull", channel=channel)
                lsm_db = body.get("lsm_db_path") or os.environ.get("LSM_MEMORY_LSM_DB") or None
                pack = build_consolidation_pack(
                    store,
                    scope=body.get("scope", "personal"),
                    inbox_limit=int(body.get("inbox_limit", 30)),
                    lsm_db_path=lsm_db,
                    lsm_limit=int(body.get("lsm_limit", 40)),
                )
                self._send_json(200, {"ok": True, "data": pack})
                return

            if path == "/v1/consolidate/commit" and method == "POST":
                body = self._read_json()
                enforce_operation(gov, "consolidate_commit", channel=channel)
                res = commit_consolidation_pack(
                    store,
                    body,
                    default_scope=body.get("scope", "personal"),
                    actor=body.get("actor", "consolidator"),
                )
                self._send_json(200, {"ok": True, "data": res})
                return

            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
        except GovernanceDeniedError as exc:
            self._send_json(403, {"ok": False, "error": "permission_denied", "reason": exc.reason, "message": str(exc)})
        except ContentBlockedError as exc:
            self._send_json(400, {"ok": False, "error": "content_blocked", "rule": exc.rule_id, "message": str(exc)})
        except NotFoundError as exc:
            self._send_json(404, {"ok": False, "error": "not_found", "message": str(exc)})
        except ConflictError as exc:
            self._send_json(409, {"ok": False, "error": "conflict", "message": str(exc)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"ok": False, "error": "bad_request", "message": str(exc)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": "internal_error", "message": type(exc).__name__})

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSMMemory loopback daemon")
    parser.add_argument("--db", required=True, help="Path to SQLite WAL memory database")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (must be loopback)")
    parser.add_argument("--port", type=int, default=8775, help="Bind port")
    parser.add_argument("--token-file", help="Path to Bearer token file (created if missing)")
    parser.add_argument("--token", help="Explicit Bearer token (or LSM_MEMORY_TOKEN env)")
    parser.add_argument("--max-cached-vectors", type=int, default=256, help="1C1G vector LRU size")
    parser.add_argument("--embed-url", default="", help="Optional OpenAI-compatible /v1/embeddings URL (enables Tier 1/2)")
    parser.add_argument("--embed-model", default="default", help="Embedding model name")
    parser.add_argument("--embed-dim", type=int, default=0, help="Expected embedding dimension")
    parser.add_argument("--embed-model-sha256", default="api", help="Logical model SHA-256 for cross-platform cache")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("Refusing to bind non-loopback host for security; use 127.0.0.1")

    token = args.token or os.environ.get("LSM_MEMORY_TOKEN")
    if not token:
        tf = Path(args.token_file) if args.token_file else Path(args.db).with_suffix(".token")
        token = load_or_create_token(tf)

    store = MemoryStore(args.db, max_cached_vectors=args.max_cached_vectors)
    saved_embed_cfg = _load_saved_embed_config(store.db_path)

    provider = None
    if args.embed_url:
        provider = OpenAICompatibleEmbeddingProvider(
            args.embed_url,
            model=args.embed_model,
            dim=args.embed_dim,
            model_sha256=args.embed_model_sha256,
            api_key=os.environ.get("LSM_MEMORY_EMBED_API_KEY", ""),
            timeout_s=0.65,
        )
    else:
        provider = provider_from_env() or _provider_from_saved_cfg(saved_embed_cfg)

    retriever = AdaptiveRetriever(store, embedding_provider=provider)
    httpd = MemoryDaemonServer((args.host, args.port), store, retriever, token, embed_config=saved_embed_cfg)

    def _shutdown(_sig: int, _frame: Any) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    sys.stderr.write(f"[lsm-memory] listening on http://{args.host}:{args.port} db={store.db_path}\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        retriever.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
