"""Zero-dependency LSM-styled WebUI management console & Starlette routes for LSMMemory.

Mounts cleanly inside LSM's Starlette application at:
  - GET  /memories/vault              (LSM-styled HTML/CSS/JS SPA, zero external CDN)
  - GET  /memories/vault-nav.js       (Top-right launcher pill synced to LSM language & theme)
  - GET  /memory-vault                (Compatibility alias)
  - GET/POST /api/memory-vault/*      (Authenticated JSON bridge to Daemon or Embedded SQLite)

Authentication & security:
  - Reuses LSM's native WebUI HttpOnly session cookie (`webui/auth.py`) and/or OAuth principal.
  - Unauthenticated HTML requests redirect to `/auth/login`; unauthenticated API calls return 401.
  - Mutating POST endpoints require `X-Requested-With: LSMMemory` (or OAuth Bearer token) for CSRF defense.
  - Daemon Bearer token and Tier 1A Embedding API keys are never exposed to the browser.
  - Tier 1B Worker backfill dispatches via LSM's authenticated `remote_manager` without sending any
    controller tokens, Admin PINs, or SSH keys to the Worker.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
from typing import Any

from . import __version__
from .daemon import (
    _clean_worker_profile,
    _infer_tier_mode,
    _load_saved_embed_config,
    _provider_from_saved_cfg,
    _rss_mb,
    _save_embed_config,
)
from .embedder import ManagedLlamaEmbedServer, OpenAICompatibleEmbeddingProvider
from .governance import load_governance, save_governance
from .lsm_bridge import VaultBridgeClient, VaultBridgeConfig, get_bridge_client
from .retrieval import logical_embedding_fingerprint
from .vector_sync import (
    DirectStoreTransport,
    build_worker_embed_script,
    parse_worker_embed_output,
    sync_missing_vectors,
)


def _require_lsm_owner(request: Any, *, is_html: bool = False) -> tuple[str | None, Any | None]:
    """Authenticate against LSM's native principal / WebUI cookie session."""
    from starlette.exceptions import HTTPException
    from starlette.responses import RedirectResponse

    principal = getattr(request.state, "principal", None)
    subject = getattr(principal, "subject", None)
    if isinstance(subject, str) and subject:
        return subject, None

    try:
        from local_shell_mcp.auth import verify_request
        from local_shell_mcp.settings import get_settings
        from local_shell_mcp.webui.auth import COOKIE, WebAuthStore

        settings = get_settings()
        if settings.auth_mode == "none":
            return "local", None
        cookie = request.cookies.get(COOKIE, "")
        if cookie:
            session = WebAuthStore(settings.state_dir).session(cookie)
            if session is not None and session.owner:
                return session.owner, None
        if (request.headers.get("authorization") or "").lower().startswith("bearer "):
            p = verify_request(request)
            if getattr(p, "subject", None):
                return str(p.subject), None
    except Exception:
        pass

    if is_html:
        return None, RedirectResponse("/auth/login", status_code=302)
    raise HTTPException(401, detail="WebUI authentication required")


def _check_csrf(request: Any) -> None:
    from starlette.exceptions import HTTPException

    if request.method in ("GET", "HEAD", "OPTIONS"):
        return
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return
    if (request.headers.get("x-requested-with") or "").strip() != "LSMMemory":
        raise HTTPException(403, detail="Missing X-Requested-With header")


def _live_lsm_workers() -> list[dict[str, Any]]:
    """Best-effort discovery of connected LSM remote workers for Governance & Tier 1B UI."""
    try:
        from local_shell_mcp.remote import remote_manager

        mgr = remote_manager()
        snapshot = mgr.list_machines() if hasattr(mgr, "list_machines") else {}
        machines = snapshot.get("machines", []) if isinstance(snapshot, dict) else (snapshot or [])
        out = []
        for item in machines:
            if isinstance(item, dict) and item.get("name"):
                out.append({
                    "machine": str(item["name"]),
                    "online": item.get("status") == "online",
                    "platform": str(item.get("platform") or item.get("os") or ""),
                    "source": "lsm_worker",
                })
        return out
    except Exception:
        return []


def _client_call(
    client: VaultBridgeClient,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    qs: dict[str, str] | None = None,
    *,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    """Dispatch management calls uniformly to Daemon HTTP or Embedded SQLite."""
    if client.config.mode == "daemon":
        import urllib.parse

        full_path = path
        if qs:
            clean_qs = {k: v for k, v in qs.items() if v is not None and v != ""}
            if clean_qs:
                full_path += "?" + urllib.parse.urlencode(clean_qs)
        return client._http(method, full_path, body, timeout_s=timeout_s)

    # Embedded single-process fallback
    store, retriever = client._embedded()
    payload = body or {}
    q = qs or {}

    if path == "/v1/status" and method == "GET":
        gov = load_governance(store)
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
        info["embedding"] = _client_call(client, "GET", "/v1/config/embedding")
        return info
    if path == "/v1/memories" and method == "GET":
        scope = q.get("scope")
        if scope in (None, "", "*"):
            scope = None
        return {
            "memories": store.list_memories(
                scope=scope,
                include_suppressed=q.get("include_suppressed") in ("1", "true"),
                include_forgotten=q.get("include_forgotten") in ("1", "true"),
                machine=q.get("machine") or None,
                limit=int(q.get("limit") or 200),
            ),
            "global_revision": store.global_revision(),
        }
    if path == "/v1/memories" and method == "POST":
        return client.add_memory(
            str(payload.get("content") or ""),
            kind=str(payload.get("kind") or "episode"),
            scope=str(payload.get("scope") or "personal"),
            search_terms=payload.get("search_terms") if isinstance(payload.get("search_terms"), list) else [],
            topic=str(payload.get("topic") or ""),
            source_client=str(payload.get("source_client") or "webui-console"),
            machine=str(payload.get("machine") or "controller"),
        )
    if path == "/v1/search" and method == "POST":
        return retriever.search(
            str(payload.get("query") or ""),
            scope=payload.get("scope") or None,
            limit=int(payload.get("limit") or 10),
            machine=payload.get("machine") or None,
        )
    if path == "/v1/graph" and method == "GET":
        scope = q.get("scope")
        if scope in (None, "", "*"):
            scope = None
        return store.graph_snapshot(
            scope=scope,
            include_suppressed=q.get("include_suppressed", "1") in ("1", "true"),
            limit=int(q.get("limit") or 300),
        )
    if path == "/v1/relations" and method == "POST":
        return store.add_relation(
            int(payload["source_id"]),
            int(payload["target_id"]),
            str(payload.get("relation_type") or "related"),
            weight=float(payload.get("weight", 1.0)),
        )
    if path == "/v1/relations/delete" and method == "POST":
        return store.delete_relation(
            int(payload["source_id"]),
            int(payload["target_id"]),
            payload.get("relation_type"),
        )
    if path == "/v1/inbox" and method == "GET":
        return {"events": store.list_inbox(status=q.get("status") or "pending", limit=int(q.get("limit") or 100))}
    if path == "/v1/inbox" and method == "POST":
        return client.capture(
            str(payload.get("content") or ""),
            scope=str(payload.get("scope") or "personal"),
            source_client=str(payload.get("source_client") or "webui-console"),
            machine=str(payload.get("machine") or "controller"),
        )
    if path == "/v1/inbox/mark" and method == "POST":
        return store.mark_inbox(payload.get("ids") or [], str(payload.get("status") or "consolidated"))
    if path == "/v1/pause" and method == "POST":
        return store.set_global_pause(bool(payload.get("paused")))
    if path == "/v1/governance" and method == "GET":
        return {"policy": load_governance(store), "known_machines": store.list_known_machines()}
    if path == "/v1/governance" and method == "POST":
        return {
            "policy": save_governance(store, payload, replace=bool(payload.get("replace", False))),
            "known_machines": store.list_known_machines(),
        }
    if path == "/v1/vectors/missing" and method == "GET":
        scope = q.get("scope")
        if scope in (None, "", "*"):
            scope = None
        return {
            "items": store.list_missing_vectors(
                str(q.get("fingerprint") or ""),
                scope=scope,
                limit=int(q.get("limit") or 256),
            )
        }
    if path == "/v1/vectors/batch-upsert" and method == "POST":
        return store.save_vectors_batch(payload.get("items") or [])
    if path == "/v1/config/embedding" and method == "GET":
        cfg = _load_saved_embed_config(store.db_path)
        prov = retriever.embedding_provider
        return {
            "enabled": bool(prov is not None),
            "tier_mode": _infer_tier_mode(cfg),
            "url": cfg.get("url") or "",
            "model": cfg.get("model") or "",
            "dim": int(cfg.get("dim") or 0),
            "model_sha256": cfg.get("model_sha256") or "api",
            "timeout_s": float(cfg.get("timeout_s") or 0.65),
            "has_api_key": bool(cfg.get("api_key")),
            "active_tier": "tier1_hybrid_rrf" if prov else "tier0_zero_model",
            "active_fingerprint": prov._fingerprint if prov else "",
            "worker_profile": _clean_worker_profile(cfg.get("worker_profile")),
        }
    if path == "/v1/config/embedding" and method == "POST":
        cur = _load_saved_embed_config(store.db_path)
        wp = _clean_worker_profile(payload.get("worker_profile"), cur.get("worker_profile"))
        if payload.get("worker_profile_only"):
            new_cfg = dict(cur)
            new_cfg["worker_profile"] = wp
            if payload.get("tier_mode") in ("tier0", "tier1a", "tier1b"):
                new_cfg["tier_mode"] = payload["tier_mode"]
            _save_embed_config(store.db_path, new_cfg)
            return _client_call(client, "GET", "/v1/config/embedding")
        url = str(payload.get("url") or "").strip()
        api_key = (
            str(payload["api_key"]).strip()
            if payload.get("api_key") is not None
            else ("" if payload.get("clear_api_key") else cur.get("api_key", ""))
        )
        new_cfg = {
            "enabled": bool(payload.get("enabled", bool(url))) and bool(url),
            "tier_mode": str(payload.get("tier_mode") or "").strip().lower(),
            "url": url,
            "model": str(payload.get("model") or "default").strip() or "default",
            "dim": int(payload.get("dim") or 0),
            "model_sha256": str(payload.get("model_sha256") or "api").strip() or "api",
            "api_key": api_key,
            "timeout_s": max(0.1, min(float(payload.get("timeout_s") or 0.65), 10.0)),
            "worker_profile": wp,
        }
        new_cfg["tier_mode"] = _infer_tier_mode(new_cfg)
        prov = _provider_from_saved_cfg(new_cfg)
        _save_embed_config(store.db_path, new_cfg)
        retriever.embedding_provider = prov
        return _client_call(client, "GET", "/v1/config/embedding")
    if path == "/v1/config/embedding/test" and method == "POST":
        cur = _load_saved_embed_config(store.db_path)
        url = str(payload.get("url") or cur.get("url") or "").strip()
        if not url:
            raise ValueError("url is required")
        t0 = time.perf_counter()
        probe = OpenAICompatibleEmbeddingProvider(
            url,
            model=str(payload.get("model") or cur.get("model") or "default"),
            model_sha256=str(payload.get("model_sha256") or cur.get("model_sha256") or "api"),
            api_key=str(payload["api_key"]).strip() if payload.get("api_key") else str(cur.get("api_key") or ""),
            timeout_s=10.0,
        )
        dim = probe.probe_dimension()
        return {
            "reachable": True,
            "dim": dim,
            "fingerprint": probe.fingerprint,
            "model": probe.model,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
    if path == "/v1/config/embedding/backfill" and method == "POST":
        cur = _load_saved_embed_config(store.db_path)
        url = str(payload.get("url") or cur.get("url") or "").strip()
        if not url:
            raise ValueError("Configure and save an Embedding Endpoint URL first.")
        prov = OpenAICompatibleEmbeddingProvider(
            url,
            model=str(payload.get("model") or cur.get("model") or "default"),
            model_sha256=str(payload.get("model_sha256") or cur.get("model_sha256") or "api"),
            api_key=str(payload["api_key"]).strip() if payload.get("api_key") else str(cur.get("api_key") or ""),
            timeout_s=20.0,
        )
        return sync_missing_vectors(
            DirectStoreTransport(store),
            prov,
            scope=payload.get("scope") or None,
            batch_size=max(1, min(int(payload.get("batch_size") or 32), 64)),
            max_items=max(1, min(int(payload.get("max_items") or 256), 1000)),
        )
    if path.startswith("/v1/memories/"):
        parts = path.split("/")
        mid = int(parts[3])
        action = parts[4]
        if action == "revisions":
            return {"revisions": store.list_revisions(mid)}
        if action == "correct":
            return store.correct_memory(
                mid,
                str(payload.get("content") or ""),
                int(payload.get("expected_version") or 0),
                search_terms=payload.get("search_terms"),
                topic=payload.get("topic"),
                actor="webui-console",
            )
        if action == "suppress":
            return store.set_suppressed(mid, True, actor="webui-console")
        if action == "restore":
            return store.set_suppressed(mid, False, actor="webui-console")
        if action == "forget":
            return store.forget_memory(mid, reason=str(payload.get("reason") or ""), actor="webui-console")
    raise ValueError(f"Unsupported path: {method} {path}")


def _extract_remote_stdout(res: Any) -> str:
    """Extract stdout string from LSM RemoteManager.call('run_python') response shapes."""
    if isinstance(res, str):
        return res
    if not isinstance(res, dict):
        return str(res)
    if res.get("ok") is False:
        err = res.get("stderr") or res.get("error") or res.get("message") or str(res)
        raise RuntimeError(f"Remote worker execution failed: {str(err)[-400:]}")
    data = res.get("data") if isinstance(res.get("data"), dict) else res
    if isinstance(data, dict):
        rc = data.get("returncode")
        if rc is None:
            rc = data.get("exit_code")
        stdout = str(data.get("stdout") or data.get("output") or "")
        stderr = str(data.get("stderr") or "")
        if rc not in (None, 0):
            raise RuntimeError(f"Worker script exited ({rc}): {(stderr or stdout)[-400:]}")
        if stdout:
            return stdout
        if isinstance(data.get("result"), dict):
            return _extract_remote_stdout(data["result"])
    return json.dumps(res, ensure_ascii=False)


def render_console_html() -> str:
    """Zero-dependency LSM-styled management console + lightweight 2D Canvas relation graph."""
    return _CONSOLE_HTML.replace("__VERSION__", __version__)


_NAV_LAUNCHER_JS = r"""(function(){
  if (document.getElementById('lsm-memory-vault-launcher') || location.pathname.startsWith('/memories/vault')) return;
  function isZh(){
    var pref = (localStorage.getItem('lsm_memory_lang') || localStorage.getItem('lang') || navigator.language || 'en').toLowerCase();
    return pref.indexOf('zh') === 0;
  }
  function updateLabel(a){
    var zh = isZh();
    a.title = zh ? '打开 LSM 记忆库管理台' : 'Open LSM Memory Vault';
    a.innerHTML = '<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#10a37f;margin-right:7px;"></span>' + (zh ? '记忆库' : 'Memory Vault');
  }
  function mount(){
    if (document.getElementById('lsm-memory-vault-launcher') || !document.body) return;
    var a = document.createElement('a');
    a.id = 'lsm-memory-vault-launcher';
    a.href = '/memories/vault';
    updateLabel(a);
    var isDark = document.documentElement.classList.contains('dark') || (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
    Object.assign(a.style, {
      position: 'fixed', top: '12px', right: '20px', zIndex: '9999',
      display: 'inline-flex', alignItems: 'center', height: '32px', padding: '0 12px',
      borderRadius: '10px', fontSize: '13px', fontWeight: '500', textDecoration: 'none',
      fontFamily: '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
      background: isDark ? '#212121' : '#ffffff',
      color: isDark ? '#ececec' : '#111827',
      border: '1px solid ' + (isDark ? '#3a3a3a' : '#e5e7eb'),
      boxShadow: '0 1px 2px rgba(0,0,0,0.06)', opacity: '0.96'
    });
    a.onmouseenter = function(){ a.style.opacity = '1'; a.style.borderColor = '#10a37f'; };
    a.onmouseleave = function(){ a.style.opacity = '0.96'; a.style.borderColor = isDark ? '#3a3a3a' : '#e5e7eb'; };
    document.body.appendChild(a);
    window.addEventListener('storage', function(ev){
      if (!ev || ev.key === 'lang' || ev.key === 'lsm_memory_lang') updateLabel(a);
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount);
  else mount();
})();
"""


def inject_vault_nav_into_index_html(html: str) -> str:
    """Inject lightweight Memory Vault launcher script into LSM WebUI index.html when vault is enabled."""
    if not VaultBridgeConfig.from_env().enabled:
        return html
    tag = '<script src="/memories/vault-nav.js" defer></script>'
    if tag in html:
        return html
    if "</body>" in html:
        return html.replace("</body>", tag + "\n</body>", 1)
    return html + "\n" + tag


def memory_vault_web_routes(
    *,
    client_factory: Any = None,
    auth_override: Any = None,
) -> list[Any]:
    """Return Starlette Routes for `/memories/vault` and `/api/memory-vault/*`."""
    from starlette.exceptions import HTTPException
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    def _get_client() -> VaultBridgeClient:
        c = client_factory() if client_factory is not None else get_bridge_client()
        if c is None:
            raise HTTPException(503, detail="LSMMemory vault is not enabled (set LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN or _DB)")
        return c

    def _auth(request: Request, *, is_html: bool = False):
        if auth_override is not None:
            return auth_override(request, is_html=is_html)
        return _require_lsm_owner(request, is_html=is_html)

    async def _json_body(request: Request) -> dict[str, Any]:
        raw = await request.body()
        if not raw:
            return {}
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(400, detail="Invalid JSON body") from exc
        if not isinstance(obj, dict):
            raise HTTPException(400, detail="JSON body must be an object")
        return obj

    def _run(fn):
        try:
            return JSONResponse({"ok": True, "data": fn()})
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status_code=400)

    async def page(request: Request):
        owner, redir = _auth(request, is_html=True)
        if redir is not None:
            return redir
        return HTMLResponse(render_console_html(), headers={"Cache-Control": "no-store"})

    async def nav_js(request: Request):
        from starlette.responses import Response

        return Response(
            _NAV_LAUNCHER_JS,
            media_type="application/javascript; charset=utf-8",
            headers={"Cache-Control": "no-cache"},
        )

    async def api_status(request: Request):
        _auth(request)
        return _run(lambda: _client_call(_get_client(), "GET", "/v1/status"))

    async def api_memories(request: Request):
        _auth(request)
        if request.method == "GET":
            qs = dict(request.query_params)
            return _run(lambda: _client_call(_get_client(), "GET", "/v1/memories", qs=qs))
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/memories", body=body))

    async def api_search(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/search", body=body))

    async def api_memory_action(request: Request):
        _auth(request)
        mid = int(request.path_params["memory_id"])
        action = request.path_params["action"]
        if action == "revisions" and request.method == "GET":
            return _run(lambda: _client_call(_get_client(), "GET", f"/v1/memories/{mid}/revisions"))
        if action not in ("correct", "suppress", "restore", "forget"):
            raise HTTPException(404, detail="Unknown memory action")
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", f"/v1/memories/{mid}/{action}", body=body))

    async def api_graph(request: Request):
        _auth(request)
        qs = dict(request.query_params)
        return _run(lambda: _client_call(_get_client(), "GET", "/v1/graph", qs=qs))

    async def api_relations(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/relations", body=body))

    async def api_relations_delete(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/relations/delete", body=body))

    async def api_inbox(request: Request):
        _auth(request)
        if request.method == "GET":
            qs = dict(request.query_params)
            return _run(lambda: _client_call(_get_client(), "GET", "/v1/inbox", qs=qs))
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/inbox", body=body))

    async def api_inbox_mark(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/inbox/mark", body=body))

    async def api_pause(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/pause", body=body))

    async def api_governance(request: Request):
        _auth(request)
        if request.method == "GET":
            def _fetch():
                data = _client_call(_get_client(), "GET", "/v1/governance")
                data["live_workers"] = _live_lsm_workers()
                return data
            return _run(_fetch)
        _check_csrf(request)
        body = await _json_body(request)
        def _save():
            data = _client_call(_get_client(), "POST", "/v1/governance", body=body)
            data["live_workers"] = _live_lsm_workers()
            return data
        return _run(_save)

    async def api_embed_config(request: Request):
        _auth(request)
        if request.method == "GET":
            return _run(lambda: _client_call(_get_client(), "GET", "/v1/config/embedding"))
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/config/embedding", body=body))

    async def api_embed_test(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        return _run(lambda: _client_call(_get_client(), "POST", "/v1/config/embedding/test", body=body))

    async def api_embed_backfill(request: Request):
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        try:
            data = await asyncio.to_thread(
                lambda: _client_call(_get_client(), "POST", "/v1/config/embedding/backfill", body=body, timeout_s=120.0)
            )
            return JSONResponse({"ok": True, "data": data})
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status_code=400)

    async def api_worker_backfill(request: Request):
        """1-click Tier 1B offline vector backfill on a chosen online LSM Worker (or controller)."""
        _auth(request)
        _check_csrf(request)
        body = await _json_body(request)
        try:
            worker = str(body.get("worker") or "").strip()
            llama_server = str(body.get("llama_server") or "").strip()
            model_path = str(body.get("model_path") or "").strip()
            max_items = max(1, min(int(body.get("max_items") or 200), 500))
            if not worker:
                raise ValueError("worker is required")
            if not llama_server or not model_path:
                raise ValueError("llama_server and model_path are required")

            client = _get_client()
            # Persist non-secret worker profile immediately
            cur_cfg = await asyncio.to_thread(lambda: _client_call(client, "GET", "/v1/config/embedding"))
            wp = dict(cur_cfg.get("worker_profile") or {})
            wp.update({"worker": worker, "llama_server": llama_server, "model_path": model_path})
            await asyncio.to_thread(
                lambda: _client_call(
                    client,
                    "POST",
                    "/v1/config/embedding",
                    body={"worker_profile_only": True, "worker_profile": wp},
                )
            )

            t0 = time.perf_counter()
            if worker == "controller":
                def _run_local() -> dict[str, Any]:
                    with ManagedLlamaEmbedServer(llama_server, model_path) as prov:
                        missing = _client_call(
                            client,
                            "GET",
                            "/v1/vectors/missing",
                            qs={"fingerprint": prov.fingerprint, "limit": str(max_items)},
                        )["items"]
                        if not missing:
                            return {
                                "worker": "controller",
                                "model_name": prov.model,
                                "model_sha256": prov.model_sha256,
                                "fingerprint": prov.fingerprint,
                                "dim": prov.dim,
                                "fetched": 0,
                                "accepted": 0,
                                "skipped_stale": 0,
                                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                            }
                        accepted = 0
                        skipped = 0
                        for i in range(0, len(missing), 32):
                            chunk = missing[i : i + 32]
                            from .vector_sync import format_memory_for_embedding

                            vecs = prov.embed_batch([format_memory_for_embedding(it) for it in chunk], is_query=False)
                            up = [
                                {
                                    "memory_id": int(it["memory_id"]),
                                    "content_version": int(it["content_version"]),
                                    "fingerprint": prov.fingerprint,
                                    "vector": [round(float(x), 6) for x in v],
                                }
                                for it, v in zip(chunk, vecs, strict=True)
                            ]
                            r = _client_call(client, "POST", "/v1/vectors/batch-upsert", body={"items": up}, timeout_s=45.0)
                            accepted += int(r.get("accepted", 0))
                            skipped += int(r.get("skipped_stale", 0))
                        return {
                            "worker": "controller",
                            "model_name": prov.model,
                            "model_sha256": prov.model_sha256,
                            "fingerprint": prov.fingerprint,
                            "dim": prov.dim,
                            "fetched": len(missing),
                            "accepted": accepted,
                            "skipped_stale": skipped,
                            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                        }

                report = await asyncio.to_thread(_run_local)
            else:
                from local_shell_mcp.remote import remote_manager

                mgr = remote_manager()
                last_fp = str(wp.get("last_fingerprint") or "")
                missing_res = await asyncio.to_thread(
                    lambda: _client_call(
                        client,
                        "GET",
                        "/v1/vectors/missing",
                        qs={"fingerprint": last_fp, "limit": str(max_items)},
                    )
                )
                missing = missing_res.get("items") or []
                if not missing and last_fp:
                    return JSONResponse({
                        "ok": True,
                        "data": {
                            "worker": worker,
                            "fingerprint": last_fp,
                            "fetched": 0,
                            "accepted": 0,
                            "skipped_stale": 0,
                            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                        },
                    })
                script = build_worker_embed_script(llama_server, model_path, missing, batch_size=32)
                raw_res = await mgr.call(worker, "run_python", {"code": script, "timeout_s": 180}, timeout_s=210)
                stdout_text = _extract_remote_stdout(raw_res)
                parsed = parse_worker_embed_output(stdout_text)
                dim = int(parsed["dim"])
                fp = logical_embedding_fingerprint(
                    model_name=str(parsed["model_name"]),
                    model_sha256=str(parsed["model_sha256"]),
                    dim=dim,
                    pooling=str(parsed.get("pooling") or "last"),
                )
                vecs = parsed.get("vectors") or []
                accepted = 0
                skipped = 0
                for i in range(0, len(vecs), 32):
                    chunk = vecs[i : i + 32]
                    up = []
                    for entry in chunk:
                        raw_bytes = base64.b64decode(entry["vec_b64"])
                        floats = list(struct.unpack(f"<{dim}f", raw_bytes))
                        up.append({
                            "memory_id": int(entry["memory_id"]),
                            "content_version": int(entry["content_version"]),
                            "fingerprint": fp,
                            "vector": [round(float(x), 6) for x in floats],
                        })
                    r = await asyncio.to_thread(
                        lambda items=up: _client_call(client, "POST", "/v1/vectors/batch-upsert", body={"items": items}, timeout_s=45.0)
                    )
                    accepted += int(r.get("accepted", 0))
                    skipped += int(r.get("skipped_stale", 0))
                report = {
                    "worker": worker,
                    "model_name": str(parsed["model_name"]),
                    "model_sha256": str(parsed["model_sha256"]),
                    "fingerprint": fp,
                    "dim": dim,
                    "fetched": len(missing),
                    "accepted": accepted,
                    "skipped_stale": skipped,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                }

            # Save discovered fingerprint + model metadata so future backfills are incremental
            # and the Query-Time fingerprint dropdown auto-fills model & SHA-256 across reloads.
            if report.get("fingerprint"):
                wp["last_fingerprint"] = str(report["fingerprint"])
                if report.get("model_name"):
                    wp["last_model_name"] = str(report["model_name"])
                if report.get("model_sha256"):
                    wp["last_model_sha256"] = str(report["model_sha256"])
                await asyncio.to_thread(
                    lambda: _client_call(
                        client,
                        "POST",
                        "/v1/config/embedding",
                        body={"worker_profile_only": True, "worker_profile": wp},
                    )
                )
            return JSONResponse({"ok": True, "data": report})
        except HTTPException:
            raise
        except Exception as exc:
            return JSONResponse({"ok": False, "error": type(exc).__name__, "message": str(exc)}, status_code=400)

    async def api_legacy_import(request: Request):
        owner, _ = _auth(request)
        if request.method == "GET":
            def _count():
                try:
                    from local_shell_mcp.webui.memory import MemoryRepository
                    from local_shell_mcp.webui.routes import _repo

                    items = MemoryRepository(_repo().database).list(owner or "local")
                    return {"available": True, "count": len(items)}
                except Exception:
                    return {"available": False, "count": 0}
            return _run(_count)
        _check_csrf(request)
        def _do_import():
            from local_shell_mcp.webui.memory import MemoryRepository
            from local_shell_mcp.webui.routes import _repo

            items = MemoryRepository(_repo().database).list(owner or "local")
            c = _get_client()
            imported = 0
            skipped = 0
            for entry in items:
                scope = f"agent:{entry['agentId']}" if entry.get("agentId") else "personal"
                try:
                    res = _client_call(
                        c,
                        "POST",
                        "/v1/memories",
                        body={
                            "content": entry["value"],
                            "kind": "user_profile",
                            "scope": scope,
                            "topic": entry.get("key") or "legacy",
                            "search_terms": [entry["key"]] if entry.get("key") else [],
                            "source_client": "legacy-import",
                            "machine": "controller",
                        },
                    )
                    if res.get("deduplicated"):
                        skipped += 1
                    else:
                        imported += 1
                except Exception:
                    skipped += 1
            return {"total_legacy": len(items), "imported": imported, "skipped": skipped}
        return _run(_do_import)

    return [
        Route("/memories/vault", page, methods=["GET"]),
        Route("/memories/vault-nav.js", nav_js, methods=["GET"]),
        Route("/memory-vault", page, methods=["GET"]),
        Route("/api/memory-vault/status", api_status, methods=["GET"]),
        Route("/api/memory-vault/memories", api_memories, methods=["GET", "POST"]),
        Route("/api/memory-vault/search", api_search, methods=["POST"]),
        Route("/api/memory-vault/memories/{memory_id:int}/{action:str}", api_memory_action, methods=["GET", "POST"]),
        Route("/api/memory-vault/graph", api_graph, methods=["GET"]),
        Route("/api/memory-vault/relations", api_relations, methods=["POST"]),
        Route("/api/memory-vault/relations/delete", api_relations_delete, methods=["POST"]),
        Route("/api/memory-vault/inbox", api_inbox, methods=["GET", "POST"]),
        Route("/api/memory-vault/inbox/mark", api_inbox_mark, methods=["POST"]),
        Route("/api/memory-vault/pause", api_pause, methods=["POST"]),
        Route("/api/memory-vault/governance", api_governance, methods=["GET", "POST"]),
        Route("/api/memory-vault/config/embedding", api_embed_config, methods=["GET", "POST"]),
        Route("/api/memory-vault/config/embedding/test", api_embed_test, methods=["POST"]),
        Route("/api/memory-vault/config/embedding/backfill", api_embed_backfill, methods=["POST"]),
        Route("/api/memory-vault/worker-backfill", api_worker_backfill, methods=["POST"]),
        Route("/api/memory-vault/legacy", api_legacy_import, methods=["GET", "POST"]),
    ]


_CONSOLE_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>LSM Memory Vault</title>
<style>
:root {
  --bg: #ffffff; --surface: #f7f7f8; --surface-2: #ececf1; --border: #e2e2e8;
  --text: #111827; --muted: #6b7280; --accent: #10a37f; --accent-hover: #0d8a6a;
  --accent-soft: rgba(16,163,127,0.10);
  --danger: #dc2626; --warn: #d97706; --info: #2563eb; --radius: 8px;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #171717; --surface: #212121; --surface-2: #2f2f2f; --border: #383838;
    --text: #ececec; --muted: #9b9b9b; --accent: #10a37f; --accent-hover: #14b88f;
    --accent-soft: rgba(16,163,127,0.16);
    --danger: #ef4444; --warn: #f59e0b; --info: #60a5fa;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
         gap: 10px; padding: 12px 20px; border-bottom: 1px solid var(--border); background: var(--surface); }
.brand { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-weight: 600; font-size: 15px; }
.badge { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: var(--surface-2); color: var(--muted); border: 1px solid var(--border); }
.badge.ok { color: var(--accent); border-color: var(--accent); }
.badge.warn { color: var(--warn); border-color: var(--warn); }
.badge.danger { color: var(--danger); border-color: var(--danger); }
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
nav.tabs { display: flex; gap: 4px; padding: 0 20px; border-bottom: 1px solid var(--border); background: var(--surface); overflow-x: auto; }
nav.tabs button { background: none; border: none; border-bottom: 2px solid transparent; padding: 10px 14px;
                  color: var(--muted); cursor: pointer; font-size: 13px; font-weight: 500; white-space: nowrap; }
nav.tabs button.active { color: var(--text); border-bottom-color: var(--accent); }
main { max-width: 1240px; margin: 0 auto; padding: 18px 20px 48px; }
.panel { display: none; }
.panel.active { display: block; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px 16px; margin-bottom: 14px; }
.subcard { background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; margin-top: 12px; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.grow { flex: 1 1 220px; }
input, select, textarea { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 7px 10px; font-size: 13px; font-family: inherit; }
textarea { width: 100%; min-height: 76px; resize: vertical; }
button.btn { background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; font-size: 13px; cursor: pointer; font-weight: 500; }
button.btn:hover { border-color: var(--muted); }
button.btn:disabled { opacity: 0.55; cursor: not-allowed; }
button.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
button.btn.primary:hover { background: var(--accent-hover); }
button.btn.danger { color: var(--danger); border-color: var(--danger); background: transparent; }
button.btn.sm { padding: 3px 8px; font-size: 12px; }
a.link { color: var(--muted); text-decoration: none; font-size: 13px; }
a.link:hover { color: var(--text); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
.mono { font-family: var(--mono); font-size: 12px; }
.muted { color: var(--muted); font-size: 12px; }
.tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.tag { font-size: 11px; padding: 1px 6px; border-radius: 4px; background: var(--surface-2); color: var(--muted); }
.suppressed-row { opacity: 0.58; }
.graph-wrap { display: grid; grid-template-columns: 1fr 340px; gap: 14px; }
@media (max-width: 900px) { .graph-wrap { grid-template-columns: 1fr; } }
#graphCanvas { width: 100%; height: 520px; background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius); display: block; cursor: grab; }
#graphCanvas:active { cursor: grabbing; }
.toast { position: fixed; right: 18px; bottom: 18px; padding: 10px 14px; border-radius: var(--radius);
         background: var(--surface-2); border: 1px solid var(--border); box-shadow: 0 4px 16px rgba(0,0,0,.25);
         font-size: 13px; z-index: 120; display: none; max-width: 460px; }
dialog { background: var(--surface); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius);
         max-width: 580px; width: 92vw; padding: 18px; }
dialog::backdrop { background: rgba(0,0,0,.5); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.grid3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }
label.field { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: var(--muted); }
label.check { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text); cursor: pointer; }
.field-head { display: inline-flex; align-items: center; gap: 5px; }
/* Inline ? help icon & floating popover */
.help-tip { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px;
  border-radius: 50%; border: 1px solid var(--border); background: var(--surface-2); color: var(--muted);
  font-size: 11px; font-weight: 700; line-height: 1; cursor: pointer; padding: 0; flex: 0 0 auto;
  vertical-align: middle; transition: border-color .15s, color .15s, background .15s; }
.help-tip:hover, .help-tip[aria-expanded="true"] { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }
#helpPopover { position: fixed; z-index: 200; max-width: 340px; min-width: 220px; padding: 11px 13px;
  border-radius: 8px; background: var(--surface); color: var(--text); border: 1px solid var(--border);
  box-shadow: 0 10px 28px rgba(0,0,0,.22); font-size: 12.5px; line-height: 1.55; display: none; }
#helpPopover .hp-title { font-weight: 600; margin-bottom: 4px; font-size: 13px; display: flex; justify-content: space-between; gap: 8px; }
#helpPopover .hp-close { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 14px; padding: 0 2px; line-height: 1; }
#helpPopover .hp-close:hover { color: var(--text); }
/* Tier selector cards */
.tier-cards { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }
@media (max-width: 780px) { .tier-cards { grid-template-columns: 1fr; } }
.tier-card { border: 1px solid var(--border); border-radius: var(--radius); padding: 12px; background: var(--bg);
  cursor: pointer; transition: border-color .15s, background .15s; display: flex; flex-direction: column; gap: 4px; }
.tier-card:hover { border-color: var(--muted); }
.tier-card.selected { border-color: var(--accent); background: var(--accent-soft); box-shadow: inset 0 0 0 1px var(--accent); }
.tier-card .tc-head { display: flex; align-items: center; justify-content: space-between; gap: 6px; font-weight: 600; font-size: 13.5px; }
.tier-card .tc-sub { font-size: 12px; color: var(--muted); }
.tier-pane { display: none; }
.tier-pane.active { display: block; }
.radio-card { display: flex; align-items: flex-start; gap: 8px; padding: 10px 12px; border: 1px solid var(--border);
  border-radius: 6px; background: var(--surface); cursor: pointer; }
.radio-card.selected { border-color: var(--accent); background: var(--accent-soft); }
pre.cmd-box { margin: 8px 0 0; padding: 10px 12px; border-radius: 6px; background: var(--surface); border: 1px solid var(--border);
  font-family: var(--mono); font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-all; }
details.fold { margin-top: 10px; }
details.fold summary { cursor: pointer; font-size: 12.5px; color: var(--muted); user-select: none; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span data-i18n="app_title">LSM Memory Vault</span>
    <span class="badge">v__VERSION__</span>
    <span id="tierBadge" class="badge">...</span>
    <span id="rssBadge" class="badge">RAM --</span>
    <span id="countBadge" class="badge">0 active</span>
  </div>
  <div class="toolbar">
    <span class="field-head">
      <button id="pauseBtn" class="btn" onclick="togglePause()">Global Pause</button>
      <button type="button" class="help-tip" data-help="global_pause">?</button>
    </span>
    <button id="langBtn" class="btn sm" onclick="toggleLang()">中文 / EN</button>
    <a class="link" href="/" data-i18n="back_to_lsm">← 返回 LSM WebUI</a>
  </div>
</header>

<nav class="tabs">
  <button class="active" data-tab="memories" onclick="switchTab('memories')" data-i18n="tab_memories">记忆管理</button>
  <button data-tab="graph" onclick="switchTab('graph')" data-i18n="tab_graph">关系星图 (2D)</button>
  <button data-tab="inbox" onclick="switchTab('inbox')" data-i18n="tab_inbox">收件箱与整理</button>
  <button data-tab="gov" onclick="switchTab('gov')" data-i18n="tab_gov">机器权限与多档配置</button>
</nav>

<main>
  <!-- TAB 1: MEMORIES -->
  <section id="tab-memories" class="panel active">
    <div class="card">
      <div class="row">
        <input id="searchInput" class="grow" data-i18n-placeholder="search_ph" placeholder="搜索记忆（支持中英文别名、端口、路径）..." onkeydown="if(event.key==='Enter')runSearch()" />
        <button type="button" class="help-tip" data-help="search_engine">?</button>
        <select id="filterScope">
          <option value="*" data-i18n="opt_all_scopes">全部作用域</option>
          <option value="personal" selected>personal</option>
        </select>
        <select id="filterMachine"><option value="" data-i18n="opt_all_machines">全部机器</option></select>
        <span class="field-head">
          <label class="check"><input type="checkbox" id="chkSuppressed" checked onchange="loadMemories()" /> <span data-i18n="show_suppressed">含已挂起</span></label>
          <label class="check"><input type="checkbox" id="chkForgotten" onchange="loadMemories()" /> <span data-i18n="show_forgotten">含已遗忘</span></label>
          <button type="button" class="help-tip" data-help="suppress_vs_forget">?</button>
        </span>
        <button class="btn primary" onclick="runSearch()" data-i18n="btn_search">检索</button>
        <button class="btn" onclick="clearSearch()" data-i18n="btn_reset">全部</button>
        <button class="btn primary" onclick="openAddDialog()" data-i18n="btn_add">+ 新建记忆</button>
        <button id="legacyImportBtn" class="btn" style="display:none" onclick="importLegacy()"></button>
      </div>
    </div>
    <div class="card" style="padding:0; overflow-x:auto;">
      <table>
        <thead>
          <tr>
            <th style="width:60px">ID</th>
            <th data-i18n="col_content">内容与搜索别名</th>
            <th style="width:150px" data-i18n="col_meta">来源 / 机器</th>
            <th style="width:100px" data-i18n="col_status">状态</th>
            <th style="width:240px" data-i18n="col_actions">操作</th>
          </tr>
        </thead>
        <tbody id="memTbody"><tr><td colspan="5" class="muted">Loading...</td></tr></tbody>
      </table>
    </div>
  </section>

  <!-- TAB 2: 2D RELATION GRAPH -->
  <section id="tab-graph" class="panel">
    <div class="card row" style="justify-content:space-between;">
      <div class="row">
        <span class="muted" data-i18n="graph_hint">拖拽节点可调整位置，滚轮缩放，点击节点高亮 1-hop 关联邻居并在右侧操作。</span>
        <button type="button" class="help-tip" data-help="relation_graph">?</button>
      </div>
      <div class="row">
        <button class="btn sm" onclick="loadGraph()" data-i18n="btn_refresh">刷新图谱</button>
        <button class="btn sm" onclick="resetGraphView()" data-i18n="btn_reset_view">重置视图</button>
      </div>
    </div>
    <div class="graph-wrap">
      <canvas id="graphCanvas"></canvas>
      <div class="card" id="graphInspector">
        <div class="muted" data-i18n="graph_select_prompt">点击左侧任意节点查看详情、编辑、挂起或建立关系边。</div>
      </div>
    </div>
  </section>

  <!-- TAB 3: INBOX & CONSOLIDATION -->
  <section id="tab-inbox" class="panel">
    <div class="card">
      <div class="row" style="justify-content:space-between; margin-bottom:10px;">
        <span class="field-head">
          <strong data-i18n="inbox_title">离线整理收件箱（Staging Inbox）</strong>
          <button type="button" class="help-tip" data-help="staging_inbox">?</button>
        </span>
        <div class="row">
          <select id="inboxStatus" onchange="loadInbox()">
            <option value="pending" selected data-i18n="opt_pending">pending (待整理)</option>
            <option value="consolidated" data-i18n="opt_consolidated">consolidated (已整理)</option>
            <option value="skipped" data-i18n="opt_skipped">skipped (已跳过)</option>
          </select>
          <span class="field-head">
            <button class="btn" onclick="copyConsolidatePrompt()" data-i18n="btn_copy_prompt">复制 Agent 整理指令</button>
            <button type="button" class="help-tip" data-help="consolidate_prompt">?</button>
          </span>
          <button class="btn primary" onclick="openCaptureDialog()" data-i18n="btn_capture">+ 写入草稿</button>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th style="width:60px">ID</th>
              <th data-i18n="col_draft">草稿内容</th>
              <th style="width:150px" data-i18n="col_meta">来源 / 机器</th>
              <th style="width:170px" data-i18n="col_actions">操作</th>
            </tr>
          </thead>
          <tbody id="inboxTbody"><tr><td colspan="4" class="muted">Loading...</td></tr></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- TAB 4: GOVERNANCE & MULTI-TIER RETRIEVAL -->
  <section id="tab-gov" class="panel">
    <div class="card">
      <div class="field-head" style="margin-bottom:12px;">
        <h3 style="margin:0; font-size:15px;" data-i18n="gov_global_title">自动化与权限总控</h3>
        <button type="button" class="help-tip" data-help="gov_global">?</button>
      </div>
      <div class="grid2">
        <div class="field-head">
          <label class="check"><input type="checkbox" id="govAutoInject" /> <span data-i18n="gov_auto_inject">开启 WebUI 自动分体注入</span></label>
          <button type="button" class="help-tip" data-help="auto_inject">?</button>
        </div>
        <div class="field-head">
          <label class="check"><input type="checkbox" id="govDefaultIncluded" /> <span data-i18n="gov_default_inc">新发现机器默认纳入</span></label>
          <button type="button" class="help-tip" data-help="default_included">?</button>
        </div>
        <label class="field">
          <span class="field-head"><span data-i18n="gov_mcp_role">外部 /mcp 默认权限档位</span><button type="button" class="help-tip" data-help="mcp_role">?</button></span>
          <select id="govMcpRole">
            <option value="read_write" data-i18n="role_rw_full">read_write（完全读写管理）</option>
            <option value="read_capture" data-i18n="role_rc_full">read_capture（只读 + 仅可丢收件箱草稿）</option>
            <option value="read_only" data-i18n="role_ro_full">read_only（纯只读，防误删与提示词注入）</option>
          </select>
        </label>
        <label class="field">
          <span class="field-head"><span data-i18n="gov_consolidate_mode">离线整理模式</span><button type="button" class="help-tip" data-help="consolidate_mode">?</button></span>
          <select id="govConsolidateMode">
            <option value="manual" data-i18n="mode_manual">manual（手动 / Agent 触发）</option>
            <option value="auto" data-i18n="mode_auto">auto（定时全自动整理）</option>
          </select>
        </label>
        <label class="field">
          <span class="field-head"><span data-i18n="gov_interval">自动整理间隔（分钟，5–1440）</span><button type="button" class="help-tip" data-help="consolidate_interval">?</button></span>
          <input id="govInterval" type="number" min="5" max="1440" value="60" />
        </label>
      </div>
      <div class="row" style="margin-top:12px;">
        <button class="btn primary" onclick="saveGlobalGov()" data-i18n="btn_save_gov">保存总控策略</button>
      </div>
    </div>

    <div class="card">
      <div class="row" style="justify-content:space-between; margin-bottom:10px;">
        <span class="field-head">
          <strong data-i18n="gov_machines_title">按机器选入 / 剔除与权限治理</strong>
          <button type="button" class="help-tip" data-help="machine_gov">?</button>
        </span>
        <div class="row">
          <input id="newMachineName" data-i18n-placeholder="machine_ph" placeholder="机器名，如 gpu-worker / node-01" style="width:220px;" />
          <button class="btn sm" onclick="addMachineRow()" data-i18n="btn_add_machine">+ 添加机器</button>
        </div>
      </div>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th data-i18n="col_machine">机器名</th>
              <th data-i18n="col_mem_inbox">记忆数 / 待整理</th>
              <th><span class="field-head"><span data-i18n="col_include">选入 / 剔除</span><button type="button" class="help-tip" data-help="machine_include">?</button></span></th>
              <th><span class="field-head"><span data-i18n="col_role">权限档位</span><button type="button" class="help-tip" data-help="machine_role">?</button></span></th>
              <th><span class="field-head"><span data-i18n="col_auto_cons">自动整理</span><button type="button" class="help-tip" data-help="machine_auto">?</button></span></th>
              <th style="width:90px" data-i18n="col_save">保存</th>
            </tr>
          </thead>
          <tbody id="machineTbody"><tr><td colspan="6" class="muted">Loading...</td></tr></tbody>
        </table>
      </div>
    </div>

    <!-- MULTI-TIER RETRIEVAL ENGINE (3-MODE CARD SWITCHER) -->
    <div class="card">
      <div class="row" style="justify-content:space-between; margin-bottom:10px;">
        <span class="field-head">
          <h3 style="margin:0; font-size:15px;" data-i18n="embed_title">多档检索引擎配置（Tier 0 / Tier 1A / Tier 1B 热切换）</h3>
          <button type="button" class="help-tip" data-help="tier_overview">?</button>
        </span>
        <span id="embStatusText" class="badge">...</span>
      </div>

      <div class="tier-cards">
        <div id="card-tier0" class="tier-card" onclick="selectTierCard('tier0')">
          <div class="tc-head">
            <span class="field-head"><span data-i18n="tier0_name">Tier 0 · 零模型默认</span><button type="button" class="help-tip" data-help="tier0_card" onclick="event.stopPropagation()">?</button></span>
            <span id="badge-tier0" class="badge">~14 MB</span>
          </div>
          <div class="tc-sub" data-i18n="tier0_short">词法 BM25 + 3× 双语别名 + MaxSim + 1-hop 关系图谱，无需外部 API 或 GPU。</div>
        </div>
        <div id="card-tier1a" class="tier-card" onclick="selectTierCard('tier1a')">
          <div class="tc-head">
            <span class="field-head"><span data-i18n="tier1a_name">Tier 1A · 云端向量 API</span><button type="button" class="help-tip" data-help="tier1a_card" onclick="event.stopPropagation()">?</button></span>
            <span id="badge-tier1a" class="badge">~14 MB</span>
          </div>
          <div class="tc-sub" data-i18n="tier1a_short">接入任意 OpenAI 兼容 /v1/embeddings API，主控保持 1C1G 轻量并支持一键云端补齐向量。</div>
        </div>
        <div id="card-tier1b" class="tier-card" onclick="selectTierCard('tier1b')">
          <div class="tc-head">
            <span class="field-head"><span data-i18n="tier1b_name">Tier 1B · Worker / 本地模型</span><button type="button" class="help-tip" data-help="tier1b_card" onclick="event.stopPropagation()">?</button></span>
            <span id="badge-tier1b" class="badge">Local GGUF</span>
          </div>
          <div class="tc-sub" data-i18n="tier1b_short">调用在线 Worker 的本地 GPU/CPU llama-server 批量算向量（空闲 0 显存），可选主控本地检索。</div>
        </div>
      </div>

      <!-- PANE: TIER 0 -->
      <div id="pane-tier0" class="tier-pane subcard">
        <div class="row" style="justify-content:space-between;">
          <div>
            <strong data-i18n="tier0_pane_title">零模型模式（推荐 1C1G 主控）</strong>
            <div class="muted" style="margin-top:4px;" data-i18n="tier0_pane_desc">无需填写任何 API Key 或本地模型路径。所有向量检索超时或未启用时也会自动安全降级到此档位。</div>
          </div>
          <button class="btn primary" onclick="activateTier0()" data-i18n="btn_use_tier0">启用 Tier 0 零模型</button>
        </div>
      </div>

      <!-- PANE: TIER 1A (CLOUD API) -->
      <div id="pane-tier1a" class="tier-pane subcard">
        <div class="field-head" style="margin-bottom:10px;">
          <strong data-i18n="tier1a_pane_title">云端 OpenAI 兼容 /v1/embeddings 配置</strong>
          <button type="button" class="help-tip" data-help="tier1a_fields">?</button>
        </div>
        <div class="grid3">
          <label class="field">
            <span>Embedding Endpoint URL</span>
            <input id="embUrl" placeholder="https://api.siliconflow.cn/v1/embeddings" />
          </label>
          <label class="field">
            <span>Model Name</span>
            <input id="embModel" placeholder="BAAI/bge-m3" />
          </label>
          <label class="field">
            <span data-i18n="lbl_api_key">API Key（留空表示保留已存密钥）</span>
            <input id="embKey" type="password" placeholder="sk-..." autocomplete="off" />
          </label>
        </div>
        <div class="row" style="margin-top:12px;">
          <button class="btn" onclick="testTier1A()" data-i18n="btn_test_conn">测试连通与维度</button>
          <button class="btn primary" onclick="saveTier1A()" data-i18n="btn_save_apply">保存并立即热生效</button>
          <span class="field-head">
            <button id="btnCloudBackfill" class="btn" onclick="runCloudBackfill()" data-i18n="btn_cloud_backfill">一键补齐缺失向量</button>
            <button type="button" class="help-tip" data-help="cloud_backfill">?</button>
          </span>
        </div>
      </div>

      <!-- PANE: TIER 1B (WORKER / LOCAL MODEL) -->
      <div id="pane-tier1b" class="tier-pane">
        <!-- Box 1: Offline Worker Batch Backfill -->
        <div class="subcard">
          <div class="field-head" style="margin-bottom:8px;">
            <strong data-i18n="t1b_worker_title">① 选一台在线 Worker 执行离线批量算向量（按需拉起 llama-server，用完即退、空闲 0 显存）</strong>
            <button type="button" class="help-tip" data-help="t1b_worker_box">?</button>
          </div>
          <div class="grid3">
            <label class="field">
              <span class="field-head"><span data-i18n="t1b_select_worker">选择执行计算的机器（Worker）</span><button type="button" class="help-tip" data-help="t1b_worker_pick">?</button></span>
              <select id="wWorker" onchange="updateWorkerCliPreview()"></select>
            </label>
            <label class="field">
              <span data-i18n="t1b_llama_path">该机器上的 llama-server 路径</span>
              <input id="wLlamaPath" placeholder="D:\tools\llama-server.exe or /usr/local/bin/llama-server" oninput="updateWorkerCliPreview()" />
            </label>
            <label class="field">
              <span data-i18n="t1b_gguf_path">该机器上的 .gguf 模型路径</span>
              <input id="wModelPath" placeholder="D:\models\Qwen3-Embedding-0.6B-Q8_0.gguf" oninput="updateWorkerCliPreview()" />
            </label>
          </div>
          <div class="row" style="margin-top:12px;">
            <button id="btnWorkerRun" class="btn primary" onclick="runWorkerBackfill()" data-i18n="btn_run_worker">一键在所选 Worker 补齐向量</button>
            <button class="btn" onclick="saveWorkerProfileOnly()" data-i18n="btn_save_worker_profile">保存 Worker 路径配置</button>
          </div>
          <details class="fold">
            <summary data-i18n="t1b_cli_summary">查看等效命令行（适合超大语料或离线终端手动执行）</summary>
            <pre id="workerCliBox" class="cmd-box"></pre>
            <div class="row" style="margin-top:6px;">
              <button class="btn sm" onclick="copyWorkerCli()" data-i18n="btn_copy_cli">复制命令</button>
            </div>
          </details>
        </div>

        <!-- Box 2: Query-Time Vector Source -->
        <div class="subcard">
          <div class="field-head" style="margin-bottom:8px;">
            <strong data-i18n="t1b_query_title">② 主控提问阶段（Query-Time）向量来源</strong>
            <button type="button" class="help-tip" data-help="t1b_query_box">?</button>
          </div>
          <div class="grid2">
            <label id="qmode-offline-card" class="radio-card" onclick="selectQueryMode('offline')">
              <input type="radio" name="t1bQMode" id="qmodeOffline" value="offline" checked />
              <div>
                <div class="field-head"><strong style="font-size:13px;" data-i18n="t1b_qmode_offline">仅保留离线向量库（主控维持 Tier 0 · ~14 MB）</strong><button type="button" class="help-tip" data-help="t1b_qmode_offline_help" onclick="event.stopPropagation()">?</button></div>
                <div class="muted" data-i18n="t1b_qmode_offline_desc">适合 1C1G 低配主控：Worker 算好的向量已存入库中，日常提问仍走零模型极速召回。</div>
              </div>
            </label>
            <label id="qmode-local-card" class="radio-card" onclick="selectQueryMode('local')">
              <input type="radio" name="t1bQMode" id="qmodeLocal" value="local" />
              <div>
                <div class="field-head"><strong style="font-size:13px;" data-i18n="t1b_qmode_local">主控本机 llama-server 混合检索（4G+ 主机）</strong><button type="button" class="help-tip" data-help="t1b_qmode_local_help" onclick="event.stopPropagation()">?</button></div>
                <div class="muted" data-i18n="t1b_qmode_local_desc">主控本机常驻同款 GGUF 环回服务，每次提问实时走混合 RRF 语义召回。</div>
              </div>
            </label>
          </div>
          <div id="t1bLocalFields" style="display:none; margin-top:12px;">
            <div class="grid3">
              <label class="field">
                <span data-i18n="t1b_local_url">主控环回 Endpoint URL</span>
                <input id="embLocalUrl" value="http://127.0.0.1:8081/v1/embeddings" placeholder="http://127.0.0.1:8081/v1/embeddings" />
              </label>
              <label class="field">
                <span data-i18n="t1b_local_model">模型名称（与 GGUF 文件名一致）</span>
                <input id="embLocalModel" placeholder="Qwen3-Embedding-0.6B-Q8_0.gguf" />
              </label>
              <label class="field">
                <span class="field-head"><span data-i18n="t1b_fp_select">匹配已有向量缓存（自动识别指纹）</span><button type="button" class="help-tip" data-help="t1b_fp_help">?</button></span>
                <select id="embFpSelect" onchange="onFpSelectChange()"></select>
              </label>
            </div>
            <div id="customShaWrap" style="display:none; margin-top:8px;">
              <label class="field">
                <span data-i18n="t1b_custom_sha">手动指定 GGUF SHA-256（仅当库中尚无缓存时填写）</span>
                <input id="embLocalSha" placeholder="64-char SHA-256 or api" />
              </label>
            </div>
          </div>
          <div class="row" style="margin-top:12px;">
            <button id="btnTestLocal" class="btn" style="display:none" onclick="testTier1BLocal()" data-i18n="btn_test_conn">测试连通与维度</button>
            <button class="btn primary" onclick="saveTier1BQueryMode()" data-i18n="btn_apply_t1b">应用提问阶段设置</button>
          </div>
        </div>
      </div>

      <div id="vectorCoverage" class="muted mono" style="margin-top:10px;"></div>
    </div>
  </section>
</main>

<!-- Floating ? Help Popover -->
<div id="helpPopover" role="tooltip">
  <div class="hp-title"><span id="hpTitle"></span><button type="button" class="hp-close" onclick="closeHelp()">×</button></div>
  <div id="hpBody"></div>
</div>

<dialog id="editDialog">
  <h3 id="dlgTitle" style="margin:0 0 12px; font-size:15px;">新建记忆</h3>
  <input type="hidden" id="dlgId" /><input type="hidden" id="dlgVer" />
  <div style="display:flex; flex-direction:column; gap:10px;">
    <label class="field"><span data-i18n="dlg_content_lbl">内容（原子、完整事实，自动拦截密钥）</span><textarea id="dlgContent"></textarea></label>
    <div class="grid2">
      <label class="field">
        <span class="field-head"><span data-i18n="dlg_kind_lbl">类型 (kind)</span><button type="button" class="help-tip" data-help="mem_kind">?</button></span>
        <select id="dlgKind"><option value="episode">episode</option><option value="user_profile">user_profile</option><option value="persona">persona</option><option value="transient">transient</option></select>
      </label>
      <label class="field">
        <span class="field-head"><span data-i18n="dlg_scope_lbl">作用域 (scope)</span><button type="button" class="help-tip" data-help="mem_scope">?</button></span>
        <input id="dlgScope" value="personal" />
      </label>
      <label class="field"><span data-i18n="dlg_topic_lbl">主题 (topic)</span><input id="dlgTopic" placeholder="infra / proxy / preference" /></label>
      <label class="field"><span data-i18n="dlg_machine_lbl">来源机器 (machine)</span><input id="dlgMachine" value="controller" /></label>
    </div>
    <label class="field">
      <span class="field-head"><span data-i18n="dlg_terms_lbl">搜索别名（逗号分隔，建议补中英文关键词）</span><button type="button" class="help-tip" data-help="mem_terms">?</button></span>
      <input id="dlgTerms" data-i18n-placeholder="dlg_terms_ph" placeholder="东京主机, 反向代理, Caddy" />
    </label>
    <div class="row" style="justify-content:flex-end; margin-top:6px;">
      <button class="btn" onclick="document.getElementById('editDialog').close()" data-i18n="btn_cancel">取消</button>
      <button class="btn primary" onclick="submitEditDialog()" data-i18n="btn_save">保存</button>
    </div>
  </div>
</dialog>

<dialog id="revDialog">
  <h3 style="margin:0 0 10px; font-size:15px;" data-i18n="rev_title">修订历史</h3>
  <div id="revBody" style="max-height:360px; overflow-y:auto;"></div>
  <div class="row" style="justify-content:flex-end; margin-top:12px;">
    <button class="btn" onclick="document.getElementById('revDialog').close()" data-i18n="btn_close">关闭</button>
  </div>
</dialog>

<dialog id="capDialog">
  <h3 style="margin:0 0 10px; font-size:15px;" data-i18n="cap_title">写入离线整理收件箱</h3>
  <label class="field"><span data-i18n="cap_content_lbl">草稿内容</span><textarea id="capContent"></textarea></label>
  <label class="field" style="margin-top:8px;"><span data-i18n="dlg_machine_lbl">来源机器 (machine)</span><input id="capMachine" value="controller" /></label>
  <div class="row" style="justify-content:flex-end; margin-top:12px;">
    <button class="btn" onclick="document.getElementById('capDialog').close()" data-i18n="btn_cancel">取消</button>
    <button class="btn primary" onclick="submitCapture()" data-i18n="btn_submit_draft">提交草稿</button>
  </div>
</dialog>

<div id="toast" class="toast"></div>

<script>
function detectLsmLang() {
  const raw = (localStorage.getItem('lang') || navigator.language || 'en').toLowerCase();
  return raw.startsWith('zh') ? 'zh' : 'en';
}
let LANG = (function initLang() {
  const manual = (localStorage.getItem('lsm_memory_lang') || '').toLowerCase();
  if (manual === 'zh' || manual === 'en') return manual;
  return detectLsmLang();
})();
let GLOBAL_PAUSED = false;
let LAST_STATUS = null;
let MEMORIES = [];
let LAST_IS_SEARCH = false;
let GRAPH = { nodes: [], edges: [] };
let SELECTED_NODE = null;
let AUTH_TOKEN = '';
let GOV_DATA = null;
let EMBED_CFG = null;
let SELECTED_TIER = 'tier0';
let T1B_QMODE = 'offline';
let LEGACY_COUNT = 0;

const I18N = {
  zh: {
    app_title: 'LSM 记忆库', back_to_lsm: '← 返回 LSM WebUI',
    tab_memories: '记忆管理', tab_graph: '关系星图 (2D)', tab_inbox: '收件箱与整理', tab_gov: '机器权限与多档配置',
    search_ph: '搜索记忆（支持中英文别名、端口、路径）...',
    opt_all_scopes: '全部作用域', opt_all_machines: '全部机器',
    show_suppressed: '含已挂起', show_forgotten: '含已遗忘',
    btn_search: '检索', btn_reset: '全部', btn_add: '+ 新建记忆',
    col_content: '内容与搜索别名', col_meta: '来源 / 机器', col_status: '状态', col_actions: '操作',
    col_draft: '草稿内容', col_machine: '机器名', col_mem_inbox: '记忆数 / 待整理',
    col_include: '选入 / 剔除', col_role: '权限档位', col_auto_cons: '自动整理', col_save: '保存',
    graph_hint: '拖拽节点可调整位置，滚轮缩放，点击节点高亮 1-hop 关联邻居并在右侧操作。',
    btn_refresh: '刷新图谱', btn_reset_view: '重置视图',
    graph_select_prompt: '点击左侧任意节点查看详情、编辑、挂起或建立关系边。',
    inbox_title: '离线整理收件箱（Staging Inbox）',
    opt_pending: 'pending (待整理)', opt_consolidated: 'consolidated (已整理)', opt_skipped: 'skipped (已跳过)',
    btn_copy_prompt: '复制 Agent 整理指令', btn_capture: '+ 写入草稿',
    gov_global_title: '自动化与权限总控',
    gov_auto_inject: '开启 WebUI 自动分体注入',
    gov_default_inc: '新发现机器默认纳入',
    gov_mcp_role: '外部 /mcp 默认权限档位',
    role_rw_full: 'read_write（完全读写管理）',
    role_rc_full: 'read_capture（只读 + 仅可丢收件箱草稿）',
    role_ro_full: 'read_only（纯只读，防误删与提示词注入）',
    gov_consolidate_mode: '离线整理模式',
    mode_manual: 'manual（手动 / Agent 触发）',
    mode_auto: 'auto（定时全自动整理）',
    gov_interval: '自动整理间隔（分钟，5–1440）',
    btn_save_gov: '保存总控策略',
    gov_machines_title: '按机器选入 / 剔除与权限治理',
    machine_ph: '机器名，如 gpu-worker / node-01',
    btn_add_machine: '+ 添加机器',
    embed_title: '多档检索引擎配置（Tier 0 / Tier 1A / Tier 1B 热切换）',
    tier0_name: 'Tier 0 · 零模型默认',
    tier0_short: '词法 BM25 + 3× 双语别名 + MaxSim + 1-hop 关系图谱，无需外部 API 或 GPU。',
    tier1a_name: 'Tier 1A · 云端向量 API',
    tier1a_short: '接入任意 OpenAI 兼容 /v1/embeddings API，主控保持 1C1G 轻量并支持一键云端补齐向量。',
    tier1b_name: 'Tier 1B · Worker / 本地模型',
    tier1b_short: '调用在线 Worker 的本地 GPU/CPU llama-server 批量算向量（空闲 0 显存），可选主控本地检索。',
    tier0_pane_title: '零模型模式（推荐 1C1G 主控）',
    tier0_pane_desc: '无需填写任何 API Key 或本地模型路径。所有向量检索超时或未启用时也会自动安全降级到此档位。',
    btn_use_tier0: '启用 Tier 0 零模型',
    tier1a_pane_title: '云端 OpenAI 兼容 /v1/embeddings 配置',
    lbl_api_key: 'API Key（留空表示保留已存密钥）',
    btn_test_conn: '测试连通与维度', btn_save_apply: '保存并立即热生效', btn_cloud_backfill: '一键补齐缺失向量',
    t1b_worker_title: '① 选一台在线 Worker 执行离线批量算向量（按需拉起 llama-server，用完即退、空闲 0 显存）',
    t1b_select_worker: '选择执行计算的机器（Worker）',
    t1b_llama_path: '该机器上的 llama-server 路径',
    t1b_gguf_path: '该机器上的 .gguf 模型路径',
    btn_run_worker: '一键在所选 Worker 补齐向量',
    btn_save_worker_profile: '保存 Worker 路径配置',
    t1b_cli_summary: '查看等效命令行（适合超大语料或离线终端手动执行）',
    btn_copy_cli: '复制命令',
    t1b_query_title: '② 主控提问阶段（Query-Time）向量来源',
    t1b_qmode_offline: '仅保留离线向量库（主控维持 Tier 0 · ~14 MB）',
    t1b_qmode_offline_desc: '适合 1C1G 低配主控：Worker 算好的向量已存入库中，日常提问仍走零模型极速召回。',
    t1b_qmode_local: '主控本机 llama-server 混合检索（4G+ 主机）',
    t1b_qmode_local_desc: '主控本机常驻同款 GGUF 环回服务，每次提问实时走混合 RRF 语义召回。',
    t1b_local_url: '主控环回 Endpoint URL',
    t1b_local_model: '模型名称（与 GGUF 文件名一致）',
    t1b_fp_select: '匹配已有向量缓存（自动识别指纹）',
    t1b_custom_sha: '手动指定 GGUF SHA-256（仅当库中尚无缓存时填写）',
    btn_apply_t1b: '应用提问阶段设置',
    dlg_content_lbl: '内容（原子、完整事实，自动拦截密钥）',
    dlg_kind_lbl: '类型 (kind)', dlg_scope_lbl: '作用域 (scope)',
    dlg_topic_lbl: '主题 (topic)', dlg_machine_lbl: '来源机器 (machine)',
    dlg_terms_lbl: '搜索别名（逗号分隔，建议补中英文关键词）',
    dlg_terms_ph: '东京主机, 反向代理, Caddy',
    btn_cancel: '取消', btn_save: '保存', btn_close: '关闭',
    rev_title: '修订历史', cap_title: '写入离线整理收件箱', cap_content_lbl: '草稿内容', btn_submit_draft: '提交草稿',
    // Dynamic strings
    pause_on: '▶ 恢复记忆库（已暂停）', pause_off: '⏸ 全局暂停',
    tier0_badge: 'Tier 0 (零模型)', tier1_badge: 'Tier 1 (混合 RRF)',
    lang_btn_sync: '中文 · 随 LSM', lang_btn_manual: '中文 (手动)',
    empty_mem: '暂无记忆条目', empty_inbox: '暂无草稿', empty_revs: '无历史记录',
    empty_machines: '暂无机器记录，可在右上角手动添加机器名。',
    btn_edit: '编辑', btn_suppress: '挂起', btn_restore: '恢复', btn_history: '历史', btn_forget: '遗忘',
    btn_mark_done: '标已整理', btn_skip: '跳过', btn_apply: '应用', btn_link: '连边', btn_unlink: '删边',
    st_active: '活跃', st_suppressed: '已挂起', st_forgotten: '已遗忘',
    opt_inc: '纳入 (Included)', opt_exc: '剔除 (Excluded)',
    opt_rw: '管理读写', opt_rc: '只读+草稿', opt_ro: '纯只读',
    insp_rels: '1-hop 关系边', insp_no_rels: '暂无关联边', insp_target_ph: '目标ID',
    dlg_add_title: '新建记忆', dlg_edit_title: '编辑记忆',
    legacy_btn: '导入原版记忆', legacy_confirm: '将原版 LSM webui_memories 无损复制到 LSMMemory（不会删除原版数据）？',
    forget_confirm: '确定永久遗忘记忆 #{id}？（将写入墓碑防止备份复活）',
    vec_cache_none: '向量缓存：暂无已存储向量', vec_cache_prefix: '向量缓存：',
    fp_custom_opt: '自定义 SHA-256（手动输入）',
    fp_none_opt: '暂无缓存向量（请先在上方 ① 执行 Worker 补齐）',
    worker_ctrl_opt: 'controller（主控本机运行）',
    msg_working: '执行中，请稍候...',
    msg_pause_on: '已开启全局暂停', msg_pause_off: '已恢复自动召回',
    msg_gov_saved: '总控策略已生效', msg_machine_saved: '机器 {name} 策略已生效',
    msg_tier0_on: '已切回 Tier 0 零模型模式', msg_embed_saved: '多档向量配置已保存并实时热生效',
    msg_worker_saved: 'Worker 路径配置已保存', msg_cli_copied: '命令已复制到剪贴板',
    msg_prompt_copied: '已复制整理指令，可粘贴到任意已连接 LSM 的对话',
    msg_need_target: '请输入目标记忆 ID', msg_linked: '已连边 #{s} → #{t}', msg_unlinked: '已删除关系边',
    msg_added: '已创建记忆', msg_updated: '已更新 #{id}', msg_sup: '已软挂起 #{id}', msg_res: '已恢复 #{id}', msg_fgt: '已遗忘 #{id}',
    msg_captured: '草稿已加入收件箱', msg_need_worker_paths: '请填写该机器上的 llama-server 和 .gguf 模型路径',
    consolidate_prompt_text: '请调用 lsm_memory:memory_consolidate_pull 拉取待整理收件箱，提炼为简短原子记忆并补全中英文 search_terms 别名与必要关系边，最后调用 lsm_memory:memory_consolidate_commit 原子提交。'
  },
  en: {
    app_title: 'LSM Memory Vault', back_to_lsm: '← LSM WebUI',
    tab_memories: 'Memories', tab_graph: 'Relation Graph (2D)', tab_inbox: 'Inbox & Consolidation', tab_gov: 'Governance & Tiers',
    search_ph: 'Search memories (bilingual aliases, ports, paths)...',
    opt_all_scopes: 'All Scopes', opt_all_machines: 'All Machines',
    show_suppressed: 'Suppressed', show_forgotten: 'Forgotten',
    btn_search: 'Search', btn_reset: 'All', btn_add: '+ Add Memory',
    col_content: 'Content & Search Aliases', col_meta: 'Source / Machine', col_status: 'Status', col_actions: 'Actions',
    col_draft: 'Draft Content', col_machine: 'Machine', col_mem_inbox: 'Memories / Pending',
    col_include: 'Include / Exclude', col_role: 'Access Role', col_auto_cons: 'Auto-Consolidate', col_save: 'Save',
    graph_hint: 'Drag nodes to rearrange, scroll to zoom, click a node to highlight 1-hop neighbors and inspect on the right.',
    btn_refresh: 'Refresh Graph', btn_reset_view: 'Reset View',
    graph_select_prompt: 'Click any node on the left to inspect, edit, suppress, or link relations.',
    inbox_title: 'Offline Consolidation Staging Inbox',
    opt_pending: 'pending', opt_consolidated: 'consolidated', opt_skipped: 'skipped',
    btn_copy_prompt: 'Copy Agent Consolidation Prompt', btn_capture: '+ Add Draft',
    gov_global_title: 'Global Automation & Access Control',
    gov_auto_inject: 'Enable WebUI Auto Split-Injection',
    gov_default_inc: 'Include Newly Discovered Machines by Default',
    gov_mcp_role: 'External /mcp Default Access Role',
    role_rw_full: 'read_write (Full read/write management)',
    role_rc_full: 'read_capture (Read + staging inbox drafts only)',
    role_ro_full: 'read_only (Read-only; prevents deletion & prompt injection)',
    gov_consolidate_mode: 'Offline Consolidation Mode',
    mode_manual: 'manual (User / Agent triggered)',
    mode_auto: 'auto (Scheduled automatic consolidation)',
    gov_interval: 'Auto-Consolidation Interval (minutes, 5–1440)',
    btn_save_gov: 'Save Global Policy',
    gov_machines_title: 'Per-Machine Include / Exclude & Role Governance',
    machine_ph: 'Machine name, e.g. gpu-worker / node-01',
    btn_add_machine: '+ Add Machine',
    embed_title: 'Multi-Tier Retrieval Engine (Tier 0 / Tier 1A / Tier 1B Hot-Switch)',
    tier0_name: 'Tier 0 · Zero-Model Default',
    tier0_short: 'Lexical BM25 + 3× bilingual aliases + MaxSim + 1-hop relation graph. Zero API keys or GPU required.',
    tier1a_name: 'Tier 1A · Cloud Embeddings API',
    tier1a_short: 'Connect any OpenAI-compatible /v1/embeddings API. Keeps controller at ~14 MB with 1-click cloud backfill.',
    tier1b_name: 'Tier 1B · Worker / Local Model',
    tier1b_short: 'Run on-demand llama-server on an online GPU/CPU Worker (0 MB idle VRAM), with optional local query-time RRF.',
    tier0_pane_title: 'Zero-Model Mode (Recommended for 1C1G Controllers)',
    tier0_pane_desc: 'Zero configuration required. Any vector provider timeout or error also fails closed automatically to Tier 0.',
    btn_use_tier0: 'Activate Tier 0 (Zero-Model)',
    tier1a_pane_title: 'Cloud OpenAI-Compatible /v1/embeddings Configuration',
    lbl_api_key: 'API Key (leave blank to keep stored key)',
    btn_test_conn: 'Test Connection & Dim', btn_save_apply: 'Save & Hot-Apply', btn_cloud_backfill: 'Backfill Missing Vectors',
    t1b_worker_title: '1. Offline Batch Vector Backfill on an Online Worker (on-demand llama-server, 0 MB idle VRAM)',
    t1b_select_worker: 'Target Compute Machine (Worker)',
    t1b_llama_path: 'llama-server Executable Path on Worker',
    t1b_gguf_path: 'GGUF Model Path on Worker',
    btn_run_worker: 'Run Backfill on Selected Worker',
    btn_save_worker_profile: 'Save Worker Paths',
    t1b_cli_summary: 'Show equivalent CLI command (for large corpora or manual terminal runs)',
    btn_copy_cli: 'Copy Command',
    t1b_query_title: '2. Controller Query-Time Vector Source',
    t1b_qmode_offline: 'Offline Index Only (Keep Controller at Tier 0 · ~14 MB)',
    t1b_qmode_offline_desc: 'Best for 1C1G hosts: Worker vectors are cached in SQLite while live turns stay on fast zero-model retrieval.',
    t1b_qmode_local: 'Local llama-server Hybrid RRF on Controller (4G+ Host)',
    t1b_qmode_local_desc: 'Controller runs the same GGUF locally on loopback for real-time query-time semantic RRF.',
    t1b_local_url: 'Controller Loopback Endpoint URL',
    t1b_local_model: 'Model Name (matching GGUF filename)',
    t1b_fp_select: 'Match Existing Vector Cache (Auto-Detected Fingerprint)',
    t1b_custom_sha: 'Custom GGUF SHA-256 (only if no vectors cached yet)',
    btn_apply_t1b: 'Apply Query-Time Setting',
    dlg_content_lbl: 'Content (atomic fact; API keys & tokens are blocked automatically)',
    dlg_kind_lbl: 'Kind', dlg_scope_lbl: 'Scope',
    dlg_topic_lbl: 'Topic', dlg_machine_lbl: 'Source Machine',
    dlg_terms_lbl: 'Search Aliases (comma-separated bilingual keywords recommended)',
    dlg_terms_ph: 'tokyo-host, reverse-proxy, Caddy',
    btn_cancel: 'Cancel', btn_save: 'Save', btn_close: 'Close',
    rev_title: 'Revision History', cap_title: 'Capture Draft to Staging Inbox', cap_content_lbl: 'Draft Content', btn_submit_draft: 'Submit Draft',
    pause_on: '▶ Resume Vault (Paused)', pause_off: '⏸ Global Pause',
    tier0_badge: 'Tier 0 (Zero-Model)', tier1_badge: 'Tier 1 (Hybrid RRF)',
    lang_btn_sync: 'EN · Synced', lang_btn_manual: 'EN (Manual)',
    empty_mem: 'No memories found', empty_inbox: 'No inbox drafts', empty_revs: 'No revision history',
    empty_machines: 'No machines recorded yet. Add one at top-right.',
    btn_edit: 'Edit', btn_suppress: 'Suppress', btn_restore: 'Restore', btn_history: 'History', btn_forget: 'Forget',
    btn_mark_done: 'Mark Done', btn_skip: 'Skip', btn_apply: 'Apply', btn_link: 'Link', btn_unlink: 'Unlink',
    st_active: 'active', st_suppressed: 'suppressed', st_forgotten: 'forgotten',
    opt_inc: 'Included', opt_exc: 'Excluded',
    opt_rw: 'read_write', opt_rc: 'read_capture', opt_ro: 'read_only',
    insp_rels: '1-hop Relations', insp_no_rels: 'No relations yet', insp_target_ph: 'Target ID',
    dlg_add_title: 'Add Memory', dlg_edit_title: 'Edit Memory',
    legacy_btn: 'Import Legacy Memories', legacy_confirm: 'Copy legacy LSM webui_memories into LSMMemory non-destructively? (Original data is kept intact.)',
    forget_confirm: 'Permanently forget memory #{id}? (Writes a tombstone hash to block resurrection from backups.)',
    vec_cache_none: 'Vector Cache: 0 vectors stored', vec_cache_prefix: 'Vector Cache: ',
    fp_custom_opt: 'Custom SHA-256 (manual input)',
    fp_none_opt: 'No cached vectors yet (run Worker Backfill in step 1 first)',
    worker_ctrl_opt: 'controller (run locally on controller)',
    msg_working: 'Running, please wait...',
    msg_pause_on: 'Global pause enabled', msg_pause_off: 'Global pause disabled',
    msg_gov_saved: 'Global governance policy applied', msg_machine_saved: 'Policy applied for {name}',
    msg_tier0_on: 'Switched to Tier 0 (Zero-Model)', msg_embed_saved: 'Embedding configuration saved & hot-applied',
    msg_worker_saved: 'Worker profile saved', msg_cli_copied: 'Command copied to clipboard',
    msg_prompt_copied: 'Consolidation prompt copied — paste into any LSM-connected chat',
    msg_need_target: 'Enter a target memory ID', msg_linked: 'Linked #{s} → #{t}', msg_unlinked: 'Relation deleted',
    msg_added: 'Memory created', msg_updated: 'Updated #{id}', msg_sup: 'Suppressed #{id}', msg_res: 'Restored #{id}', msg_fgt: 'Forgotten #{id}',
    msg_captured: 'Draft added to inbox', msg_need_worker_paths: 'Enter both llama-server and .gguf paths on the target worker',
    consolidate_prompt_text: 'Call lsm_memory:memory_consolidate_pull to fetch pending inbox drafts, distill them into concise atomic memories with bilingual search_terms and relations, then commit atomically via lsm_memory:memory_consolidate_commit.'
  }
};

const HELP = {
  zh: {
    global_pause: ['全局暂停', '立即暂停全网自动记忆召回与 WebUI 分体注入（包括 AgentSuspended 挂起恢复时），但仍允许手动检索、审计和编辑记忆。'],
    search_engine: ['混合检索与别名召回', '支持中文 2-gram、英文词项、精确短标识（端口/IP/路径/版本号）及 3 倍加权的 search_terms 别名；命中后还会自动沿 1-hop 关系边联想相关记忆。'],
    suppress_vs_forget: ['软挂起 (Suppress) vs 永久遗忘 (Forget)', '• 挂起：临时移出自动召回，数据与修订历史保留，可随时一键恢复。\n• 遗忘：标记永久删除并写入内容哈希墓碑（tombstone），防止旧备份或同步脚本将其复活。'],
    mem_kind: ['记忆类型 (kind)', '• episode：具体事件、部署与排障事实（默认）\n• user_profile：长期个人偏好与环境约定\n• persona：助手协作风格设定\n• transient：短期临时上下文'],
    mem_scope: ['作用域隔离 (scope)', '默认 personal 供全局共享；填 agent:<id> 则仅对对应 LSM 专属 Agent 可见，与 LSM 原生分区隔离保持一致。'],
    mem_terms: ['搜索别名 (search_terms)', '在零模型（Tier 0）下获得高召回的关键：用逗号补充中英文同义词、缩写或主机别名（如“东京主机, tokyo-arm, Caddy”），检索时享受 3 倍权重加成。'],
    relation_graph: ['2D 关系星图与 1-hop 联想', '节点按 topic 着色，虚线半透明节点代表已挂起记忆。这里维护的关系边不只是展示——检索命中任一节点时，检索引擎会自动顺着边拉出 1-hop 关联记忆。'],
    graph_rels: ['关系边类型说明', '• related：普通主题关联\n• depends_on：配置/服务依赖关系\n• supersedes：新结论替代旧结论（提示模型优先信新记忆）'],
    staging_inbox: ['为什么先进入收件箱（Staging Inbox）？', '收件箱是原始笔记与会话碎片的缓冲池，避免把零散对话直接塞入正式记忆库。你可以随时让任意已接入 LSM 的 Agent 批量提炼成原子记忆并补全双语别名。'],
    consolidate_prompt: ['零额外 API Key 离线整理', '复制此指令粘贴到任意已连接 LSM 的对话（WebUI / Gemini / Claude / Codex），Agent 会自动调用 pull 拉取待整理草稿并调用 commit 原子入库，无需单独配置 LLM API Key。'],
    gov_global: ['自动化与权限总控', '所有策略直接落盘在主控 SQLite 元数据中并由服务端强制校验，对 LSM WebUI 以及全部外部 /mcp 客户端实时生效。'],
    auto_inject: ['WebUI 自动分体注入', '关闭后，LSM WebUI 对话不再自动注入 <lsm-memory-recall> 块；开启时仅贴附于当轮 user 投影副本，绝不改动 messages[0]（完整保 Prompt Cache）也不污染 Checkpoint。'],
    default_included: ['新机器默认纳入 vs 严格白名单', '• 勾选：新接入的机器自动参与召回与整理。\n• 取消勾选：进入严格白名单模式，未知机器默认被隔离，必须先在下方表格手动设为“纳入”才生效。'],
    mcp_role: ['外部 /mcp 默认权限档位', '控制 Gemini Web / Claude / Codex 等外部客户端通过 /mcp 写入的上限：\n• read_write：完整读写与管理\n• read_capture：只读 + 允许丢草稿到收件箱（推荐协作模式）\n• read_only：纯只读，彻底防止误删或提示词注入污染记忆库'],
    consolidate_mode: ['离线整理模式', '• manual：由你在需要时手动点按钮或让 Agent 整理（推荐，零后台消耗）。\n• auto：标记启用周期自动整理策略，供外部定时任务或调度器按下方间隔执行。'],
    consolidate_interval: ['自动整理间隔', '仅在整理模式设为 auto 时生效，范围 5–1440 分钟。'],
    machine_gov: ['跨网络按机器治理', '自动合并显示当前在线的 LSM Worker（含 online/offline 状态）及历史写入过记忆的机器，可对单台机器单独开关“选入/剔除”和权限档位。'],
    machine_include: ['选入 (Included) / 剔除 (Excluded)', '设为“剔除”后，该机器产出的记忆立即从全网检索与自动注入中隐藏，其收件箱草稿也会被整理流程跳过（不删原始数据，随时可重新纳入）。'],
    machine_role: ['单机权限档位', '若单机权限比全局外部 /mcp 权限更严格，服务端会按更严格的一方执行（例如把某台测试机锁为 read_only 或 read_capture）。'],
    machine_auto: ['单机自动整理开关', '取消勾选后，该机器的草稿不会被自动整理任务纳入，仅保留人工手动处理。'],
    tier_overview: ['三档检索架构与 750ms 安全降级', 'LSMMemory 针对 1C1G 低配主控设计了三档平滑升级路径：默认 Tier 0 仅占 ~14 MB 内存；启用 Tier 1A/1B 后自动走“词法 + 向量 RRF 混合检索”。任何外部向量接口超时（>750ms）或离线时，系统自动无感降级回 Tier 0，永不阻塞对话。'],
    tier0_card: ['Tier 0 · 零模型默认', '仅需 Python 标准库，不加载任何本地模型、不调外部 Embedding API。依靠 BM25 + 3× 双语别名 + 精确短标识 + MaxSim + 1-hop 关系图谱，在 200 条双语库实测 Recall@5 达 96.0%。'],
    tier1a_card: ['Tier 1A · 云端向量 API', '主控仍保持 ~14 MB 极低内存：提问与记忆批量向量化均由你指定的云端 OpenAI 兼容 /v1/embeddings API 完成（API Key 以 0600 落盘且永不明文回显）。'],
    tier1b_card: ['Tier 1B · Worker / 本地模型', '完全私有离线方案，分为两步：\n1. 离线批量算向量：选一台带 GPU/CPU 的在线 Worker，临时拉起其本地 llama-server 算完向量写回主控后立即退出（空闲 0 显存）；\n2. 提问阶段向量源：1C1G 主控可保持 Tier 0 轻量，4G+ 主控可挂载本机同款 GGUF llama-server 启用实时语义 RRF。'],
    tier1a_fields: ['云端 API 必填项说明', '只需填 3 项：Endpoint URL（如 SiliconFlow / OpenAI / 任意兼容接口）、Model Name 和 API Key。云端模式的逻辑哈希会自动设为 api，无需手动填写 SHA-256。'],
    cloud_backfill: ['一键云端补齐缺失向量', '直接由主控调用当前已保存的云端 /v1/embeddings API，为库中尚未生成该模型向量的活跃记忆批量计算并写入 embedding_cache。'],
    t1b_worker_box: ['安全一键 Worker 补齐原理', '主控通过 LSM 既有加密 Worker 通道向所选在线 Worker 下发纯标准库脚本，无需 SSH、无需在 Worker 预装 lsm_memory、也不下发任何主控密钥或 PIN：Worker 在本地哈希 GGUF、临时拉起环回 llama-server 算完向量后立即关闭释放全部显存，主控收到向量后写入数据库。'],
    t1b_worker_pick: ['如何选择计算机器', '下拉框自动列出当前已接入 LSM 的在线 Worker（如带独显的工作站节点）以及 controller 本机。推荐选择带 GPU 的在线 Worker，百条记忆通常 1 秒内算完。'],
    t1b_query_box: ['为什么要分“离线补齐”和“提问阶段”？', '向量的余弦对比要求“文档向量”和“提问向量”来自同一模型。如果你的主控是 1C1G 内存主机，跑不动本地 GGUF 模型，就选左侧“仅保留离线向量库（主控维持 Tier 0）”；如果主控是 4G+ 主机并常驻了同款 GGUF llama-server，选右侧即可实时启用混合语义检索。'],
    t1b_qmode_offline_help: ['仅保留离线向量库', '主控保持 ~14 MB 内存不运行任何模型。Worker 算好的跨平台向量已安全存储在 SQLite，未来升级主控硬件或挂载云端/本地同款模型后可直接零重算复用。'],
    t1b_qmode_local_help: ['主控本机 llama-server 混合检索', '要求主控本机（127.0.0.1）运行与 Worker 相同的 GGUF 文件，使逻辑指纹完全匹配，即可实时计算提问向量并与库中已有向量做混合 RRF 排序。'],
    t1b_fp_help: ['逻辑向量指纹自动匹配', '下拉列表直接从主控 /v1/status 读取已入库的向量缓存指纹与维度，省去手动复制 64 位 SHA-256 的麻烦。']
  },
  en: {
    global_pause: ['Global Pause', 'Immediately halts automatic memory recall and WebUI split-injection globally (including after AgentSuspended resume), while keeping manual search, audit, and editing available.'],
    search_engine: ['Adaptive Search & Alias Recall', 'Combines CJK bigrams, ASCII terms, exact short identifiers (ports/IPs/paths/versions), and 3× boosted search_terms aliases, then expands 1-hop relation neighbors automatically.'],
    suppress_vs_forget: ['Soft Suppress vs. Permanent Forget', '• Suppress: Temporarily hides a memory from recall while keeping its data and revisions; reversible anytime.\n• Forget: Permanently marks deleted and writes a content-hash tombstone to prevent resurrection from backups.'],
    mem_kind: ['Memory Kind', '• episode: Concrete events, deployments, or troubleshooting facts (default)\n• user_profile: Durable personal preferences and conventions\n• persona: Assistant collaboration style\n• transient: Short-lived temporary context'],
    mem_scope: ['Scope Isolation', 'Default "personal" is shared globally; "agent:<id>" isolates the memory to that specific LSM agent, matching LSM native partitioning.'],
    mem_terms: ['Search Aliases (search_terms)', 'Key to high Recall@5 in Tier 0 zero-model mode: add comma-separated bilingual synonyms, abbreviations, or host nicknames for a 3× BM25 weight boost.'],
    relation_graph: ['2D Relation Graph & 1-Hop Expansion', 'Nodes are colored by topic; dashed faded nodes are suppressed. Edges are live retrieval paths — when a memory matches a query, its 1-hop linked neighbors are recalled automatically.'],
    graph_rels: ['Relation Edge Types', '• related: General topical link\n• depends_on: Service or configuration dependency\n• supersedes: Newer fact replaces an older one'],
    staging_inbox: ['Why Staging Inbox?', 'Raw notes and session fragments land here first so long-term memory stays clean and atomic. Any LSM-connected Agent can batch-distill drafts into clean memories with bilingual aliases.'],
    consolidate_prompt: ['Zero-Extra-API-Key Consolidation', 'Paste this prompt into any LSM-connected chat (WebUI, Gemini, Claude, Codex). The Agent calls memory_consolidate_pull, distills atomic facts + aliases, and commits atomically via memory_consolidate_commit.'],
    gov_global: ['Global Automation & Access Control', 'Policies are stored in the controller SQLite metadata and enforced server-side across LSM WebUI and all external /mcp clients immediately.'],
    auto_inject: ['WebUI Auto Split-Injection', 'When enabled, attaches <lsm-memory-recall> only to an ephemeral copy of the latest user turn projection — never mutates messages[0] (preserving Prompt Cache) or SQLite checkpoints.'],
    default_included: ['Open-by-Default vs. Strict Allowlist', '• Checked: Newly discovered machines are automatically included in recall and consolidation.\n• Unchecked: Strict allowlist mode — unknown machines are excluded until explicitly included below.'],
    mcp_role: ['External /mcp Default Role', 'Caps write privileges for external /mcp clients:\n• read_write: Full read/write management\n• read_capture: Read + staging inbox drafts only (recommended for shared agents)\n• read_only: Strictly read-only, blocking accidental deletion and prompt injection'],
    consolidate_mode: ['Offline Consolidation Mode', '• manual: Triggered on demand by you or an Agent (recommended, zero background overhead).\n• auto: Enables scheduled periodic consolidation policy using the interval below.'],
    consolidate_interval: ['Auto-Consolidation Interval', 'Used when Consolidation Mode is set to auto (range: 5–1440 minutes).'],
    machine_gov: ['Cross-Network Machine Governance', 'Merges live online/offline LSM Workers with historical machines seen in memory records. Override include/exclude and access roles per host.'],
    machine_include: ['Include / Exclude Switch', 'Excluded machines have their memories hidden from search & auto-injection and their inbox drafts skipped during consolidation (non-destructive; reversible anytime).'],
    machine_role: ['Per-Machine Access Role', 'Enforces the stricter of the global /mcp role and this machine role (e.g., lock a staging host to read_only or read_capture).'],
    machine_auto: ['Per-Machine Auto-Consolidate', 'Uncheck to exclude this machine’s drafts from automated consolidation runs while keeping manual processing available.'],
    tier_overview: ['3-Tier Retrieval & 750ms Fail-Closed Fallback', 'Designed for 1C1G hosts: Tier 0 uses ~14 MB RAM with zero models; Tier 1A/1B adds Hybrid RRF vector search. If any vector endpoint times out (>750ms) or fails, retrieval falls back silently to Tier 0.'],
    tier0_card: ['Tier 0 · Zero-Model Default', 'Pure Python stdlib (~14 MB RAM), zero API keys or GPU needed. Uses BM25 + 3× bilingual aliases + exact short-ID channel + MaxSim + 1-hop graph (96.0% Recall@5 on 200-item bilingual benchmark).'],
    tier1a_card: ['Tier 1A · Cloud Embeddings API', 'Keeps the controller at ~14 MB while routing query & batch embeddings to any OpenAI-compatible /v1/embeddings API (API key saved chmod 600 and never returned in plaintext).'],
    tier1b_card: ['Tier 1B · Worker / Local Model', 'Fully private local setup in two steps:\n1. Offline Batch Backfill: Pick an online GPU/CPU Worker to launch llama-server on demand, upload vectors, and exit (0 MB idle VRAM).\n2. Query-Time Source: Keep 1C1G controllers on Tier 0, or attach a local controller llama-server (4G+ host) for live Hybrid RRF.'],
    tier1a_fields: ['Cloud API Fields', 'Only 3 fields are needed: Endpoint URL, Model Name, and API Key. Logical model hash defaults automatically to "api" — no manual SHA-256 needed.'],
    cloud_backfill: ['1-Click Cloud Vector Backfill', 'Runs directly on the controller using your saved /v1/embeddings API to compute and cache vectors for all active memories missing this model fingerprint.'],
    t1b_worker_box: ['How 1-Click Worker Backfill Works', 'Dispatches a self-contained stdlib Python task over LSM’s authenticated Worker channel — zero SSH, zero lsm_memory install on the Worker, and zero secrets sent to the Worker. The Worker hashes the GGUF, starts loopback llama-server, returns vectors, and terminates llama-server (0 MB idle VRAM).'],
    t1b_worker_pick: ['Choosing a Compute Worker', 'Automatically lists online LSM Workers discovered via remote_manager plus "controller". Pick a GPU-equipped Worker for fastest throughput (~150+ items/s).'],
    t1b_query_box: ['Offline Backfill vs. Query-Time Source', 'Cosine search requires query and document vectors from the same model. On a 1C1G controller, choose "Offline Index Only" to stay at ~14 MB RAM; on a 4G+ controller running the same GGUF locally, choose "Local llama-server" for live Hybrid RRF.'],
    t1b_qmode_offline_help: ['Offline Index Only (Tier 0 on Controller)', 'Keeps controller RAM at ~14 MB with no resident model. Cross-platform vectors remain cached in SQLite for instant reuse if you later enable a matching query-time model.'],
    t1b_qmode_local_help: ['Controller Local llama-server', 'Connects to a loopback llama-server on the controller hosting the exact same GGUF model so live queries run Hybrid RRF against cached Worker vectors.'],
    t1b_fp_help: ['Auto-Detected Vector Fingerprint', 'Populated directly from /v1/status cached vector fingerprints & dimensions so you never need to copy a 64-character SHA-256 manually.']
  }
};

function t(key, vars) {
  let s = (I18N[LANG] && I18N[LANG][key]) || (I18N.en && I18N.en[key]) || key;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.replaceAll('{' + k + '}', String(v));
  return s;
}

function applyI18n() {
  document.documentElement.lang = LANG === 'zh' ? 'zh-CN' : 'en';
  document.title = t('app_title');
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const k = el.getAttribute('data-i18n');
    if (I18N[LANG][k]) el.textContent = I18N[LANG][k];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const k = el.getAttribute('data-i18n-placeholder');
    if (I18N[LANG][k]) el.placeholder = I18N[LANG][k];
  });
  const isSynced = !localStorage.getItem('lsm_memory_lang');
  document.getElementById('langBtn').textContent = isSynced ? t('lang_btn_sync') : t('lang_btn_manual');
  if (LAST_STATUS) renderStatusBar(LAST_STATUS);
  if (MEMORIES.length || document.getElementById('memTbody').children.length) renderMemories(MEMORIES, LAST_IS_SEARCH);
  if (SELECTED_NODE) renderInspector(SELECTED_NODE);
  if (GOV_DATA) renderMachineTable();
  if (EMBED_CFG) renderEmbedPanel();
  if (LEGACY_COUNT > 0) document.getElementById('legacyImportBtn').textContent = `${t('legacy_btn')} (${LEGACY_COUNT})`;
  closeHelp();
}

function toggleLang() {
  const next = LANG === 'zh' ? 'en' : 'zh';
  LANG = next;
  if (next === detectLsmLang()) localStorage.removeItem('lsm_memory_lang');
  else localStorage.setItem('lsm_memory_lang', next);
  applyI18n();
}

window.addEventListener('storage', (ev) => {
  if (!ev || ev.key === 'lang' || ev.key === 'lsm_memory_lang') {
    const manual = (localStorage.getItem('lsm_memory_lang') || '').toLowerCase();
    const target = (manual === 'zh' || manual === 'en') ? manual : detectLsmLang();
    if (target !== LANG) { LANG = target; applyI18n(); }
  }
});

/* ===== FLOATING ? HELP POPOVER ===== */
let ACTIVE_HELP_BTN = null;
function openHelp(btn) {
  const key = btn.getAttribute('data-help');
  const entry = (HELP[LANG] && HELP[LANG][key]) || (HELP.en && HELP.en[key]);
  if (!entry) return;
  if (ACTIVE_HELP_BTN === btn) { closeHelp(); return; }
  closeHelp();
  ACTIVE_HELP_BTN = btn;
  btn.setAttribute('aria-expanded', 'true');
  const pop = document.getElementById('helpPopover');
  document.getElementById('hpTitle').textContent = entry[0];
  document.getElementById('hpBody').innerHTML = esc(entry[1]).replace(/\n/g, '<br/>');
  pop.style.display = 'block';
  const br = btn.getBoundingClientRect();
  const pr = pop.getBoundingClientRect();
  let left = br.left + br.width / 2 - pr.width / 2;
  left = Math.max(12, Math.min(left, window.innerWidth - pr.width - 12));
  let top = br.bottom + 8;
  if (top + pr.height > window.innerHeight - 12) top = Math.max(12, br.top - pr.height - 8);
  pop.style.left = left + 'px';
  pop.style.top = top + 'px';
}
function closeHelp() {
  if (ACTIVE_HELP_BTN) ACTIVE_HELP_BTN.setAttribute('aria-expanded', 'false');
  ACTIVE_HELP_BTN = null;
  const pop = document.getElementById('helpPopover');
  if (pop) pop.style.display = 'none';
}
document.addEventListener('click', (ev) => {
  const tip = ev.target.closest && ev.target.closest('.help-tip');
  if (tip) { ev.preventDefault(); ev.stopPropagation(); openHelp(tip); return; }
  const pop = document.getElementById('helpPopover');
  if (pop && pop.style.display === 'block' && !pop.contains(ev.target)) closeHelp();
});
document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') closeHelp(); });
window.addEventListener('scroll', closeHelp, true);
window.addEventListener('resize', closeHelp);

function toast(msg, isErr=false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.borderColor = isErr ? 'var(--danger)' : 'var(--accent)';
  el.style.display = 'block';
  clearTimeout(window._tTimer);
  window._tTimer = setTimeout(() => el.style.display = 'none', 3800);
}

async function ensureAuthToken(force=false) {
  if (AUTH_TOKEN && !force) return AUTH_TOKEN;
  const r = await fetch('/api/auth/refresh', { method: 'POST', credentials: 'same-origin' });
  if (r.status === 401) {
    location.href = '/auth/login';
    throw new Error('Authentication required');
  }
  const body = await r.json().catch(() => ({}));
  AUTH_TOKEN = body.token || body.access_token || '';
  return AUTH_TOKEN;
}

async function api(path, opts={}) {
  const send = async (tok) => {
    const headers = Object.assign({ 'X-Requested-With': 'LSMMemory' }, opts.headers || {});
    if (tok) headers['Authorization'] = 'Bearer ' + tok;
    let body = opts.body;
    if (body && typeof body === 'object') {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(body);
    }
    return fetch('/api/memory-vault' + path, Object.assign({}, opts, { headers, body, credentials: 'same-origin' }));
  };
  let tok = await ensureAuthToken(false);
  let res = await send(tok);
  if (res.status === 401) {
    tok = await ensureAuthToken(true);
    res = await send(tok);
  }
  const data = await res.json().catch(() => ({ ok: false, message: 'HTTP ' + res.status }));
  if (!res.ok || !data.ok) throw new Error(data.message || data.detail || data.error || ('HTTP ' + res.status));
  return data.data;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function switchTab(name) {
  closeHelp();
  document.querySelectorAll('nav.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('section.panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  if (name === 'graph') loadGraph();
  if (name === 'inbox') loadInbox();
  if (name === 'gov') { loadGovernance(); loadEmbedConfig(); }
}

function renderStatusBar(st) {
  GLOBAL_PAUSED = !!st.global_pause;
  const tb = document.getElementById('tierBadge');
  tb.textContent = st.tier_mode === 'tier0_zero_model' ? t('tier0_badge') : t('tier1_badge');
  tb.className = 'badge ' + (st.tier_mode === 'tier0_zero_model' ? 'ok' : 'warn');
  document.getElementById('rssBadge').textContent = 'RAM ' + (st.rss_mb ?? '--') + ' MB';
  document.getElementById('countBadge').textContent = `${st.counts.active} active / ${st.counts.suppressed} sup / ${st.counts.inbox_pending} inbox`;
  const pb = document.getElementById('pauseBtn');
  pb.textContent = GLOBAL_PAUSED ? t('pause_on') : t('pause_off');
  pb.className = 'btn ' + (GLOBAL_PAUSED ? 'danger' : '');
  const cov = (st.vectors || []).map(v => `fp=${v.fingerprint.slice(0,12)}… dim=${v.dim} count=${v.count}`).join(' | ');
  document.getElementById('vectorCoverage').textContent = cov ? (t('vec_cache_prefix') + cov) : t('vec_cache_none');
  populateFingerprintDropdown(st.vectors || []);
}

async function refreshStatus() {
  try {
    LAST_STATUS = await api('/status');
    renderStatusBar(LAST_STATUS);
  } catch (e) { toast(e.message, true); }
}

async function togglePause() {
  try {
    await api('/pause', { method: 'POST', body: { paused: !GLOBAL_PAUSED } });
    await refreshStatus();
    toast(GLOBAL_PAUSED ? t('msg_pause_on') : t('msg_pause_off'));
  } catch (e) { toast(e.message, true); }
}

async function checkLegacy() {
  try {
    const leg = await api('/legacy');
    const btn = document.getElementById('legacyImportBtn');
    if (leg.available && leg.count > 0) {
      LEGACY_COUNT = leg.count;
      btn.style.display = 'inline-block';
      btn.textContent = `${t('legacy_btn')} (${leg.count})`;
    }
  } catch (_) {}
}

async function importLegacy() {
  if (!confirm(t('legacy_confirm'))) return;
  try {
    const r = await api('/legacy', { method: 'POST', body: {} });
    toast(`+${r.imported} / skipped ${r.skipped}`);
    loadMemories(); refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function loadMemories() {
  const scope = document.getElementById('filterScope').value;
  const machine = document.getElementById('filterMachine').value;
  const sup = document.getElementById('chkSuppressed').checked ? '1' : '0';
  const fgt = document.getElementById('chkForgotten').checked ? '1' : '0';
  const qs = new URLSearchParams({ scope, include_suppressed: sup, include_forgotten: fgt, limit: '200' });
  if (machine) qs.set('machine', machine);
  try {
    const data = await api('/memories?' + qs.toString());
    MEMORIES = data.memories || [];
    LAST_IS_SEARCH = false;
    renderMemories(MEMORIES, false);
    populateMachineFilter(MEMORIES);
  } catch (e) { toast(e.message, true); }
}

function populateMachineFilter(rows) {
  const sel = document.getElementById('filterMachine');
  const cur = sel.value;
  const names = [...new Set(rows.map(r => r.machine).filter(Boolean))].sort();
  sel.innerHTML = `<option value="">${esc(t('opt_all_machines'))}</option>` + names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join('');
  sel.value = cur;
}

async function runSearch() {
  const q = document.getElementById('searchInput').value.trim();
  if (!q) return loadMemories();
  const scope = document.getElementById('filterScope').value;
  const machine = document.getElementById('filterMachine').value || undefined;
  try {
    const data = await api('/search', { method: 'POST', body: { query: q, scope: scope === '*' ? null : scope, machine, limit: 20 } });
    MEMORIES = data.results || [];
    LAST_IS_SEARCH = true;
    renderMemories(MEMORIES, true);
    toast(`${data.tier} · ${MEMORIES.length} hits · ${data.elapsed_ms} ms`);
  } catch (e) { toast(e.message, true); }
}

function clearSearch() {
  document.getElementById('searchInput').value = '';
  loadMemories();
}

function renderMemories(rows, isSearch=false) {
  const tb = document.getElementById('memTbody');
  if (!rows.length) { tb.innerHTML = `<tr><td colspan="5" class="muted">${esc(t('empty_mem'))}</td></tr>`; return; }
  tb.innerHTML = rows.map(m => {
    const stBadge = m.status === 'forgotten'
      ? `<span class="badge danger">${esc(t('st_forgotten'))}</span>`
      : (m.suppressed ? `<span class="badge warn">${esc(t('st_suppressed'))}</span>` : `<span class="badge ok">${esc(t('st_active'))}</span>`);
    const terms = (m.search_terms || []).map(x => `<span class="tag">${esc(x)}</span>`).join('');
    const score = isSearch && m.score != null ? `<span class="tag">score:${m.score} ${esc(m.match_channel||'')}</span>` : '';
    const acts = m.status === 'forgotten'
      ? `<button class="btn sm" onclick="viewRevs(${m.id})">${esc(t('btn_history'))}</button>`
      : `<button class="btn sm" onclick="openEditDialog(${m.id})">${esc(t('btn_edit'))}</button>
         <button class="btn sm" onclick="toggleSuppress(${m.id},${!m.suppressed})">${esc(m.suppressed ? t('btn_restore') : t('btn_suppress'))}</button>
         <button class="btn sm" onclick="viewRevs(${m.id})">${esc(t('btn_history'))}</button>
         <button class="btn sm danger" onclick="forgetMem(${m.id})">${esc(t('btn_forget'))}</button>`;
    return `<tr class="${m.suppressed ? 'suppressed-row' : ''}">
      <td class="mono">#${m.id}<div class="muted">v${m.version}</div></td>
      <td>
        <div>${esc(m.content)}</div>
        <div class="tags"><span class="tag">${esc(m.kind)}</span>${m.topic ? `<span class="tag">${esc(m.topic)}</span>` : ''}${score}${terms}</div>
      </td>
      <td class="mono">${esc(m.machine)}<div class="muted">${esc(m.source_client)} · ${esc(m.scope)}</div></td>
      <td>${stBadge}</td>
      <td><div class="row">${acts}</div></td>
    </tr>`;
  }).join('');
}

function openAddDialog() {
  document.getElementById('dlgTitle').textContent = t('dlg_add_title');
  document.getElementById('dlgId').value = '';
  document.getElementById('dlgVer').value = '';
  document.getElementById('dlgContent').value = '';
  document.getElementById('dlgTopic').value = '';
  document.getElementById('dlgTerms').value = '';
  document.getElementById('dlgScope').value = 'personal';
  document.getElementById('dlgMachine').value = 'controller';
  document.getElementById('editDialog').showModal();
}

function openEditDialog(id) {
  const m = MEMORIES.find(x => x.id === id) || GRAPH.nodes.find(x => x.id === id);
  if (!m) return;
  document.getElementById('dlgTitle').textContent = `${t('dlg_edit_title')} #${m.id} (v${m.version})`;
  document.getElementById('dlgId').value = m.id;
  document.getElementById('dlgVer').value = m.version;
  document.getElementById('dlgContent').value = m.content;
  document.getElementById('dlgKind').value = m.kind || 'episode';
  document.getElementById('dlgScope').value = m.scope || 'personal';
  document.getElementById('dlgTopic').value = m.topic || '';
  document.getElementById('dlgMachine').value = m.machine || 'controller';
  document.getElementById('dlgTerms').value = (m.search_terms || []).join(', ');
  document.getElementById('editDialog').showModal();
}

async function submitEditDialog() {
  const id = document.getElementById('dlgId').value;
  const ver = parseInt(document.getElementById('dlgVer').value || '0', 10);
  const content = document.getElementById('dlgContent').value.trim();
  const terms = document.getElementById('dlgTerms').value.split(/[,，]/).map(s => s.trim()).filter(Boolean);
  const topic = document.getElementById('dlgTopic').value.trim();
  try {
    if (id) {
      await api(`/memories/${id}/correct`, { method: 'POST', body: { expected_version: ver, content, search_terms: terms, topic } });
      toast(t('msg_updated', { id }));
    } else {
      await api('/memories', {
        method: 'POST',
        body: {
          content, kind: document.getElementById('dlgKind').value,
          scope: document.getElementById('dlgScope').value.trim() || 'personal',
          topic, search_terms: terms, machine: document.getElementById('dlgMachine').value.trim() || 'controller',
        }
      });
      toast(t('msg_added'));
    }
    document.getElementById('editDialog').close();
    loadMemories(); refreshStatus();
    if (document.getElementById('tab-graph').classList.contains('active')) loadGraph();
  } catch (e) { toast(e.message, true); }
}

async function toggleSuppress(id, suppress) {
  try {
    await api(`/memories/${id}/${suppress ? 'suppress' : 'restore'}`, { method: 'POST', body: {} });
    toast(suppress ? t('msg_sup', { id }) : t('msg_res', { id }));
    loadMemories(); refreshStatus();
    if (document.getElementById('tab-graph').classList.contains('active')) loadGraph();
  } catch (e) { toast(e.message, true); }
}

async function forgetMem(id) {
  if (!confirm(t('forget_confirm', { id }))) return;
  try {
    await api(`/memories/${id}/forget`, { method: 'POST', body: { reason: 'webui console' } });
    toast(t('msg_fgt', { id }));
    loadMemories(); refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function viewRevs(id) {
  try {
    const data = await api(`/memories/${id}/revisions`);
    const body = document.getElementById('revBody');
    body.innerHTML = (data.revisions || []).map(r => `
      <div class="card" style="margin-bottom:8px;">
        <div class="row" style="justify-content:space-between;">
          <strong class="mono">v${r.version} · ${esc(r.action)}</strong>
          <span class="muted mono">${esc(r.actor)} · ${esc(r.created_at)}</span>
        </div>
        <div style="margin-top:6px;">${esc(r.content)}</div>
      </div>`).join('') || `<div class="muted">${esc(t('empty_revs'))}</div>`;
    document.getElementById('revDialog').showModal();
  } catch (e) { toast(e.message, true); }
}

/* ===== TAB 2: ZERO-DEPENDENCY 2D CANVAS RELATION GRAPH ===== */
const GVIEW = { tx: 0, ty: 0, scale: 1, dragNode: null, panning: false, lx: 0, ly: 0 };
const TOPIC_COLORS = ['#10a37f','#3b82f6','#f59e0b','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316'];
function topicColor(topic) {
  let h = 0; for (const c of String(topic || 'general')) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return TOPIC_COLORS[h % TOPIC_COLORS.length];
}

async function loadGraph() {
  try {
    const data = await api('/graph?include_suppressed=1&limit=300');
    const canvas = document.getElementById('graphCanvas');
    const w = canvas.clientWidth || 760, h = canvas.clientHeight || 520;
    canvas.width = w * devicePixelRatio; canvas.height = h * devicePixelRatio;
    const prevPos = new Map((GRAPH.nodes || []).map(n => [n.id, { x: n.x, y: n.y }]));
    const nodes = (data.nodes || []).map((n, i) => {
      const old = prevPos.get(n.id);
      const angle = (i / Math.max(1, data.nodes.length)) * Math.PI * 2;
      const rad = Math.min(w, h) * 0.34 * (0.4 + 0.6 * ((i * 7) % 11) / 11);
      return Object.assign({}, n, { x: old ? old.x : w/2 + Math.cos(angle)*rad, y: old ? old.y : h/2 + Math.sin(angle)*rad, vx: 0, vy: 0 });
    });
    GRAPH = { nodes, edges: data.edges || [] };
    runForceLayout(w, h, 80);
    drawGraph();
    if (SELECTED_NODE) {
      const cur = GRAPH.nodes.find(n => n.id === SELECTED_NODE.id);
      if (cur) renderInspector(cur);
    }
  } catch (e) { toast(e.message, true); }
}

function runForceLayout(w, h, steps) {
  const nodes = GRAPH.nodes, byId = new Map(nodes.map(n => [n.id, n]));
  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx*dx + dy*dy + 1;
        let f = 2200 / d2;
        a.vx += dx * f * 0.04; a.vy += dy * f * 0.04;
        b.vx -= dx * f * 0.04; b.vy -= dy * f * 0.04;
      }
    }
    for (const e of GRAPH.edges) {
      const a = byId.get(e.source_id), b = byId.get(e.target_id);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, dist = Math.hypot(dx, dy) || 1;
      const pull = (dist - 110) * 0.03;
      a.vx += (dx/dist)*pull; a.vy += (dy/dist)*pull;
      b.vx -= (dx/dist)*pull; b.vy -= (dy/dist)*pull;
    }
    for (const n of nodes) {
      n.vx += (w/2 - n.x) * 0.008; n.vy += (h/2 - n.y) * 0.008;
      n.vx *= 0.82; n.vy *= 0.82;
      n.x += n.vx; n.y += n.vy;
    }
  }
}

function resetGraphView() { GVIEW.tx = 0; GVIEW.ty = 0; GVIEW.scale = 1; drawGraph(); }

function drawGraph() {
  const canvas = document.getElementById('graphCanvas');
  const ctx = canvas.getContext('2d');
  const dpr = devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  ctx.save();
  ctx.translate(GVIEW.tx, GVIEW.ty);
  ctx.scale(GVIEW.scale, GVIEW.scale);
  const byId = new Map(GRAPH.nodes.map(n => [n.id, n]));
  const nbrs = new Set();
  if (SELECTED_NODE) {
    nbrs.add(SELECTED_NODE.id);
    for (const e of GRAPH.edges) {
      if (e.source_id === SELECTED_NODE.id) nbrs.add(e.target_id);
      if (e.target_id === SELECTED_NODE.id) nbrs.add(e.source_id);
    }
  }
  const mutedColor = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim() || '#888';
  const textColor = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#eee';

  for (const e of GRAPH.edges) {
    const a = byId.get(e.source_id), b = byId.get(e.target_id);
    if (!a || !b) continue;
    const hi = SELECTED_NODE && (e.source_id === SELECTED_NODE.id || e.target_id === SELECTED_NODE.id);
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = hi ? '#10a37f' : mutedColor;
    ctx.globalAlpha = hi ? 0.95 : 0.35;
    ctx.lineWidth = hi ? 2.2 : 1.2;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const n of GRAPH.nodes) {
    const hi = !SELECTED_NODE || nbrs.has(n.id);
    const isSel = SELECTED_NODE && SELECTED_NODE.id === n.id;
    ctx.save();
    ctx.globalAlpha = n.suppressed ? 0.42 : (hi ? 1 : 0.25);
    ctx.beginPath(); ctx.arc(n.x, n.y, isSel ? 12 : 8.5, 0, Math.PI * 2);
    ctx.fillStyle = topicColor(n.topic);
    ctx.fill();
    if (n.suppressed) { ctx.setLineDash([3,2]); ctx.strokeStyle = mutedColor; ctx.lineWidth = 1.5; ctx.stroke(); }
    else if (isSel) { ctx.strokeStyle = textColor; ctx.lineWidth = 2; ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.fillStyle = textColor; ctx.font = '11px sans-serif';
    ctx.fillText(`#${n.id} ${(n.topic||'').slice(0,14)}`, n.x + 12, n.y + 4);
    ctx.restore();
  }
  ctx.restore();
}

(function bindGraphEvents() {
  const c = document.getElementById('graphCanvas');
  function toWorld(ev) {
    const r = c.getBoundingClientRect();
    return { x: (ev.clientX - r.left - GVIEW.tx) / GVIEW.scale, y: (ev.clientY - r.top - GVIEW.ty) / GVIEW.scale };
  }
  c.addEventListener('mousedown', ev => {
    const p = toWorld(ev);
    const hit = GRAPH.nodes.find(n => Math.hypot(n.x - p.x, n.y - p.y) <= 13 / GVIEW.scale);
    if (hit) { GVIEW.dragNode = hit; SELECTED_NODE = hit; renderInspector(hit); drawGraph(); }
    else { GVIEW.panning = true; GVIEW.lx = ev.clientX; GVIEW.ly = ev.clientY; }
  });
  window.addEventListener('mousemove', ev => {
    if (GVIEW.dragNode) { const p = toWorld(ev); GVIEW.dragNode.x = p.x; GVIEW.dragNode.y = p.y; drawGraph(); }
    else if (GVIEW.panning) { GVIEW.tx += ev.clientX - GVIEW.lx; GVIEW.ty += ev.clientY - GVIEW.ly; GVIEW.lx = ev.clientX; GVIEW.ly = ev.clientY; drawGraph(); }
  });
  window.addEventListener('mouseup', () => { GVIEW.dragNode = null; GVIEW.panning = false; });
  c.addEventListener('wheel', ev => {
    ev.preventDefault();
    const factor = ev.deltaY < 0 ? 1.12 : 0.89;
    GVIEW.scale = Math.max(0.35, Math.min(3.5, GVIEW.scale * factor));
    drawGraph();
  }, { passive: false });
})();

function renderInspector(n) {
  const rels = GRAPH.edges.filter(e => e.source_id === n.id || e.target_id === n.id);
  const relHtml = rels.map(e => `
    <div class="row" style="justify-content:space-between; margin-top:4px;">
      <span class="mono">#${e.source_id} —${esc(e.relation_type)}→ #${e.target_id}</span>
      <button class="btn sm danger" onclick="delRel(${e.source_id},${e.target_id},'${esc(e.relation_type)}')">${esc(t('btn_unlink'))}</button>
    </div>`).join('') || `<div class="muted">${esc(t('insp_no_rels'))}</div>`;
  document.getElementById('graphInspector').innerHTML = `
    <div class="row" style="justify-content:space-between;">
      <strong class="mono">#${n.id} v${n.version}</strong>
      <span class="badge ${n.suppressed ? 'warn' : 'ok'}">${esc(n.suppressed ? t('st_suppressed') : t('st_active'))}</span>
    </div>
    <div style="margin:8px 0;">${esc(n.content)}</div>
    <div class="muted mono">${esc(n.machine)} · ${esc(n.scope)} · ${esc(n.topic)}</div>
    <div class="row" style="margin:10px 0;">
      <button class="btn sm" onclick="openEditDialog(${n.id})">${esc(t('btn_edit'))}</button>
      <button class="btn sm" onclick="toggleSuppress(${n.id},${!n.suppressed})">${esc(n.suppressed ? t('btn_restore') : t('btn_suppress'))}</button>
      <button class="btn sm" onclick="viewRevs(${n.id})">${esc(t('btn_history'))}</button>
    </div>
    <hr style="border:none;border-top:1px solid var(--border);margin:10px 0;" />
    <div class="field-head"><strong style="font-size:12px;">${esc(t('insp_rels'))}</strong><button type="button" class="help-tip" data-help="graph_rels">?</button></div>
    ${relHtml}
    <div class="row" style="margin-top:10px;">
      <input id="relTargetId" type="number" placeholder="${esc(t('insp_target_ph'))}" style="width:92px;" />
      <select id="relType"><option value="related">related</option><option value="depends_on">depends_on</option><option value="supersedes">supersedes</option></select>
      <button class="btn sm primary" onclick="addRel(${n.id})">${esc(t('btn_link'))}</button>
    </div>`;
}

async function addRel(srcId) {
  const dst = parseInt(document.getElementById('relTargetId').value || '0', 10);
  if (!dst) return toast(t('msg_need_target'), true);
  try {
    await api('/relations', { method: 'POST', body: { source_id: srcId, target_id: dst, relation_type: document.getElementById('relType').value } });
    toast(t('msg_linked', { s: srcId, t: dst }));
    loadGraph();
  } catch (e) { toast(e.message, true); }
}

async function delRel(s, tId, rtype) {
  try {
    await api('/relations/delete', { method: 'POST', body: { source_id: s, target_id: tId, relation_type: rtype } });
    toast(t('msg_unlinked'));
    loadGraph();
  } catch (e) { toast(e.message, true); }
}

/* ===== TAB 3: INBOX ===== */
async function loadInbox() {
  const st = document.getElementById('inboxStatus').value;
  try {
    const data = await api('/inbox?status=' + encodeURIComponent(st));
    const tb = document.getElementById('inboxTbody');
    const rows = data.events || [];
    if (!rows.length) { tb.innerHTML = `<tr><td colspan="4" class="muted">${esc(t('empty_inbox'))}</td></tr>`; return; }
    tb.innerHTML = rows.map(ev => `<tr>
      <td class="mono">#${ev.id}</td>
      <td>${esc(ev.content)}</td>
      <td class="mono">${esc(ev.machine)}<div class="muted">${esc(ev.source_client)}</div></td>
      <td><div class="row">
        ${ev.status === 'pending' ? `<button class="btn sm primary" onclick="markInbox([${ev.id}],'consolidated')">${esc(t('btn_mark_done'))}</button><button class="btn sm" onclick="markInbox([${ev.id}],'skipped')">${esc(t('btn_skip'))}</button>` : `<span class="badge">${esc(ev.status)}</span>`}
      </div></td>
    </tr>`).join('');
  } catch (e) { toast(e.message, true); }
}

function openCaptureDialog() { document.getElementById('capContent').value = ''; document.getElementById('capDialog').showModal(); }
async function submitCapture() {
  try {
    await api('/inbox', { method: 'POST', body: { content: document.getElementById('capContent').value.trim(), machine: document.getElementById('capMachine').value.trim() || 'controller' } });
    document.getElementById('capDialog').close();
    toast(t('msg_captured'));
    loadInbox(); refreshStatus();
  } catch (e) { toast(e.message, true); }
}
async function markInbox(ids, status) {
  try { await api('/inbox/mark', { method: 'POST', body: { ids, status } }); loadInbox(); refreshStatus(); }
  catch (e) { toast(e.message, true); }
}
function copyConsolidatePrompt() {
  const txt = t('consolidate_prompt_text');
  navigator.clipboard?.writeText(txt).then(() => toast(t('msg_prompt_copied'))).catch(() => prompt(t('btn_copy_prompt'), txt));
}

/* ===== TAB 4: GOVERNANCE & 3-MODE TIER CONFIG ===== */
async function loadGovernance() {
  try {
    GOV_DATA = await api('/governance');
    const p = GOV_DATA.policy;
    document.getElementById('govAutoInject').checked = !!p.auto_inject;
    document.getElementById('govDefaultIncluded').checked = !!p.default_machine_included;
    document.getElementById('govMcpRole').value = p.mcp_default_role || 'read_write';
    document.getElementById('govConsolidateMode').value = p.auto_consolidate_mode || 'manual';
    document.getElementById('govInterval').value = p.auto_consolidate_interval_min || 60;
    renderMachineTable();
    populateWorkerDropdown();
  } catch (e) { toast(e.message, true); }
}

function renderMachineTable() {
  const p = GOV_DATA.policy;
  const stats = new Map((GOV_DATA.known_machines || []).map(x => [x.machine, x]));
  const live = new Map((GOV_DATA.live_workers || []).map(x => [x.machine, x]));
  const allNames = [...new Set([...Object.keys(p.machines || {}), ...stats.keys(), ...live.keys()])].sort();
  const tb = document.getElementById('machineTbody');
  if (!allNames.length) { tb.innerHTML = `<tr><td colspan="6" class="muted">${esc(t('empty_machines'))}</td></tr>`; return; }
  tb.innerHTML = allNames.map(name => {
    const rule = (p.machines || {})[name] || { included: p.default_machine_included, role: 'read_write', auto_consolidate: true };
    const s = stats.get(name) || { memories_active: 0, inbox_pending: 0 };
    const lw = live.get(name);
    const liveTag = lw ? `<span class="badge ${lw.online ? 'ok' : ''}">${lw.online ? 'online' : 'offline'}</span>` : '';
    const sid = encodeURIComponent(name);
    return `<tr>
      <td class="mono">${esc(name)} ${liveTag}</td>
      <td class="mono">${s.memories_active} active / ${s.inbox_pending} inbox</td>
      <td><select id="m_inc_${sid}"><option value="1" ${rule.included?'selected':''}>${esc(t('opt_inc'))}</option><option value="0" ${!rule.included?'selected':''}>${esc(t('opt_exc'))}</option></select></td>
      <td><select id="m_role_${sid}">
        <option value="read_write" ${rule.role==='read_write'?'selected':''}>${esc(t('opt_rw'))}</option>
        <option value="read_capture" ${rule.role==='read_capture'?'selected':''}>${esc(t('opt_rc'))}</option>
        <option value="read_only" ${rule.role==='read_only'?'selected':''}>${esc(t('opt_ro'))}</option>
      </select></td>
      <td><input type="checkbox" id="m_auto_${sid}" ${rule.auto_consolidate?'checked':''} /></td>
      <td><button class="btn sm primary" onclick="saveMachineRule('${esc(name)}')">${esc(t('btn_apply'))}</button></td>
    </tr>`;
  }).join('');
}

async function saveGlobalGov() {
  try {
    await api('/governance', {
      method: 'POST',
      body: {
        auto_inject: document.getElementById('govAutoInject').checked,
        default_machine_included: document.getElementById('govDefaultIncluded').checked,
        mcp_default_role: document.getElementById('govMcpRole').value,
        auto_consolidate_mode: document.getElementById('govConsolidateMode').value,
        auto_consolidate_interval_min: parseInt(document.getElementById('govInterval').value || '60', 10),
      }
    });
    toast(t('msg_gov_saved'));
    loadGovernance(); refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function saveMachineRule(name) {
  const sid = encodeURIComponent(name);
  const included = document.getElementById('m_inc_' + sid).value === '1';
  const role = document.getElementById('m_role_' + sid).value;
  const auto_consolidate = document.getElementById('m_auto_' + sid).checked;
  try {
    await api('/governance', { method: 'POST', body: { machines: { [name]: { included, role, auto_consolidate } } } });
    toast(t('msg_machine_saved', { name }));
    loadGovernance(); refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function addMachineRow() {
  const name = document.getElementById('newMachineName').value.trim();
  if (!name) return;
  try {
    await api('/governance', { method: 'POST', body: { machines: { [name]: { included: true, role: 'read_write', auto_consolidate: true } } } });
    document.getElementById('newMachineName').value = '';
    loadGovernance();
  } catch (e) { toast(e.message, true); }
}

function populateWorkerDropdown() {
  const sel = document.getElementById('wWorker');
  if (!sel) return;
  const saved = (EMBED_CFG && EMBED_CFG.worker_profile && EMBED_CFG.worker_profile.worker) || sel.value || '';
  const live = (GOV_DATA && GOV_DATA.live_workers) || [];
  const opts = live.map(w => `<option value="${esc(w.machine)}">${esc(w.machine)} (${w.online ? 'online' : 'offline'})</option>`);
  opts.push(`<option value="controller">${esc(t('worker_ctrl_opt'))}</option>`);
  sel.innerHTML = opts.join('');
  if (saved && [...sel.options].some(o => o.value === saved)) sel.value = saved;
  updateWorkerCliPreview();
}

function populateFingerprintDropdown(vecs) {
  const sel = document.getElementById('embFpSelect');
  if (!sel) return;
  const prev = sel.value;
  const opts = [];
  if (vecs.length) {
    for (const v of vecs) {
      opts.push(`<option value="${esc(v.fingerprint)}">fp=${esc(v.fingerprint.slice(0,12))}… · dim=${v.dim} · ${v.count} vecs</option>`);
    }
  } else {
    opts.push(`<option value="">${esc(t('fp_none_opt'))}</option>`);
  }
  opts.push(`<option value="__custom__">${esc(t('fp_custom_opt'))}</option>`);
  sel.innerHTML = opts.join('');
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  onFpSelectChange();
}

function onFpSelectChange() {
  const sel = document.getElementById('embFpSelect');
  const val = sel ? sel.value : '';
  document.getElementById('customShaWrap').style.display = (val === '__custom__') ? 'block' : 'none';
  const wp = (EMBED_CFG && EMBED_CFG.worker_profile) || {};
  if (val && val !== '__custom__' && val === wp.last_fingerprint) {
    if (wp.last_model_name) document.getElementById('embLocalModel').value = wp.last_model_name;
    if (wp.last_model_sha256) document.getElementById('embLocalSha').value = wp.last_model_sha256;
  }
}

function selectTierCard(tier) {
  SELECTED_TIER = tier;
  ['tier0', 'tier1a', 'tier1b'].forEach(k => {
    document.getElementById('card-' + k).classList.toggle('selected', k === tier);
    document.getElementById('pane-' + k).classList.toggle('active', k === tier);
  });
  updateWorkerCliPreview();
}

function selectQueryMode(mode) {
  T1B_QMODE = mode;
  document.getElementById('qmodeOffline').checked = (mode === 'offline');
  document.getElementById('qmodeLocal').checked = (mode === 'local');
  document.getElementById('qmode-offline-card').classList.toggle('selected', mode === 'offline');
  document.getElementById('qmode-local-card').classList.toggle('selected', mode === 'local');
  document.getElementById('t1bLocalFields').style.display = (mode === 'local') ? 'block' : 'none';
  document.getElementById('btnTestLocal').style.display = (mode === 'local') ? 'inline-block' : 'none';
}

function updateWorkerCliPreview() {
  const box = document.getElementById('workerCliBox');
  if (!box) return;
  const w = (document.getElementById('wWorker') && document.getElementById('wWorker').value) || 'worker';
  const bin = (document.getElementById('wLlamaPath') && document.getElementById('wLlamaPath').value.trim()) || '/path/to/llama-server';
  const mdl = (document.getElementById('wModelPath') && document.getElementById('wModelPath').value.trim()) || '/path/to/embedding-model.gguf';
  const origin = location.origin;
  const isWin = /win/i.test(w) || /^[a-zA-Z]:\\/.test(bin) || /\.exe$/i.test(bin);
  if (w === 'controller') {
    box.textContent = `python3 -B -m lsm_memory.vector_sync \\\n  --daemon-origin http://127.0.0.1:8775 \\\n  --daemon-token-file /var/lib/lsm-memory/daemon.token \\\n  --llama-server "${bin}" \\\n  --model "${mdl}"`;
  } else if (isWin) {
    box.textContent = `$env:LOCAL_SHELL_MCP_OAUTH_ADMIN_PIN="<LSM_ADMIN_PIN>"\npython -B -m lsm_memory.vector_sync \`\n  --lsm-url "${origin}" \`\n  --llama-server "${bin}" \`\n  --model "${mdl}"`;
  } else {
    box.textContent = `export LOCAL_SHELL_MCP_OAUTH_ADMIN_PIN="<LSM_ADMIN_PIN>"\npython3 -B -m lsm_memory.vector_sync \\\n  --lsm-url "${origin}" \\\n  --llama-server "${bin}" \\\n  --model "${mdl}"`;
  }
}

function copyWorkerCli() {
  const txt = document.getElementById('workerCliBox').textContent;
  navigator.clipboard?.writeText(txt).then(() => toast(t('msg_cli_copied'))).catch(() => prompt(t('btn_copy_cli'), txt));
}

function renderEmbedPanel() {
  const c = EMBED_CFG;
  if (!c) return;
  const badge = document.getElementById('embStatusText');
  const activeTier = c.enabled ? (c.tier_mode === 'tier1b' ? 'Tier 1B · Local RRF' : 'Tier 1A · Cloud RRF') : 'Tier 0 · Zero-Model';
  badge.textContent = activeTier + (c.has_api_key ? ' · API Key ✓' : '');
  badge.className = 'badge ' + (c.enabled ? 'warn' : 'ok');

  if (c.tier_mode === 'tier1a') {
    document.getElementById('embUrl').value = c.url || '';
    document.getElementById('embModel').value = c.model || '';
  } else if (c.tier_mode === 'tier1b' && c.url) {
    document.getElementById('embLocalUrl').value = c.url;
    document.getElementById('embLocalModel').value = c.model || '';
    document.getElementById('embLocalSha').value = c.model_sha256 || '';
  }
  document.getElementById('embKey').value = '';
  const wp = c.worker_profile || {};
  document.getElementById('wLlamaPath').value = wp.llama_server || '';
  document.getElementById('wModelPath').value = wp.model_path || '';
  populateWorkerDropdown();
  selectQueryMode(c.enabled && c.tier_mode === 'tier1b' ? 'local' : 'offline');
  selectTierCard(c.tier_mode || 'tier0');
}

async function loadEmbedConfig() {
  try {
    EMBED_CFG = await api('/config/embedding');
    renderEmbedPanel();
  } catch (e) { toast(e.message, true); }
}

async function activateTier0() {
  try {
    await api('/config/embedding', { method: 'POST', body: { enabled: false, tier_mode: 'tier0', url: '', model: '', clear_api_key: true } });
    toast(t('msg_tier0_on'));
    await loadEmbedConfig(); await refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function testTier1A() {
  try {
    const body = { url: document.getElementById('embUrl').value.trim(), model: document.getElementById('embModel').value.trim(), model_sha256: 'api' };
    const k = document.getElementById('embKey').value.trim();
    if (k) body.api_key = k;
    const r = await api('/config/embedding/test', { method: 'POST', body });
    toast(`OK · dim=${r.dim} · ${r.latency_ms} ms · fp=${r.fingerprint.slice(0,12)}…`);
  } catch (e) { toast(e.message, true); }
}

async function saveTier1A() {
  try {
    const body = {
      enabled: true, tier_mode: 'tier1a',
      url: document.getElementById('embUrl').value.trim(),
      model: document.getElementById('embModel').value.trim(),
      model_sha256: 'api',
    };
    const k = document.getElementById('embKey').value.trim();
    if (k) body.api_key = k;
    await api('/config/embedding', { method: 'POST', body });
    toast(t('msg_embed_saved'));
    await loadEmbedConfig(); await refreshStatus();
  } catch (e) { toast(e.message, true); }
}

async function runCloudBackfill() {
  const btn = document.getElementById('btnCloudBackfill');
  btn.disabled = true;
  toast(t('msg_working'));
  try {
    const body = { url: document.getElementById('embUrl').value.trim(), model: document.getElementById('embModel').value.trim(), model_sha256: 'api' };
    const k = document.getElementById('embKey').value.trim();
    if (k) body.api_key = k;
    const r = await api('/config/embedding/backfill', { method: 'POST', body });
    toast(`Backfill OK · +${r.accepted}/${r.fetched} · dim=${r.dim} · ${r.elapsed_ms} ms`);
    await refreshStatus();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; }
}

async function saveWorkerProfileOnly() {
  try {
    const wp = {
      worker: document.getElementById('wWorker').value,
      llama_server: document.getElementById('wLlamaPath').value.trim(),
      model_path: document.getElementById('wModelPath').value.trim(),
    };
    await api('/config/embedding', { method: 'POST', body: { worker_profile_only: true, tier_mode: 'tier1b', worker_profile: wp } });
    toast(t('msg_worker_saved'));
    await loadEmbedConfig();
  } catch (e) { toast(e.message, true); }
}

async function runWorkerBackfill() {
  const worker = document.getElementById('wWorker').value;
  const llama_server = document.getElementById('wLlamaPath').value.trim();
  const model_path = document.getElementById('wModelPath').value.trim();
  if (!llama_server || !model_path) return toast(t('msg_need_worker_paths'), true);
  const btn = document.getElementById('btnWorkerRun');
  btn.disabled = true;
  toast(t('msg_working'));
  try {
    const r = await api('/worker-backfill', { method: 'POST', body: { worker, llama_server, model_path } });
    toast(`Worker ${r.worker} OK · +${r.accepted}/${r.fetched} · dim=${r.dim || '-'} · ${r.elapsed_ms} ms`);
    if (r.model_name && !document.getElementById('embLocalModel').value) document.getElementById('embLocalModel').value = r.model_name;
    if (r.model_sha256) document.getElementById('embLocalSha').value = r.model_sha256;
    await loadEmbedConfig(); await refreshStatus();
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; }
}

async function testTier1BLocal() {
  try {
    const body = {
      url: document.getElementById('embLocalUrl').value.trim(),
      model: document.getElementById('embLocalModel').value.trim() || 'default',
      model_sha256: document.getElementById('embLocalSha').value.trim() || 'api',
    };
    const r = await api('/config/embedding/test', { method: 'POST', body });
    toast(`OK · dim=${r.dim} · ${r.latency_ms} ms · fp=${r.fingerprint.slice(0,12)}…`);
  } catch (e) { toast(e.message, true); }
}

async function saveTier1BQueryMode() {
  try {
    const wp = {
      worker: document.getElementById('wWorker').value,
      llama_server: document.getElementById('wLlamaPath').value.trim(),
      model_path: document.getElementById('wModelPath').value.trim(),
    };
    if (T1B_QMODE === 'offline') {
      await api('/config/embedding', { method: 'POST', body: { enabled: false, tier_mode: 'tier1b', url: '', model: '', worker_profile: wp } });
      toast(t('msg_tier0_on'));
    } else {
      await api('/config/embedding', {
        method: 'POST',
        body: {
          enabled: true, tier_mode: 'tier1b',
          url: document.getElementById('embLocalUrl').value.trim(),
          model: document.getElementById('embLocalModel').value.trim() || 'default',
          model_sha256: document.getElementById('embLocalSha').value.trim() || 'api',
          worker_profile: wp,
        }
      });
      toast(t('msg_embed_saved'));
    }
    await loadEmbedConfig(); await refreshStatus();
  } catch (e) { toast(e.message, true); }
}

applyI18n();
refreshStatus(); loadMemories(); checkLegacy();
</script>
</body>
</html>
"""
