"""Distributed vector backfill worker (workstation GPU or OpenAI API -> Vault).

Pure Python 3.11+ stdlib with three pluggable transports:
1. `DirectStoreTransport(db_path)` — embedded SQLite vault
2. `DaemonHttpTransport(origin, token)` — local loopback lsm-memory daemon
3. `LsmMcpOAuthTransport(base_url, admin_pin | access_token)` — remote HTTPS `/mcp`
   OAuth 2.1 PKCE via `lsm_memory:memory_vectors_missing` / `memory_vectors_upsert`
   (zero SSH keys, zero extra open ports on the VPS!).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from .embedder import ManagedLlamaEmbedServer, OpenAICompatibleEmbeddingProvider
from .store import MemoryStore


class VectorSyncTransport(Protocol):
    def list_missing(self, fingerprint: str, *, scope: str | None = None, limit: int = 256) -> list[dict[str, Any]]: ...
    def upsert_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]: ...


class DirectStoreTransport:
    def __init__(self, db_or_store: MemoryStore | str | Path) -> None:
        self.store = db_or_store if isinstance(db_or_store, MemoryStore) else MemoryStore(db_or_store)

    def list_missing(self, fingerprint: str, *, scope: str | None = None, limit: int = 256) -> list[dict[str, Any]]:
        return self.store.list_missing_vectors(fingerprint, scope=scope, limit=limit)

    def upsert_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.store.save_vectors_batch(items)


class DaemonHttpTransport:
    def __init__(self, origin: str, token: str, *, timeout_s: float = 30.0) -> None:
        parsed = urllib.parse.urlsplit(origin)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("DaemonHttpTransport requires loopback http://127.0.0.1:<port>")
        self.origin = origin.rstrip("/")
        self.token = token.strip()
        self.timeout_s = timeout_s
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _req(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.origin + path, data=data, headers=headers, method=method)
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("message") or payload.get("error") or "daemon error"))
        return payload["data"]

    def list_missing(self, fingerprint: str, *, scope: str | None = None, limit: int = 256) -> list[dict[str, Any]]:
        qs: dict[str, str] = {"fingerprint": fingerprint, "limit": str(limit)}
        if scope is not None:
            qs["scope"] = scope
        return self._req("GET", "/v1/vectors/missing?" + urllib.parse.urlencode(qs))["items"]

    def upsert_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self._req("POST", "/v1/vectors/batch-upsert", {"items": items})


class _CaptureRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LsmMcpOAuthTransport:
    """Talks to any remote LSM Controller over HTTPS `/mcp` using OAuth 2.1 PKCE (pure stdlib)."""

    def __init__(
        self,
        base_url: str,
        *,
        admin_pin: str = "",
        access_token: str = "",
        server_name: str = "lsm_memory",
        proxy_url: str = "",
        timeout_s: float = 45.0,
    ) -> None:
        self.base_url = base_url.strip().rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme not in ("https", "http"):
            raise ValueError("base_url must be https:// (or loopback http://)")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Non-loopback LSM URL must use https://")
        self.admin_pin = admin_pin.strip()
        self.access_token = access_token.strip()
        self.server_name = server_name.strip() or "lsm_memory"
        self.timeout_s = timeout_s
        handlers: list[Any] = [_CaptureRedirect()]
        if proxy_url:
            handlers.append(urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url}))
        self._opener = urllib.request.build_opener(*handlers)
        self._session_id: str | None = None
        self._rpc_id: int = 0

    def _raw(
        self,
        method: str,
        url: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], str]:
        last_err: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
            try:
                with self._opener.open(req, timeout=self.timeout_s) as resp:
                    hdrs = {k.lower(): v for k, v in resp.headers.items()}
                    return resp.status, hdrs, resp.read().decode("utf-8", errors="replace")
            except urllib.error.HTTPError as exc:
                hdrs = {k.lower(): v for k, v in exc.headers.items()}
                if exc.code in (301, 302, 303, 307, 308):
                    return exc.code, hdrs, ""
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {exc.code} on {url}: {body[:240]}") from None
            except (urllib.error.URLError, ConnectionResetError, TimeoutError, OSError) as exc:
                last_err = exc
                if attempt < 2:
                    time.sleep(0.35 * (attempt + 1))
                    continue
                raise
        assert last_err is not None
        raise last_err

    def _authenticate(self) -> None:
        if not self.access_token:
            if not self.admin_pin:
                raise ValueError("Either access_token or admin_pin is required for LsmMcpOAuthTransport")
            _, _, meta_raw = self._raw("GET", f"{self.base_url}/.well-known/oauth-authorization-server")
            meta = json.loads(meta_raw)
            _, _, pr_raw = self._raw("GET", f"{self.base_url}/.well-known/oauth-protected-resource")
            resource = json.loads(pr_raw).get("resource", self.base_url)

            reg_body = json.dumps({
                "client_name": "lsm-memory-vector-sync",
                "redirect_uris": ["http://127.0.0.1:49155/callback"],
                "grant_types": ["authorization_code"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            }).encode("utf-8")
            _, _, reg_raw = self._raw(
                "POST",
                meta["registration_endpoint"],
                data=reg_body,
                headers={"Content-Type": "application/json"},
            )
            client_id = json.loads(reg_raw)["client_id"]
            verifier = secrets.token_urlsafe(32)
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            state = secrets.token_urlsafe(16)
            auth_qs = urllib.parse.urlencode({
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://127.0.0.1:49155/callback",
                "scope": " ".join(meta.get("scopes_supported", ["remote:use"])),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": resource,
            })
            _, _, auth_html = self._raw("GET", f"{meta['authorization_endpoint']}?{auth_qs}")
            form = {
                m.group(1): m.group(2)
                for m in re.finditer(r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', auth_html)
            }
            form["pin"] = self.admin_pin
            form["decision"] = "approve"
            status, hdrs, _ = self._raw(
                "POST",
                meta["authorization_endpoint"],
                data=urllib.parse.urlencode(form).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            loc = hdrs.get("location", "")
            if status not in (302, 303, 307) or "code=" not in loc:
                raise RuntimeError(f"OAuth approval failed (HTTP {status})")
            code = urllib.parse.parse_qs(urllib.parse.urlsplit(loc).query)["code"][0]
            tok_form = urllib.parse.urlencode({
                "grant_type": "authorization_code",
                "code": code,
                "client_id": client_id,
                "redirect_uri": "http://127.0.0.1:49155/callback",
                "code_verifier": verifier,
                "resource": resource,
            }).encode("utf-8")
            _, _, tok_raw = self._raw(
                "POST",
                meta["token_endpoint"],
                data=tok_form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            self.access_token = json.loads(tok_raw)["access_token"]

        if self._session_id is None:
            headers = self._mcp_headers()
            init_body = json.dumps({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "lsm-memory-vector-sync", "version": "0.3.0"},
                },
            }).encode("utf-8")
            _, hdrs, _ = self._raw("POST", f"{self.base_url}/mcp", data=init_body, headers=headers)
            self._session_id = hdrs.get("mcp-session-id") or ""
            notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}).encode("utf-8")
            self._raw("POST", f"{self.base_url}/mcp", data=notif, headers=self._mcp_headers())

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _mcp_headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        if self._session_id:
            h["mcp-session-id"] = self._session_id
        return h

    def call_nested(self, tool_short: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._authenticate()
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": "mcp_tool_call",
                "arguments": {
                    "name": f"{self.server_name}:{tool_short}",
                    "arguments": arguments,
                },
            },
        }, ensure_ascii=False).encode("utf-8")
        _, hdrs, text = self._raw("POST", f"{self.base_url}/mcp", data=body, headers=self._mcp_headers())
        parsed = None
        if "text/event-stream" in hdrs.get("content-type", ""):
            for ln in text.splitlines():
                if ln.startswith("data:") and ln[5:].strip():
                    parsed = json.loads(ln[5:].strip())
                    break
        else:
            parsed = json.loads(text)
        if not isinstance(parsed, dict) or "error" in parsed:
            raise RuntimeError(f"MCP error calling {tool_short}: {parsed}")
        sc = parsed["result"].get("structuredContent", {})
        if not sc.get("ok"):
            raise RuntimeError(f"Nested MCP {tool_short} failed: {sc}")
        inner = sc.get("data", {}).get("result", {})
        if isinstance(inner, dict) and "structuredContent" in inner:
            return inner["structuredContent"]
        return inner

    def list_missing(self, fingerprint: str, *, scope: str | None = None, limit: int = 256) -> list[dict[str, Any]]:
        args: dict[str, Any] = {"fingerprint": fingerprint, "limit": limit}
        if scope is not None:
            args["scope"] = scope
        return self.call_nested("memory_vectors_missing", args)["items"]

    def upsert_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return self.call_nested("memory_vectors_upsert", {"items": items})


def format_memory_for_embedding(item: dict[str, Any]) -> str:
    """Canonical text combining content + topic + offline-expanded aliases."""
    parts = [str(item.get("content") or "").strip()]
    topic = str(item.get("topic") or "").strip()
    terms = item.get("search_terms") or []
    meta_bits = []
    if topic:
        meta_bits.append(f"topic: {topic}")
    if isinstance(terms, list) and terms:
        clean_terms = [str(t).strip() for t in terms if str(t).strip()]
        if clean_terms:
            meta_bits.append("aliases: " + ", ".join(clean_terms))
    if meta_bits:
        parts.append("[" + " | ".join(meta_bits) + "]")
    return "\n".join(parts)


def sync_missing_vectors(
    transport: VectorSyncTransport,
    provider: OpenAICompatibleEmbeddingProvider,
    *,
    scope: str | None = None,
    batch_size: int = 32,
    max_items: int = 512,
) -> dict[str, Any]:
    """Pull missing memory texts, compute vectors in bounded batches, and push back."""
    t0 = time.perf_counter()
    fp = provider.fingerprint
    dim = provider.dim
    missing = transport.list_missing(fp, scope=scope, limit=max_items)
    total_fetched = len(missing)
    total_embedded = 0
    total_accepted = 0
    total_skipped = 0
    bsz = max(1, min(int(batch_size), 128))

    for start in range(0, total_fetched, bsz):
        chunk = missing[start : start + bsz]
        texts = [format_memory_for_embedding(it) for it in chunk]
        vectors = provider.embed_batch(texts, is_query=False)
        total_embedded += len(vectors)
        upload = [
            {
                "memory_id": int(it["memory_id"]),
                "content_version": int(it["content_version"]),
                "fingerprint": fp,
                "vector": [round(float(x), 6) for x in vec],
            }
            for it, vec in zip(chunk, vectors, strict=True)
        ]
        res = transport.upsert_batch(upload)
        total_accepted += int(res.get("accepted", 0))
        total_skipped += int(res.get("skipped_stale", 0))

    return {
        "fingerprint": fp,
        "model": provider.model,
        "dim": dim,
        "fetched": total_fetched,
        "embedded": total_embedded,
        "accepted": total_accepted,
        "skipped_stale": total_skipped,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }


WORKER_RESULT_PREFIX = "LSM_MEMORY_WORKER_VEC_RESULT:"


def build_worker_embed_script(
    llama_server: str,
    model_path: str,
    items: list[dict[str, Any]],
    *,
    batch_size: int = 32,
    gpu_layers: int = 99,
    pooling: str = "last",
) -> str:
    """Build a self-contained stdlib Python script to run on an online LSM Worker via `run_python`.

    Zero secrets and zero external packages are sent to the Worker: it hashes the local GGUF,
    launches temporary loopback `llama-server --embeddings`, computes normalized float32 vectors,
    terminates `llama-server` in `finally:` (0 MB idle VRAM), and prints a single JSON result line.
    """
    payload = {
        "llama_server": str(llama_server),
        "model_path": str(model_path),
        "batch_size": max(1, min(int(batch_size), 64)),
        "gpu_layers": int(gpu_layers),
        "pooling": str(pooling or "last"),
        "items": [
            {
                "memory_id": int(it["memory_id"]),
                "content_version": int(it["content_version"]),
                "text": format_memory_for_embedding(it),
            }
            for it in items
        ],
    }
    payload_b64 = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
    return f"""import base64, hashlib, json, math, os, socket, struct, subprocess, sys, time, urllib.request
from pathlib import Path

CFG = json.loads(base64.b64decode({payload_b64!r}).decode("utf-8"))
srv_bin = Path(CFG["llama_server"]).expanduser()
mdl_path = Path(CFG["model_path"]).expanduser()
if not srv_bin.is_file():
    raise FileNotFoundError(f"llama-server not found: {{srv_bin}}")
if not mdl_path.is_file():
    raise FileNotFoundError(f"GGUF model not found: {{mdl_path}}")

h = hashlib.sha256()
with mdl_path.open("rb") as fh:
    while True:
        chunk = fh.read(1024 * 1024)
        if not chunk:
            break
        h.update(chunk)
model_sha256 = h.hexdigest()
model_name = mdl_path.name

def find_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

port = find_port()
cmd = [
    str(srv_bin), "--model", str(mdl_path), "--host", "127.0.0.1", "--port", str(port),
    "--embeddings", "--pooling", CFG["pooling"], "--ctx-size", "2048",
    "--batch-size", "512", "--ubatch-size", "512", "-ngl", str(CFG["gpu_layers"]),
]
proc = subprocess.Popen(
    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({{}}))
t0 = time.perf_counter()
try:
    deadline = time.monotonic() + 60.0
    ready = False
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = (proc.stderr.read() if proc.stderr else b"")[-400:].decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server exited ({{proc.returncode}}): {{tail}}")
        try:
            with opener.open(f"http://127.0.0.1:{{port}}/health", timeout=1.0) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.25)
    if not ready:
        raise TimeoutError("llama-server health timeout")

    def embed_texts(texts):
        body = json.dumps({{"model": model_name, "input": texts}}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{{port}}/v1/embeddings",
            data=body, headers={{"Content-Type": "application/json"}}, method="POST",
        )
        with opener.open(req, timeout=45.0) as resp:
            data = sorted(json.loads(resp.read().decode("utf-8"))["data"], key=lambda x: int(x.get("index", 0)))
        out = []
        for entry in data:
            vals = [float(x) for x in entry["embedding"]]
            norm = math.sqrt(sum(v * v for v in vals)) or 1.0
            out.append([v / norm for v in vals])
        return out

    probe = embed_texts(["probe"])[0]
    dim = len(probe)
    fp_payload = json.dumps({{
        "schema": "lsm-memory-embedding-v1",
        "model": model_name,
        "model_sha256": model_sha256,
        "dim": int(dim),
        "pooling": CFG["pooling"],
        "normalization": "l2",
        "query_prefix": "",
    }}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fp = hashlib.sha256(fp_payload).hexdigest()
    items = CFG["items"]
    bsz = CFG["batch_size"]
    vectors = []
    for i in range(0, len(items), bsz):
        chunk = items[i:i + bsz]
        vecs = embed_texts([it["text"] for it in chunk])
        for it, v in zip(chunk, vecs):
            raw = struct.pack(f"<{{len(v)}}f", *v)
            vectors.append({{
                "memory_id": it["memory_id"],
                "content_version": it["content_version"],
                "vec_b64": base64.b64encode(raw).decode("ascii"),
            }})
    result = {{
        "model_name": model_name,
        "model_sha256": model_sha256,
        "pooling": CFG["pooling"],
        "dim": dim,
        "fingerprint": fp,
        "vectors": vectors,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
    }}
    print({WORKER_RESULT_PREFIX!r} + json.dumps(result, separators=(",", ":")))
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()
"""


def parse_worker_embed_output(stdout_text: str) -> dict[str, Any]:
    """Extract structured JSON result from a remote Worker `run_python` stdout."""
    for line in reversed((stdout_text or "").splitlines()):
        s = line.strip()
        if s.startswith(WORKER_RESULT_PREFIX):
            return json.loads(s[len(WORKER_RESULT_PREFIX):])
    raise RuntimeError(f"Worker did not return a vector result: {(stdout_text or '')[-400:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSMMemory distributed vector backfill worker")
    # Target vault transport
    parser.add_argument("--db", help="Embedded SQLite DB path")
    parser.add_argument("--daemon-origin", help="Loopback daemon origin, e.g. http://127.0.0.1:8775")
    parser.add_argument("--daemon-token-file", help="Loopback daemon token file")
    parser.add_argument("--lsm-url", help="Remote LSM Controller HTTPS URL for /mcp OAuth 2.1 sync")
    parser.add_argument("--lsm-pin-env", default="LOCAL_SHELL_MCP_OAUTH_ADMIN_PIN", help="Env var holding LSM Admin PIN")
    parser.add_argument("--lsm-proxy", default="", help="Optional HTTP proxy for remote LSM URL")
    # Embedding provider
    parser.add_argument("--embed-url", help="Existing OpenAI-compatible /v1/embeddings endpoint")
    parser.add_argument("--embed-model", default="default", help="Model name for /v1/embeddings")
    parser.add_argument("--embed-model-sha256", default="api", help="Logical model SHA-256")
    parser.add_argument("--embed-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llama-server", help="Path to local llama-server executable (on-demand GPU mode)")
    parser.add_argument("--model", help="Path to local .gguf embedding model (used with --llama-server)")
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument("--pooling", default="last")
    # Batch controls
    parser.add_argument("--scope", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=512)
    args = parser.parse_args(argv)

    transport: VectorSyncTransport
    if args.db:
        transport = DirectStoreTransport(args.db)
    elif args.daemon_origin:
        token = Path(args.daemon_token_file).read_text(encoding="utf-8").strip() if args.daemon_token_file else os.environ.get("LSM_MEMORY_TOKEN", "")
        transport = DaemonHttpTransport(args.daemon_origin, token)
    elif args.lsm_url:
        pin = os.environ.get(args.lsm_pin_env, "")
        transport = LsmMcpOAuthTransport(args.lsm_url, admin_pin=pin, proxy_url=args.lsm_proxy)
    else:
        raise SystemExit("Specify one target: --db, --daemon-origin, or --lsm-url")

    if args.llama_server and args.model:
        with ManagedLlamaEmbedServer(
            args.llama_server,
            args.model,
            gpu_layers=args.gpu_layers,
            pooling=args.pooling,
        ) as provider:
            out = sync_missing_vectors(
                transport, provider, scope=args.scope, batch_size=args.batch_size, max_items=args.max_items
            )
    elif args.embed_url:
        provider = OpenAICompatibleEmbeddingProvider(
            args.embed_url,
            model=args.embed_model,
            model_sha256=args.embed_model_sha256,
            api_key=os.environ.get(args.embed_api_key_env, ""),
        )
        out = sync_missing_vectors(
            transport, provider, scope=args.scope, batch_size=args.batch_size, max_items=args.max_items
        )
    else:
        raise SystemExit("Specify either (--llama-server and --model) or --embed-url")

    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
