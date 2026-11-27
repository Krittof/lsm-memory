"""Pure-stdlib OpenAI-compatible embedding provider & on-demand llama-server runner.

Supports:
- Any OpenAI-compatible `/v1/embeddings` API (local llama-server, Ollama, or remote HTTPS
  cloud APIs for 1C1G VPS deployments with zero local model RAM).
- `ManagedLlamaEmbedServer`: context manager that launches a temporary loopback
  `llama-server --embeddings` process (e.g., on a Windows/Linux GPU workstation), computes
  embeddings, and shuts down cleanly so idle VRAM returns to 0 MB.
- Deterministic cross-platform logical fingerprints (`sha256(model_name + model_sha256 + dim)`)
  decoupled from platform binary hashes so vectors computed on Windows GPU work unchanged
  on Linux ARM64 / x86_64 controllers.
"""

from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .retrieval import logical_embedding_fingerprint
from .store import encode_vector


def compute_file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _validate_endpoint_url(url: str) -> str:
    clean = (url or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(clean)
    if parsed.scheme == "http":
        if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Plain HTTP embedding endpoint must be loopback; use HTTPS for remote APIs")
    elif parsed.scheme != "https":
        raise ValueError("Embedding endpoint must use http://127.0.0.1 or https://")
    if not clean.endswith("/embeddings"):
        if clean.endswith("/v1"):
            clean += "/embeddings"
        else:
            clean += "/v1/embeddings"
    return clean


class OpenAICompatibleEmbeddingProvider:
    """Pure-stdlib EmbeddingProvider for AdaptiveRetriever & vector_sync."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        dim: int = 0,
        model_sha256: str = "",
        api_key: str = "",
        timeout_s: float = 30.0,
        query_prefix: str = "",
        doc_prefix: str = "",
        use_system_proxy: bool = False,
    ) -> None:
        self.endpoint_url = _validate_endpoint_url(base_url)
        self.model = str(model or "").strip()
        if not self.model:
            raise ValueError("model name is required")
        self.api_key = str(api_key or "").strip()
        self.timeout_s = max(0.5, float(timeout_s))
        self.query_prefix = query_prefix or ""
        self.doc_prefix = doc_prefix or ""
        self.model_sha256 = (model_sha256 or "api").strip()
        self._dim = int(dim) if dim and int(dim) > 0 else 0
        self._fingerprint = (
            logical_embedding_fingerprint(model_name=self.model, model_sha256=self.model_sha256, dim=self._dim)
            if self._dim > 0
            else ""
        )
        handlers: list[Any] = []
        if not use_system_proxy:
            handlers.append(urllib.request.ProxyHandler({}))
        self._opener = urllib.request.build_opener(*handlers)

    @property
    def dim(self) -> int:
        if self._dim <= 0:
            self.probe_dimension()
        return self._dim

    @property
    def fingerprint(self) -> str:
        if not self._fingerprint:
            self.probe_dimension()
        return self._fingerprint

    def probe_dimension(self) -> int:
        vec = self.embed_batch(["probe"])[0]
        self._dim = len(vec)
        self._fingerprint = logical_embedding_fingerprint(
            model_name=self.model,
            model_sha256=self.model_sha256,
            dim=self._dim,
        )
        return self._dim

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[array]:
        if not texts:
            return []
        prefix = self.query_prefix if is_query else self.doc_prefix
        prepared = [f"{prefix}{t}" if prefix else str(t) for t in texts]
        payload = json.dumps(
            {"model": self.model, "input": prepared},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.endpoint_url, data=payload, headers=headers, method="POST")
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(prepared):
            raise RuntimeError("Invalid /v1/embeddings response shape")
        # OpenAI format guarantees index field; sort defensively.
        data_sorted = sorted(data, key=lambda item: int(item.get("index", 0)))
        vectors: list[array] = []
        for entry in data_sorted:
            raw_vec = entry.get("embedding")
            dim, _, arr = encode_vector(raw_vec)
            if self._dim <= 0:
                self._dim = dim
                self._fingerprint = logical_embedding_fingerprint(
                    model_name=self.model,
                    model_sha256=self.model_sha256,
                    dim=self._dim,
                )
            elif dim != self._dim:
                raise RuntimeError(f"Embedding dimension mismatch: expected {self._dim}, got {dim}")
            vectors.append(arr)
        return vectors

    def embed_query(self, text: str) -> array:
        return self.embed_batch([text], is_query=True)[0]


class ManagedLlamaEmbedServer:
    """On-demand loopback `llama-server --embeddings` launcher (0 MB VRAM when idle)."""

    def __init__(
        self,
        server_bin: str | Path,
        model_path: str | Path,
        *,
        model_name: str = "",
        gpu_layers: int = 99,
        ctx_size: int = 2048,
        batch_size: int = 512,
        pooling: str = "last",
        startup_timeout_s: float = 45.0,
        request_timeout_s: float = 30.0,
        query_prefix: str = "",
    ) -> None:
        self.server_bin = Path(server_bin)
        self.model_path = Path(model_path)
        if not self.server_bin.is_file():
            raise FileNotFoundError(f"llama-server binary not found: {self.server_bin}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {self.model_path}")
        self.model_name = model_name.strip() or self.model_path.name
        self.gpu_layers = int(gpu_layers)
        self.ctx_size = int(ctx_size)
        self.batch_size = int(batch_size)
        self.pooling = pooling
        self.startup_timeout_s = max(5.0, float(startup_timeout_s))
        self.request_timeout_s = max(2.0, float(request_timeout_s))
        self.query_prefix = query_prefix
        self._proc: subprocess.Popen[bytes] | None = None
        self.port: int = 0
        self.model_sha256: str = ""
        self.provider: OpenAICompatibleEmbeddingProvider | None = None

    def __enter__(self) -> OpenAICompatibleEmbeddingProvider:
        self.model_sha256 = compute_file_sha256(self.model_path)
        self.port = _find_free_loopback_port()
        cmd = [
            str(self.server_bin),
            "--model", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--embeddings",
            "--pooling", self.pooling,
            "--ctx-size", str(self.ctx_size),
            "--batch-size", str(self.batch_size),
            "--ubatch-size", str(self.batch_size),
            "-ngl", str(self.gpu_layers),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        health_url = f"http://127.0.0.1:{self.port}/health"
        deadline = time.monotonic() + self.startup_timeout_s
        ready = False
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                err_tail = (self._proc.stderr.read() if self._proc.stderr else b"")[-500:].decode("utf-8", errors="replace")
                raise RuntimeError(f"llama-server exited early ({self._proc.returncode}): {err_tail}")
            try:
                with opener.open(health_url, timeout=1.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.25)
        if not ready:
            self.close()
            raise TimeoutError(f"llama-server failed to become healthy within {self.startup_timeout_s}s")

        self.provider = OpenAICompatibleEmbeddingProvider(
            f"http://127.0.0.1:{self.port}/v1/embeddings",
            model=self.model_name,
            model_sha256=self.model_sha256,
            timeout_s=self.request_timeout_s,
            query_prefix=self.query_prefix,
        )
        self.provider.probe_dimension()
        return self.provider

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=3.0)
        finally:
            if proc.stderr:
                try:
                    proc.stderr.close()
                except Exception:
                    pass

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def provider_from_env() -> OpenAICompatibleEmbeddingProvider | None:
    """Optional daemon/controller embedding provider from env (None = Tier 0 1C1G default)."""
    url = os.environ.get("LSM_MEMORY_EMBED_URL") or os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_EMBED_URL") or ""
    url = url.strip()
    if not url:
        return None
    model = (
        os.environ.get("LSM_MEMORY_EMBED_MODEL")
        or os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_EMBED_MODEL")
        or "default"
    ).strip()
    dim_raw = (
        os.environ.get("LSM_MEMORY_EMBED_DIM")
        or os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_EMBED_DIM")
        or "0"
    ).strip()
    sha = (
        os.environ.get("LSM_MEMORY_EMBED_MODEL_SHA256")
        or os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_EMBED_MODEL_SHA256")
        or "api"
    ).strip()
    api_key = (
        os.environ.get("LSM_MEMORY_EMBED_API_KEY")
        or os.environ.get("LOCAL_SHELL_MCP_MEMORY_VAULT_EMBED_API_KEY")
        or ""
    ).strip()
    timeout_raw = os.environ.get("LSM_MEMORY_EMBED_TIMEOUT_S") or "0.65"
    try:
        dim = int(dim_raw)
    except ValueError:
        dim = 0
    try:
        timeout_s = max(0.1, min(float(timeout_raw), 10.0))
    except ValueError:
        timeout_s = 0.65
    return OpenAICompatibleEmbeddingProvider(
        url,
        model=model,
        dim=dim,
        model_sha256=sha,
        api_key=api_key,
        timeout_s=timeout_s,
    )
