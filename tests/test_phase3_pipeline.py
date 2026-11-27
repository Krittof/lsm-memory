"""Unit & integration tests for Phase 3 offline consolidation and distributed vector sync."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest

from lsm_memory.consolidator import (
    build_consolidation_pack,
    collect_lsm_turns,
    commit_consolidation_pack,
)
from lsm_memory.daemon import MemoryDaemonServer
from lsm_memory.embedder import OpenAICompatibleEmbeddingProvider
from lsm_memory.mcp_server import HttpBackend, MCPStdioHandler, serve_stdio
from lsm_memory.retrieval import AdaptiveRetriever
from lsm_memory.store import MemoryStore
from lsm_memory.vector_sync import (
    DaemonHttpTransport,
    DirectStoreTransport,
    sync_missing_vectors,
)


def _start_mock_embed_server(*, dim: int = 16, delay_s: float = 0.0) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: ANN001
            return

        def do_POST(self):  # noqa: N802
            if delay_s > 0:
                time.sleep(delay_s)
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            data = []
            for idx, txt in enumerate(inputs):
                vec = [0.01] * dim
                low = str(txt).lower()
                if any(k in low for k in ("keepalive", "甲骨文", "防回收", "oci")):
                    vec[0] = 1.0
                    vec[1] = 0.7
                elif any(k in low for k in ("proxy", "代理", "iepl")):
                    vec[2] = 1.0
                else:
                    vec[3] = 1.0
                data.append({"object": "embedding", "index": idx, "embedding": vec})
            raw = json.dumps({"object": "list", "data": data, "model": body.get("model", "mock")}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1/embeddings"


class TestPhase3Pipeline(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.db_path = self.root / "vault.sqlite3"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _create_mock_lsm_db(self) -> Path:
        lsm_db = self.root / "lsm_controller.sqlite3"
        conn = sqlite3.connect(str(lsm_db))
        try:
            conn.execute("CREATE TABLE conversations(id TEXT PRIMARY KEY, owner TEXT, title TEXT, updated_at INTEGER)")
            conn.execute(
                "CREATE TABLE messages(id TEXT PRIMARY KEY, conversation_id TEXT, role TEXT, text TEXT, content TEXT, created_at INTEGER)"
            )
            conn.execute("INSERT INTO conversations VALUES('c1', 'u1', 'Deploy Notes', 1000)")
            conn.executemany(
                "INSERT INTO messages VALUES(?, 'c1', ?, ?, '', ?)",
                [
                    ("m1", "user", "你好", 1001),  # small talk -> skipped
                    ("m2", "user", "sk-1234567890abcdef1234567890 secret key", 1002),  # secret -> skipped
                    (
                        "m3",
                        "user",
                        '<lsm-memory-recall revision="1" untrusted="true">stale</lsm-memory-recall>\n\nRemember: Tokyo ARM runs Caddy on 443.',
                        1003,
                    ),
                    ("m4", "assistant", "Noted: Tokyo ARM uses Caddy on port 443.", 1004),
                ],
            )
            conn.commit()
        finally:
            conn.close()
        return lsm_db

    def test_readonly_lsm_collector_and_consolidation_commit(self) -> None:
        lsm_db = self._create_mock_lsm_db()
        ev1 = self.store.enqueue_inbox("User prefers pure stdlib zero-dep Python for LSM.", source_client="gemini-web")
        ev2 = self.store.enqueue_inbox("Noise note to skip", source_client="gemini-web")

        pack = build_consolidation_pack(self.store, scope="personal", lsm_db_path=lsm_db)
        self.assertTrue(pack["has_work"])
        self.assertEqual(len(pack["inbox_events"]), 2)
        lsm_turns = pack["lsm_conversations"]["turns"]
        self.assertEqual(len(lsm_turns), 2)
        self.assertNotIn("<lsm-memory-recall", lsm_turns[0]["text"])
        self.assertEqual(pack["lsm_conversations"]["next_cursor_us"], 1004)

        # Commit distilled memories (including one secret candidate that MUST be rejected cleanly)
        commit_res = commit_consolidation_pack(
            self.store,
            {
                "memories": [
                    {
                        "content": "LSMMemory must use pure Python stdlib with zero third-party deps.",
                        "kind": "user_profile",
                        "topic": "architecture",
                        "search_terms": ["纯标准库", "零依赖"],
                    },
                    {
                        "content": "Tokyo ARM runs Caddy TLS reverse proxy on port 443.",
                        "kind": "episode",
                        "topic": "tokyo-arm",
                        "search_terms": ["东京主机", "反向代理"],
                    },
                    {
                        "content": "Leaked api key sk-abcdef1234567890abcdef123456",
                        "kind": "episode",
                    },
                ],
                "relations": [{"source_idx": 0, "target_idx": 1, "relation_type": "related"}],
                "inbox_ids": [ev1["id"]],
                "skip_inbox_ids": [ev2["id"]],
                "cursor_key": pack["lsm_conversations"]["cursor_key"],
                "next_cursor_us": pack["lsm_conversations"]["next_cursor_us"],
            },
        )
        self.assertEqual(len(commit_res["created_ids"]), 2)
        self.assertEqual(len(commit_res["rejected"]), 1)
        self.assertEqual(commit_res["relations_added"], 1)
        self.assertEqual(commit_res["inbox_consolidated"], 1)
        self.assertEqual(commit_res["inbox_skipped"], 1)
        self.assertTrue(commit_res["cursor_saved"])

        # Re-running pack now sees zero pending inbox & zero new LSM turns (incremental cursor!)
        pack2 = build_consolidation_pack(self.store, scope="personal", lsm_db_path=lsm_db)
        self.assertFalse(pack2["has_work"])
        self.assertEqual(pack2["inbox_events"], [])
        self.assertEqual(pack2["lsm_conversations"]["turns"], [])

    def test_vector_sync_unlocks_tier1_hybrid_rrf_and_falls_back_on_timeout(self) -> None:
        self.store.add_memory(
            "Oracle Cloud monthly compliance reminder prevents idle VPS reclamation.",
            kind="episode",
            topic="oci",
            search_terms=["OCI"],
        )
        srv, url = _start_mock_embed_server(dim=16)
        try:
            provider = OpenAICompatibleEmbeddingProvider(
                url, model="mock-qwen3", dim=16, model_sha256="b" * 64, timeout_s=2.0
            )
            sync1 = sync_missing_vectors(DirectStoreTransport(self.db_path), provider)
            self.assertEqual(sync1["fetched"], 1)
            self.assertEqual(sync1["accepted"], 1)
            self.assertEqual(sync1["dim"], 16)

            # Second sync is a clean no-op
            sync2 = sync_missing_vectors(DirectStoreTransport(self.db_path), provider)
            self.assertEqual(sync2["fetched"], 0)
            self.assertEqual(sync2["accepted"], 0)

            # Tier 1 hybrid recall now matches pure Chinese paraphrase with ZERO lexical overlap!
            ret = AdaptiveRetriever(self.store, embedding_provider=provider, deadline_ms=750.0)
            try:
                hit = ret.search("甲骨文机器防回收怎么弄", limit=5)
                self.assertEqual(hit["tier"], "tier1_hybrid_rrf")
                self.assertFalse(hit["degraded"])
                self.assertEqual(len(hit["results"]), 1)
                self.assertEqual(hit["results"][0]["id"], 1)
            finally:
                ret.close()
        finally:
            srv.shutdown()
            srv.server_close()

        # After embedding server shuts down -> fail-closed automatic fallback to Tier 0!
        ret_down = AdaptiveRetriever(self.store, embedding_provider=provider, deadline_ms=400.0)
        try:
            fb = ret_down.search("Oracle Cloud reclamation", limit=5)
            self.assertEqual(fb["tier"], "tier0_lexical_maxsim")
            self.assertTrue(fb["degraded"])
            self.assertEqual(len(fb["results"]), 1)
        finally:
            ret_down.close()

    def test_daemon_and_mcp_phase3_tools_end_to_end(self) -> None:
        token = "phase3_test_token_0123456789"
        ret = AdaptiveRetriever(self.store)
        httpd = MemoryDaemonServer(("127.0.0.1", 0), self.store, ret, token)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        t.start()
        srv_emb, emb_url = _start_mock_embed_server(dim=16)
        try:
            origin = f"http://127.0.0.1:{port}"
            handler = MCPStdioHandler(HttpBackend(origin, token))
            reqs = [
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                {
                    "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": "memory_capture", "arguments": {"content": "Proxy uses IEPL front node."}},
                },
                {
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "memory_consolidate_pull", "arguments": {"scope": "personal"}},
                },
                {
                    "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {
                        "name": "memory_consolidate_commit",
                        "arguments": {
                            "memories": [
                                {"content": "Proxy egress uses IEPL front node.", "search_terms": ["代理", "IEPL"]}
                            ],
                            "inbox_ids": [1],
                        },
                    },
                },
                {
                    "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "memory_vectors_missing", "arguments": {"fingerprint": "fp_test_16"}},
                },
                {
                    "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {
                        "name": "memory_vectors_upsert",
                        "arguments": {
                            "items": [
                                {
                                    "memory_id": 1,
                                    "content_version": 1,
                                    "fingerprint": "fp_test_16",
                                    "vector": [0.1] * 16,
                                }
                            ]
                        },
                    },
                },
            ]
            out = io.StringIO()
            serve_stdio(handler, io.StringIO("\n".join(json.dumps(r) for r in reqs) + "\n"), out)
            res = [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]
            self.assertEqual(len(res[0]["result"]["tools"]), 12)
            self.assertEqual(len(res[2]["result"]["structuredContent"]["inbox_events"]), 1)
            self.assertEqual(res[3]["result"]["structuredContent"]["created_ids"], [1])
            self.assertEqual(len(res[4]["result"]["structuredContent"]["items"]), 1)
            self.assertEqual(res[5]["result"]["structuredContent"]["accepted"], 1)

            # DaemonHttpTransport sync with a fresh logical fingerprint
            provider = OpenAICompatibleEmbeddingProvider(emb_url, model="mock-daemon", dim=16, model_sha256="c" * 64)
            synced = sync_missing_vectors(DaemonHttpTransport(origin, token), provider)
            self.assertEqual(synced["accepted"], 1)
        finally:
            srv_emb.shutdown()
            srv_emb.server_close()
            httpd.shutdown()
            httpd.server_close()
            ret.close()


if __name__ == "__main__":
    unittest.main()
