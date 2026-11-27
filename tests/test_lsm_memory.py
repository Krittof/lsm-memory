"""Unit, integration, and 1C1G memory/latency budget tests for LSMMemory."""

from __future__ import annotations

from array import array
import io
import json
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from lsm_memory.daemon import MemoryDaemonServer, _rss_mb
from lsm_memory.mcp_server import DirectBackend, HttpBackend, MCPStdioHandler, serve_stdio
from lsm_memory.policy import ContentBlockedError, assert_content_safe, decide_auto_write
from lsm_memory.retrieval import AdaptiveRetriever, logical_embedding_fingerprint
from lsm_memory.store import ConflictError, MemoryStore, encode_vector


class _MockEmbeddingProvider:
    def __init__(self, dim: int = 32, *, delay_s: float = 0.0, fail: bool = False):
        self.dim = dim
        self.delay_s = delay_s
        self.fail = fail
        self.fingerprint = logical_embedding_fingerprint(
            model_name="mock-embed", model_sha256="0" * 64, dim=dim
        )

    def embed_query(self, text: str) -> array:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("offline")
        # Deterministic semantic bucket for testing paraphrase recall without lexical overlap
        vals = [0.01] * self.dim
        if "甲骨文" in text or "防回收" in text or "keepalive" in text.lower():
            vals[0] = 1.0
            vals[1] = 0.8
        elif "代理" in text or "gateway" in text.lower():
            vals[2] = 1.0
        else:
            vals[3] = 1.0
        _, _, arr = encode_vector(vals)
        return arr


class TestLSMMemoryCore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "mem.sqlite3"
        self.store = MemoryStore(self.db_path, max_cached_vectors=128)
        self.retriever = AdaptiveRetriever(self.store)

    def tearDown(self) -> None:
        self.retriever.close()
        self._tmp.cleanup()

    def test_crud_cas_suppress_and_tombstone(self) -> None:
        m = self.store.add_memory(
            "Controller ARM 203.0.113.10 runs LSM v6.0.3 controller on port 8765 behind Caddy 443.",
            kind="episode",
            search_terms=["主控ARM", "LSM主控", "Caddy"],
            topic="infra",
            source_client="cli",
            machine="workstation",
        )
        self.assertEqual(m["version"], 1)
        self.assertFalse(m["deduplicated"])
        rev1 = m["global_revision"]

        # Idempotent dedup enriches search_terms without creating a second row
        dup = self.store.add_memory(
            "Controller ARM 203.0.113.10 runs LSM v6.0.3 controller on port 8765 behind Caddy 443.",
            search_terms=["controller.example.invalid"],
        )
        self.assertTrue(dup["deduplicated"])
        self.assertEqual(dup["id"], m["id"])
        self.assertIn("controller.example.invalid", dup["search_terms"])
        self.assertEqual(dup["version"], 2)

        # CAS mismatch rejected
        with self.assertRaises(ConflictError):
            self.store.correct_memory(m["id"], "Updated text", expected_version=1)

        # CAS success
        upd = self.store.correct_memory(m["id"], "Controller ARM 203.0.113.10 runs LSM v6.0.3 + LSMMemory.", expected_version=2)
        self.assertEqual(upd["version"], 3)
        self.assertGreater(upd["global_revision"], rev1)

        # Soft-suppress excludes from recall without deleting
        sup = self.store.set_suppressed(m["id"], True)
        self.assertTrue(sup["suppressed"])
        res_sup = self.retriever.search("主控ARM LSM主控")
        self.assertEqual(len(res_sup["results"]), 0)

        # Restore brings it back immediately
        rst = self.store.set_suppressed(m["id"], False)
        self.assertFalse(rst["suppressed"])
        res_rst = self.retriever.search("主控ARM LSM主控")
        self.assertEqual(len(res_rst["results"]), 1)
        self.assertEqual(res_rst["results"][0]["id"], m["id"])

        # Forget creates tombstone and blocks resurrection
        self.store.forget_memory(m["id"], reason="obsolete")
        with self.assertRaises(ConflictError):
            self.store.add_memory("Controller ARM 203.0.113.10 runs LSM v6.0.3 + LSMMemory.")

    def test_tier0_aliases_maxsim_relations_contradictions(self) -> None:
        m1 = self.store.add_memory(
            "All 4 OCI VPS nodes run oci-keepalive.py v2.2 with 3x1h randomized zlib windows.",
            kind="episode",
            search_terms=["甲骨文", "防回收", "保活脚本", "Always Free"],
            topic="oci-keepalive",
            source_client="agy",
            machine="local",
        )
        m2 = self.store.add_memory(
            "Windows monthly reminder task OCI-Monthly-Keepalive-Reminder fires on the 25th at 20:00.",
            kind="episode",
            topic="oci-keepalive",
            source_client="agy",
            machine="workstation",
        )
        self.store.add_relation(m1["id"], m2["id"], "related", weight=1.0)

        # Pure Chinese alias query matches English memory + pulls 1-hop related neighbor
        res = self.retriever.search("甲骨文防回收怎么配的", limit=5)
        self.assertEqual(res["tier"], "tier0_lexical_maxsim")
        ids = [r["id"] for r in res["results"]]
        self.assertEqual(ids[0], m1["id"])
        self.assertIn(m2["id"], ids)

        # Contradiction detection
        c1 = self.store.add_memory("默认开启 IPv6 代理转发功能用于东京节点")
        c2 = self.store.add_memory("默认关闭 IPv6 代理转发功能用于东京节点")
        cres = self.retriever.search("IPv6 代理转发 东京节点", limit=5)
        by_id = {r["id"]: r for r in cres["results"]}
        self.assertIn(c2["id"], by_id[c1["id"]].get("contradiction_hints", []))

    def test_tier1_hybrid_and_graceful_fallback(self) -> None:
        # Memory written completely in English with NO Chinese keywords
        m = self.store.add_memory(
            "Intermittent workload scheduler maintains minimum utilization across cloud compute shapes.",
            kind="episode",
        )
        provider = _MockEmbeddingProvider(dim=32)
        vec = provider.embed_query("keepalive")
        self.store.save_vectors_batch([
            {"memory_id": m["id"], "content_version": m["version"], "fingerprint": provider.fingerprint, "vector": list(vec)}
        ])
        self.assertEqual(self.store.list_missing_vectors(provider.fingerprint), [])

        # Paraphrase Chinese query hits via Tier 1 Hybrid RRF
        retr = AdaptiveRetriever(self.store, embedding_provider=provider)
        try:
            res = retr.search("甲骨文防回收策略")
            self.assertEqual(res["tier"], "tier1_hybrid_rrf")
            self.assertFalse(res["degraded"])
            self.assertEqual(res["results"][0]["id"], m["id"])
        finally:
            retr.close()

        # Timeout or offline provider gracefully degrades to Tier 0 without crashing
        slow_provider = _MockEmbeddingProvider(dim=32, delay_s=0.35)
        retr_slow = AdaptiveRetriever(self.store, embedding_provider=slow_provider, deadline_ms=80.0)
        try:
            res_fb = retr_slow.search("cloud compute shapes")
            self.assertEqual(res_fb["tier"], "tier0_lexical_maxsim")
            self.assertTrue(res_fb["degraded"])
            self.assertEqual(res_fb["reason"], "embedding_timeout")
            self.assertEqual(res_fb["results"][0]["id"], m["id"])
        finally:
            retr_slow.close()

    def test_policy_secret_blocking_and_auto_write(self) -> None:
        with self.assertRaises(ContentBlockedError):
            assert_content_safe("Use key sk-proj-1234567890abcdef1234567890")
        with self.assertRaises(ContentBlockedError):
            assert_content_safe("token lsmcp_wk_hetHqh1VJuiWdcjStjGLkXr0ulPbByxHdUm7Ch2HsN8")

        d1 = decide_auto_write({
            "auto_write_enabled": True, "global_pause": False, "level": "balanced",
            "kind": "user_profile", "content": "以后默认所有项目放在 D 盘目录",
        })
        self.assertEqual(d1["route"], "auto_commit")

        d2 = decide_auto_write({
            "auto_write_enabled": True, "global_pause": True, "level": "balanced",
            "kind": "user_profile", "content": "以后默认所有项目放在 D 盘目录",
        })
        self.assertEqual(d2["route"], "queue_review")
        self.assertEqual(d2["reason_code"], "global_pause")

    def test_daemon_http_and_mcp_stdio_end_to_end(self) -> None:
        token = "test_bearer_token_0123456789abcdef0123456789"
        httpd = MemoryDaemonServer(("127.0.0.1", 0), self.store, self.retriever, token)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        t.start()
        try:
            origin = f"http://127.0.0.1:{port}"
            # Unauthorized blocked
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(origin + "/v1/status", timeout=5)
            self.assertEqual(ctx.exception.code, 401)

            # MCP stdio backed by HTTP daemon
            handler = MCPStdioHandler(HttpBackend(origin, token))
            requests = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                {
                    "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {
                        "name": "memory_add",
                        "arguments": {
                            "content": "Gemini web connects to Tokyo ARM via /mcp OAuth 2.1.",
                            "search_terms": ["Gemini网页端", "Custom Apps", "OAuth"],
                            "source_client": "gemini-web",
                            "machine": "local",
                        },
                    },
                },
                {
                    "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "memory_search", "arguments": {"query": "Gemini网页端怎么连的"}},
                },
                {
                    "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                    "params": {"name": "memory_suppress", "arguments": {"memory_id": 1, "action": "suppress"}},
                },
                {
                    "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                    "params": {"name": "memory_search", "arguments": {"query": "Gemini网页端怎么连的"}},
                },
                {
                    "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": "memory_capture", "arguments": {"content": "Offline summary note from workstation"}},
                },
                {
                    "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                    "params": {"name": "memory_status", "arguments": {}},
                },
            ]
            inp = io.BytesIO(("\n".join(json.dumps(r) for r in requests) + "\n").encode("utf-8"))
            out = io.StringIO()
            serve_stdio(handler, inp, out)
            responses = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
            self.assertEqual(len(responses), 8)
            self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "lsm-memory")
            self.assertEqual(len(responses[1]["result"]["tools"]), 12)
            self.assertEqual(responses[3]["result"]["structuredContent"]["results"][0]["id"], 1)
            self.assertEqual(responses[5]["result"]["structuredContent"]["results"], [])
            self.assertEqual(responses[7]["result"]["structuredContent"]["counts"]["inbox_pending"], 1)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_1c1g_budget_1000_memories_and_1024d_vectors(self) -> None:
        """Verify 1,000 memories + 1,000 1024-dim vectors stay well within 1C1G RAM & latency limits."""
        rss_before = _rss_mb() or 0.0
        fp = logical_embedding_fingerprint(model_name="qwen3-0.6b", model_sha256="a" * 64, dim=1024)
        base_vec = [math.sin(i * 0.01) for i in range(1024)]
        vec_items = []
        for idx in range(1000):
            row = self.store.add_memory(
                f"Synthetic operational memory #{idx}: host node-{idx % 5} service port {8000 + idx} status active.",
                kind="episode",
                search_terms=[f"node-{idx % 5}", f"tag-{idx % 20}"],
                machine=f"node-{idx % 5}",
            )
            vec_items.append({
                "memory_id": row["id"],
                "content_version": row["version"],
                "fingerprint": fp,
                "vector": base_vec,
            })
        saved = self.store.save_vectors_batch(vec_items)
        self.assertEqual(saved["accepted"], 1000)

        # Warm Tier 0 search timing
        t0 = time.perf_counter()
        res = self.retriever.search("port 8512 node-2", limit=5)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(res["results"][0]["content"].split()[3], "#512:")
        self.assertLess(dt_ms, 250.0, f"Tier 0 search too slow on 1000 rows: {dt_ms:.1f} ms")

        # Verify bounded vector LRU did not grow past 128 entries
        meta = self.store.meta()
        self.assertLessEqual(meta["vector_cache_entries"], 128)
        rss_after = _rss_mb()
        if rss_after is not None and rss_before > 0:
            self.assertLess(rss_after - rss_before, 45.0, f"RAM delta too high: {rss_after - rss_before:.1f} MB")


if __name__ == "__main__":
    unittest.main()
