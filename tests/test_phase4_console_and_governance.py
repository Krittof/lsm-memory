"""Unit & integration tests for Phase 4 governance, hot-reload Tier 1A, graph, and WebUI console."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from lsm_memory.consolidator import build_consolidation_pack
from lsm_memory.daemon import MemoryDaemonServer
from lsm_memory.governance import (
    GovernanceDeniedError,
    enforce_operation,
    load_governance,
    save_governance,
)
from lsm_memory.lsm_bridge import VaultBridgeClient, VaultBridgeConfig, project_memory_vault_messages
from lsm_memory.mcp_server import DirectBackend, HttpBackend
from lsm_memory.retrieval import AdaptiveRetriever
from lsm_memory.store import MemoryStore
from lsm_memory.vector_sync import WORKER_RESULT_PREFIX, build_worker_embed_script, parse_worker_embed_output
from lsm_memory.webui_console import _NAV_LAUNCHER_JS, _client_call, render_console_html


def _mock_embed_http(dim: int = 16) -> tuple[ThreadingHTTPServer, str]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # noqa: ANN001
            return

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            inputs = body.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            data = [{"object": "embedding", "index": i, "embedding": [0.25] * dim} for i in range(len(inputs))]
            raw = json.dumps({"object": "list", "data": data}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1/embeddings"


class TestPhase4GovernanceAndConsole(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.db_path = self.root / "vault.sqlite3"
        self.store = MemoryStore(self.db_path)
        self.ret = AdaptiveRetriever(self.store)

    def tearDown(self) -> None:
        self.ret.close()
        self._td.cleanup()

    def test_machine_include_exclude_and_auto_inject_governance(self) -> None:
        m_a = self.store.add_memory("Alpha host runs nginx on port 8080.", machine="host-a", search_terms=["主机A"])
        m_b = self.store.add_memory("Beta host runs caddy on port 8080.", machine="host-b", search_terms=["主机B"])
        self.store.enqueue_inbox("Alpha draft note", machine="host-a")
        self.store.enqueue_inbox("Beta draft note", machine="host-b")

        # Exclude host-b -> hidden from search AND skipped in consolidation inbox!
        save_governance(self.store, {"machines": {"host-b": {"included": False, "role": "read_only", "auto_consolidate": False}}})
        res = self.ret.search("port 8080", limit=10)
        ids = [r["id"] for r in res["results"]]
        self.assertIn(m_a["id"], ids)
        self.assertNotIn(m_b["id"], ids)

        pack = build_consolidation_pack(self.store)
        inbox_machines = [ev["machine"] for ev in pack["inbox_events"]]
        self.assertEqual(inbox_machines, ["host-a"])

        # Turning auto_inject off immediately halts WebUI split-injection (respect_pause=True)
        save_governance(self.store, {"auto_inject": False})
        client = VaultBridgeClient(VaultBridgeConfig(enabled=True, db_path=str(self.db_path)))
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "查一下 port 8080"}]
        proj = project_memory_vault_messages(msgs, client=client)
        self.assertEqual(proj, msgs)

    def test_mcp_role_enforcement_readonly_and_capture_only(self) -> None:
        save_governance(self.store, {"mcp_default_role": "read_only"})
        direct = DirectBackend(self.db_path)
        # Read allowed
        self.assertIn("counts", direct.call("memory_status", {}))
        # Write and capture denied!
        with self.assertRaises(GovernanceDeniedError):
            direct.call("memory_add", {"content": "Blocked write"})
        with self.assertRaises(GovernanceDeniedError):
            direct.call("memory_capture", {"content": "Blocked capture"})

        # read_capture allows capture, blocks add/forget
        save_governance(self.store, {"mcp_default_role": "read_capture"})
        ev = direct.call("memory_capture", {"content": "Allowed inbox draft"})
        self.assertEqual(ev["status"], "pending")
        with self.assertRaises(GovernanceDeniedError):
            direct.call("memory_add", {"content": "Still blocked direct add"})

        # Per-machine stricter role overrides read_write default
        save_governance(
            self.store,
            {"mcp_default_role": "read_write", "machines": {"untrusted-box": {"included": True, "role": "read_only"}}},
        )
        ok_mem = direct.call("memory_add", {"content": "Trusted write", "machine": "trusted-box"})
        self.assertEqual(ok_mem["id"], 1)
        with self.assertRaises(GovernanceDeniedError):
            direct.call("memory_add", {"content": "Untrusted write", "machine": "untrusted-box"})

    def test_daemon_governance_graph_revisions_and_hot_reload_tier1a(self) -> None:
        token = "phase4_daemon_token_0123456789abcdef"
        httpd = MemoryDaemonServer(("127.0.0.1", 0), self.store, self.ret, token)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        emb_srv, emb_url = _mock_embed_http(dim=16)
        try:
            origin = f"http://127.0.0.1:{port}"
            client = VaultBridgeClient(VaultBridgeConfig(enabled=True, origin=origin, token=token))

            # Add 2 memories, edit #1, add relation, inspect revisions & graph
            m1 = _client_call(client, "POST", "/v1/memories", {"content": "Node one initial", "topic": "net", "machine": "m1"})
            m2 = _client_call(client, "POST", "/v1/memories", {"content": "Node two linked", "topic": "net", "machine": "m2"})
            _client_call(client, "POST", f"/v1/memories/{m1['id']}/correct", {"expected_version": 1, "content": "Node one updated"})
            _client_call(client, "POST", "/v1/relations", {"source_id": m1["id"], "target_id": m2["id"], "relation_type": "depends_on"})

            revs = _client_call(client, "GET", f"/v1/memories/{m1['id']}/revisions")["revisions"]
            self.assertEqual(len(revs), 2)
            graph = _client_call(client, "GET", "/v1/graph")
            self.assertEqual(len(graph["nodes"]), 2)
            self.assertEqual(len(graph["edges"]), 1)

            # Delete relation
            del_r = _client_call(client, "POST", "/v1/relations/delete", {"source_id": m1["id"], "target_id": m2["id"]})
            self.assertEqual(del_r["deleted"], 1)

            # Hot-test & hot-apply Tier 1A embedding config (secret masked in GET!)
            probe = _client_call(
                client,
                "POST",
                "/v1/config/embedding/test",
                {"url": emb_url, "model": "mock-bge", "api_key": "secret-embed-key-123"},
            )
            self.assertTrue(probe["reachable"])
            self.assertEqual(probe["dim"], 16)

            applied = _client_call(
                client,
                "POST",
                "/v1/config/embedding",
                {"enabled": True, "tier_mode": "tier1a", "url": emb_url, "model": "mock-bge", "api_key": "secret-embed-key-123"},
            )
            self.assertTrue(applied["enabled"])
            self.assertEqual(applied["tier_mode"], "tier1a")
            self.assertTrue(applied["has_api_key"])
            self.assertNotIn("api_key", applied)
            self.assertEqual(applied["active_tier"], "tier1_hybrid_rrf")

            # 1-click Tier 1A server-side backfill using saved provider
            bf = _client_call(client, "POST", "/v1/config/embedding/backfill", {})
            self.assertEqual(bf["dim"], 16)
            self.assertEqual(bf["accepted"], 2)

            # Save Tier 1B worker_profile without touching active query-time provider
            wp_saved = _client_call(
                client,
                "POST",
                "/v1/config/embedding",
                {
                    "worker_profile_only": True,
                    "worker_profile": {
                        "worker": "gpu-box",
                        "llama_server": "/usr/local/bin/llama-server",
                        "model_path": "/models/embed.gguf",
                        "last_model_name": "embed.gguf",
                    },
                },
            )
            self.assertEqual(wp_saved["worker_profile"]["worker"], "gpu-box")
            self.assertEqual(wp_saved["worker_profile"]["last_model_name"], "embed.gguf")
            self.assertTrue(wp_saved["enabled"])

            # Verify MCP HttpBackend gets HTTP 403 when mcp_default_role is read_only
            _client_call(client, "POST", "/v1/governance", {"mcp_default_role": "read_only"})
            mcp_http = HttpBackend(origin, token)
            with self.assertRaises(RuntimeError) as ctx:
                mcp_http.call("memory_add", {"content": "Should be denied over MCP"})
            self.assertIn("read_only", str(ctx.exception))

            # Switch back to Tier 0 cleanly
            t0 = _client_call(client, "POST", "/v1/config/embedding", {"enabled": False, "tier_mode": "tier0", "url": "", "clear_api_key": True})
            self.assertFalse(t0["enabled"])
            self.assertEqual(t0["tier_mode"], "tier0")
            self.assertFalse(t0["has_api_key"])
            self.assertEqual(t0["active_tier"], "tier0_zero_model")
        finally:
            emb_srv.shutdown()
            emb_srv.server_close()
            httpd.shutdown()
            httpd.server_close()

    def test_console_html_is_zero_cdn_bilingual_with_help_popovers_and_tier_cards(self) -> None:
        html = render_console_html()
        self.assertIn("LSM Memory Vault", html)
        self.assertIn("graphCanvas", html)
        self.assertIn("govMcpRole", html)
        self.assertIn("embUrl", html)
        # 3-mode tier cards & worker backfill UI
        self.assertIn("card-tier0", html)
        self.assertIn("card-tier1a", html)
        self.assertIn("card-tier1b", html)
        self.assertIn("runWorkerBackfill", html)
        self.assertIn("embFpSelect", html)
        # Language sync with LSM localStorage('lang') & full bilingual HELP popovers
        self.assertIn("localStorage.getItem('lang')", html)
        self.assertIn("helpPopover", html)
        self.assertIn('class="help-tip"', html)
        self.assertIn("localStorage.getItem('lang')", _NAV_LAUNCHER_JS)
        # Zero external CDN links
        self.assertNotIn("cdn.jsdelivr", html)
        self.assertNotIn("unpkg.com", html)

    def test_worker_embed_script_builder_and_parser(self) -> None:
        script = build_worker_embed_script(
            "/usr/local/bin/llama-server",
            "/models/qwen3-embed.gguf",
            [{"memory_id": 7, "content_version": 2, "content": "Alpha fact", "topic": "infra", "search_terms": ["alias1"]}],
        )
        compile(script, "<worker_embed>", "exec")
        parsed = parse_worker_embed_output(
            "some log line\n" + WORKER_RESULT_PREFIX + json.dumps({"dim": 1024, "model_name": "qwen3-embed.gguf", "vectors": []})
        )
        self.assertEqual(parsed["dim"], 1024)
        self.assertEqual(parsed["model_name"], "qwen3-embed.gguf")


if __name__ == "__main__":
    unittest.main()
