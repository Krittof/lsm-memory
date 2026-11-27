"""Unit & integration tests for LSMMemory Phase 2 WebUI bridge and reversible patcher."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest

from lsm_memory.daemon import MemoryDaemonServer
from lsm_memory.lsm_bridge import (
    VaultBridgeClient,
    VaultBridgeConfig,
    execute_vault_tool,
    project_memory_vault_messages,
    reset_bridge_cache,
    strip_recall_blocks,
    vault_scope_for_agent,
    vault_tool_definitions,
)
from lsm_memory.lsm_patcher import apply as patch_apply, revert as patch_revert, status as patch_status
from lsm_memory.retrieval import AdaptiveRetriever
from lsm_memory.store import MemoryStore


class TestLSMBridgeAndPatcher(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.db_path = self.root / "vault.sqlite3"
        reset_bridge_cache()

    def tearDown(self) -> None:
        reset_bridge_cache()
        self._td.cleanup()

    def _embedded_client(self) -> VaultBridgeClient:
        cfg = VaultBridgeConfig(enabled=True, db_path=str(self.db_path))
        return VaultBridgeClient(cfg)

    def test_split_injection_preserves_prompt_cache_and_checkpoint_purity(self) -> None:
        client = self._embedded_client()
        client.add_memory(
            "Tokyo ARM runs Caddy reverse proxy on port 443 and LSM on loopback 8765.",
            kind="episode",
            topic="tokyo-arm",
            search_terms=["东京主机", "反向代理"],
            source_client="gemini-web",
            machine="tokyo-arm",
        )

        # Simulate canonical persisted checkpoint history (state.messages)
        state_messages = [
            {"role": "system", "content": "STATIC_SYSTEM_PROMPT_DO_NOT_MUTATE"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！请问有什么需要帮忙？"},
            {"role": "user", "content": "帮我查一下东京主机的反向代理端口"},
        ]
        snapshot_before = copy.deepcopy(state_messages)

        projected = project_memory_vault_messages(
            state_messages,
            agent=None,
            namespace="",
            memory_feature_enabled=True,
            client=client,
        )

        # 1. Original state.messages (persisted checkpoint) is 100% untouched!
        self.assertEqual(state_messages, snapshot_before)
        # 2. Prompt Cache prefix (indices 0..2) is byte-for-byte identical!
        self.assertEqual(projected[0], snapshot_before[0])
        self.assertEqual(projected[1], snapshot_before[1])
        self.assertEqual(projected[2], snapshot_before[2])
        # 3. Recall block injected ONLY into the ephemeral copy of latest user turn
        self.assertIn("<lsm-memory-recall ", projected[3]["content"])
        self.assertIn("untrusted=\"true\"", projected[3]["content"])
        self.assertIn("loopback 8765", projected[3]["content"])
        self.assertTrue(projected[3]["content"].endswith("帮我查一下东京主机的反向代理端口"))

    def test_agent_suspended_resume_revalidation_and_stale_strip(self) -> None:
        """Simulate suspend -> suppress/forget/pause -> resume: zero memory leakage."""
        client = self._embedded_client()
        m = client.add_memory(
            "Confidential migration note: switch egress node only via IEPL.",
            kind="user_profile",
            topic="proxy",
            search_terms=["代理切换", "IEPL"],
        )
        mid = m["id"]

        # Turn 1 before AgentSuspended pause
        checkpoint_msgs = [
            {"role": "system", "content": "SYSTEM_PREFIX"},
            {"role": "user", "content": "代理切换应该用什么节点？"},
        ]
        p_before = project_memory_vault_messages(checkpoint_msgs, client=client)
        self.assertIn("IEPL", p_before[1]["content"])

        # Suppose an older buggy checkpoint somehow baked a stale recall tag into messages
        dirty_checkpoint = copy.deepcopy(p_before)

        # While Agent is suspended waiting for approval, user suppresses the memory!
        client.suppress(mid, action="suppress")

        # Turn 2 immediately after AgentSuspended resume:
        # Both clean checkpoint AND dirty checkpoint MUST strip stale recall and yield zero leakage!
        p_resumed_clean = project_memory_vault_messages(checkpoint_msgs, client=client)
        p_resumed_dirty = project_memory_vault_messages(dirty_checkpoint, client=client)
        self.assertNotIn("IEPL", p_resumed_clean[1]["content"])
        self.assertNotIn("<lsm-memory-recall", p_resumed_clean[1]["content"])
        self.assertEqual(p_resumed_dirty[1]["content"], "代理切换应该用什么节点？")

        # Restore -> visible again on next turn
        client.suppress(mid, action="restore")
        p_restored = project_memory_vault_messages(checkpoint_msgs, client=client)
        self.assertIn("IEPL", p_restored[1]["content"])

        # Global pause -> immediately hidden
        store, _ = client._embedded()
        store.set_global_pause(True)
        p_paused = project_memory_vault_messages(checkpoint_msgs, client=client)
        self.assertNotIn("IEPL", p_paused[1]["content"])
        store.set_global_pause(False)

        # Forget -> tombstoned & hidden forever
        client.forget(mid, reason="user requested forget")
        p_forgotten = project_memory_vault_messages(checkpoint_msgs, client=client)
        self.assertNotIn("IEPL", p_forgotten[1]["content"])

    def test_multimodal_small_talk_subagent_and_content_filter_guards(self) -> None:
        client = self._embedded_client()
        client.add_memory("Project codename is Orion-77.", search_terms=["代号", "Orion"])
        client.add_memory("Internal token marker BLOCKED_SECRET_WORD here.", search_terms=["代号", "secret"])

        # Small talk skipped cleanly
        st_msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "好的！"}]
        self.assertEqual(project_memory_vault_messages(st_msgs, client=client), st_msgs)

        # Delegated subagent skipped
        sub_msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "项目代号是什么"}]
        self.assertEqual(
            project_memory_vault_messages(sub_msgs, namespace="/subagent:call_1", client=client),
            sub_msgs,
        )

        # Memory feature disabled in WebUI conversation -> skipped
        self.assertEqual(
            project_memory_vault_messages(sub_msgs, memory_feature_enabled=False, client=client),
            sub_msgs,
        )

        # Multimodal user content + content_blocker drops blocked item but keeps safe item
        mm_msgs = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": [{"type": "text", "text": "查一下项目代号"}]},
        ]
        projected = project_memory_vault_messages(
            mm_msgs,
            client=client,
            content_blocker=lambda v: "BLOCKED_SECRET_WORD" in v,
        )
        parts = projected[1]["content"]
        self.assertIsInstance(parts, list)
        self.assertEqual(len(parts), 2)
        self.assertIn("Orion-77", parts[0]["text"])
        self.assertNotIn("BLOCKED_SECRET_WORD", parts[0]["text"])
        # Original multimodal list untouched
        self.assertEqual(len(mm_msgs[1]["content"]), 1)

    def test_daemon_mode_and_fail_closed_when_offline(self) -> None:
        token_file = self.root / "daemon.token"
        token = "test_phase2_token_abcdef123456"
        token_file.write_text(token, encoding="utf-8")
        st = MemoryStore(self.db_path)
        ret = AdaptiveRetriever(st)
        srv = MemoryDaemonServer(("127.0.0.1", 0), st, ret, token)
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        t.start()
        try:
            cfg = VaultBridgeConfig(
                enabled=True,
                origin=f"http://127.0.0.1:{port}",
                token_file=str(token_file),
                timeout_ms=1000,
            )
            client = VaultBridgeClient(cfg)
            # Execute built-in WebUI tools via daemon
            remembered = json.loads(execute_vault_tool(
                "memory_vault_remember",
                {"content": "Daemon bridge test: deploy on 1C1G smoothly.", "search_terms": ["1C1G部署"]},
                client=client,
                session_ref="conv_abc",
            ))
            self.assertEqual(remembered["id"], 1)

            msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "怎么1C1G部署？"}]
            proj = project_memory_vault_messages(msgs, client=client)
            self.assertIn("deploy on 1C1G smoothly", proj[1]["content"])

            searched = json.loads(execute_vault_tool("memory_vault_search", {"query": "1C1G部署"}, client=client))
            self.assertEqual(searched["results"][0]["id"], 1)

            suppressed = json.loads(execute_vault_tool(
                "memory_vault_suppress", {"memory_id": 1, "action": "suppress"}, client=client
            ))
            self.assertTrue(suppressed["suppressed"])

            proj_after = project_memory_vault_messages(msgs, client=client)
            self.assertNotIn("deploy on 1C1G smoothly", proj_after[1]["content"])

            cap = json.loads(execute_vault_tool("memory_vault_capture", {"content": "raw note"}, client=client))
            self.assertEqual(cap["status"], "pending")
            st = json.loads(execute_vault_tool("memory_vault_status", {}, client=client))
            self.assertEqual(st["counts"]["inbox_pending"], 1)
        finally:
            srv.shutdown()
            srv.server_close()

        # Daemon now offline -> projection fails closed cleanly (returns unmodified msgs, no exception)
        proj_offline = project_memory_vault_messages(msgs, client=client)
        self.assertEqual(proj_offline, msgs)

    def test_scope_mapping_and_env_config(self) -> None:
        self.assertEqual(vault_scope_for_agent(None), "personal")
        self.assertEqual(vault_scope_for_agent({"id": "agt_1", "memory_scope": "user"}), "personal")
        self.assertEqual(vault_scope_for_agent({"id": "agt_1____ephem", "memory_scope": "agent"}), "agent:agt_1")

        # Non-loopback origin rejected fail-closed
        old = os.environ.copy()
        try:
            for k in list(os.environ):
                if k.startswith("LOCAL_SHELL_MCP_MEMORY_VAULT_"):
                    del os.environ[k]
            self.assertFalse(VaultBridgeConfig.from_env().enabled)
            self.assertEqual(vault_tool_definitions(), [])

            os.environ["LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN"] = "http://203.0.113.10:8775"
            self.assertFalse(VaultBridgeConfig.from_env().enabled)

            os.environ["LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN"] = "http://127.0.0.1:8775"
            cfg = VaultBridgeConfig.from_env()
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.mode, "daemon")
            self.assertEqual(len(vault_tool_definitions(cfg)), 6)
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_patcher_apply_idempotent_and_byte_exact_revert(self) -> None:
        fake_pkg = self.root / "local_shell_mcp"
        webui = fake_pkg / "webui"
        webui.mkdir(parents=True)

        # Use an explicit upstream LSM agent_runner.py when provided via env, else realistic stub
        upstream_env = os.environ.get("LSM_UPSTREAM_RUNNER_PATH", "").strip()
        upstream_runner = Path(upstream_env) if upstream_env else None
        if upstream_runner and upstream_runner.is_file():
            shutil.copy2(upstream_runner, webui / "agent_runner.py")
        else:
            (webui / "agent_runner.py").write_text(
                "def _ctx(self, tools, config):\n"
                "    if True:\n"
                "        if True:\n"
                "            tools += tool_definitions(config.get('validKeys'))\n"
                "def _inv(self, call):\n"
                "    if True:\n"
                '        if call.name in {"web_search", "web_fetch"}:\n'
                "            return None\n"
                "def _proj(self, state):\n"
                "    if True:\n"
                "        projected = project_faded_provider_messages(state.messages, state.context.fading_tier)\n"
                "        return projected\n",
                encoding="utf-8",
            )
        (webui / "approval_settings.py").write_text(
            "def is_dangerous(name, arguments):\n"
            '    if name in _READ_TOOLS or name in {"web_search", "web_fetch", "ask_user_question", "subagent"}:\n'
            "        return False\n"
            "    return True\n",
            encoding="utf-8",
        )
        (fake_pkg / "main.py").write_text(
            "def _with_oauth_routes(inner_app):\n"
            "    if True:\n"
            "        if True:\n"
            "            routes = []\n"
            "            routes[2:2] = memory_routes()\n"
            "    return routes\n",
            encoding="utf-8",
        )
        (webui / "frontend.py").write_text(
            "async def frontend(request):\n"
            '    return FileResponse(path, headers={"Cache-Control": cache}, media_type=media_type)\n',
            encoding="utf-8",
        )

        orig_runner = (webui / "agent_runner.py").read_bytes()
        orig_approval = (webui / "approval_settings.py").read_bytes()
        orig_main = (fake_pkg / "main.py").read_bytes()
        orig_frontend = (webui / "frontend.py").read_bytes()

        st0 = patch_status(fake_pkg)
        self.assertFalse(st0["all_applied"])

        # Apply #1
        res1 = patch_apply(fake_pkg)
        self.assertTrue(res1["files"]["webui/agent_runner.py"]["changed"])
        self.assertTrue(res1["files"]["webui/approval_settings.py"]["changed"])
        self.assertTrue(res1["files"]["main.py"]["changed"])
        self.assertTrue(res1["files"]["webui/frontend.py"]["changed"])
        st1 = patch_status(fake_pkg)
        self.assertTrue(st1["all_applied"])
        self.assertTrue(st1["files"]["webui/agent_runner.py"]["backup_exists"])
        self.assertTrue(st1["files"]["main.py"]["backup_exists"])
        self.assertTrue(st1["files"]["webui/frontend.py"]["backup_exists"])

        # Apply #2 (strictly idempotent!)
        res2 = patch_apply(fake_pkg)
        self.assertFalse(res2["files"]["webui/agent_runner.py"]["changed"])
        self.assertFalse(res2["files"]["webui/approval_settings.py"]["changed"])
        self.assertFalse(res2["files"]["main.py"]["changed"])
        self.assertFalse(res2["files"]["webui/frontend.py"]["changed"])

        # Revert -> byte-for-byte identical to original upstream files!
        rev = patch_revert(fake_pkg)
        self.assertTrue(rev["files"]["webui/agent_runner.py"]["changed"])
        self.assertEqual((webui / "agent_runner.py").read_bytes(), orig_runner)
        self.assertEqual((webui / "approval_settings.py").read_bytes(), orig_approval)
        self.assertEqual((fake_pkg / "main.py").read_bytes(), orig_main)
        self.assertEqual((webui / "frontend.py").read_bytes(), orig_frontend)
        self.assertFalse(patch_status(fake_pkg)["all_applied"])


if __name__ == "__main__":
    unittest.main()
