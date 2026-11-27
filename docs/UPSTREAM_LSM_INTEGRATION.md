# Upstream Integration & Re-Deployment Guide (`local-shell-mcp`)

This document describes how `lsm_memory` integrates cleanly into `fwerkor/local-shell-mcp` (LSM v6.0.3+) with **100% backward compatibility**, **zero third-party runtime dependencies**, **1C1G VPS safety**, and **repeatable deployment**.

## 1. Compatibility Guarantees

- **Zero Breaking Changes to Legacy LibreChat Memory**:
  - `webui_memories` SQLite schema, `/api/memories*` REST endpoints, `set_memory` / `delete_memory` tools, `memory_protection.py`, and `memory_agent.py` remain completely unchanged.
- **Opt-In by Default**:
  - When no `LOCAL_SHELL_MCP_MEMORY_VAULT_*` environment variable is set, the WebUI bridge is a fast no-op (`return projected`) with zero I/O and zero RAM overhead.
- **Respects Every LSM Policy Boundary**:
  - Obeys per-conversation feature toggles (`effective_features["memory"]`), user preferences (`user.memories`), Agent memory partitioning (`agent_partition(agent)` -> `personal` or `agent:<id>`), approval modes (`approval_settings.is_dangerous`), and content-protection filters (`configured_filters` / `memory_protection.inspect_memory_content`).
- **Prompt Cache & Checkpoint Safety**:
  - Dynamic recall runs exclusively inside `AgentRunner._project_model_messages()` (LSM's ephemeral provider projection layer).
  - `state.messages[0]` (static system prompt + LibreChat preamble) is **never mutated**, preserving provider prompt caching.
  - Ephemeral `<lsm-memory-recall ...>` blocks are **never written** to SQLite checkpoints or conversation history.
- **`AgentSuspended` Resume Revalidation**:
  - Because projection runs freshly on every model turn (including after checkpoint restore) and strips any stale `<lsm-memory-recall>` blocks, any memory suppressed, forgotten, or globally paused while an Agent is suspended on human approval or `ask_user_question` is immediately excluded upon resume.

## 2. Environment Variables (Standard `LOCAL_SHELL_MCP_*` Namespace)

| Variable | Mode | Default | Description |
| :--- | :--- | :--- | :--- |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED` | Both | `0` (auto `1` if `ORIGIN` or `DB` set) | Explicit toggle (`1`/`true` or `0`/`false`). |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN` | Daemon | `""` | Loopback HTTP origin, e.g. `http://127.0.0.1:8775`. Non-loopback URLs are rejected. |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN_FILE` | Daemon | `""` | Path to `chmod 600` Bearer token file. |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN` | Daemon | `""` | Optional direct token fallback if `TOKEN_FILE` is not used. |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_DB` | Embedded | `""` | SQLite file path for single-process embedded mode (no extra daemon needed). |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TIMEOUT_MS` | Both | `750` | Fail-closed recall deadline in milliseconds (clamped `50..3000`). |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_ITEMS` | Both | `5` | Max recalled memories injected per turn (`1..10`). |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_CHARS` | Both | `1200` | Max total recall packet characters (`200..4000`). |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOOLS` | Both | `1` | Register built-in WebUI vault tools (`memory_vault_*`) alongside legacy memory tools. |
| `LSM_MEMORY_EMBED_URL` | Daemon | `""` | Optional OpenAI-compatible `/v1/embeddings` URL to activate Tier 1/2 hybrid RRF (blank = Tier 0 zero-model). |
| `LSM_MEMORY_EMBED_MODEL` | Daemon | `"default"` | Logical embedding model name matching `embedding_cache` fingerprint. |
| `LSM_MEMORY_EMBED_MODEL_SHA256` | Daemon | `"api"` | Logical model SHA-256 (or GGUF SHA-256) for cross-platform vector reuse. |
| `LSM_MEMORY_LSM_DB` | Daemon | `""` | Optional path to LSM Controller SQLite DB for read-only (`mode=ro`) incremental consolidation collection. |

## 3. Upstream PR File Mapping

When contributing directly into the `fwerkor/local-shell-mcp` repository:

| Standalone `D:\LSMMemory` | Upstream `local_shell_mcp` Target | Notes |
| :--- | :--- | :--- |
| `lsm_memory/{store,policy,retrieval,daemon,mcp_server,cli,embedder,consolidator,vector_sync,governance}.py` | `src/local_shell_mcp/memory_vault/*.py` | Pure Python 3.11+ stdlib; zero new `pyproject.toml` dependencies. |
| `lsm_memory/lsm_bridge.py` | `src/local_shell_mcp/webui/memory_vault.py` | Clean WebUI adapter with lazy client & content-filter bridge. |
| `lsm_memory/webui_console.py` | `src/local_shell_mcp/webui/memory_vault_routes.py` | Authenticated `/memory-vault` LSM-styled zero-CDN console + `/api/memory-vault/*` JSON API. |
| `lsm_patcher.py` hook #1 (`_project_model_messages`) | `src/local_shell_mcp/webui/agent_runner.py` | 4-line call after `project_faded_provider_messages(...)`. |
| `lsm_patcher.py` hook #2 (`_context` & `_invoke`) | `src/local_shell_mcp/webui/agent_runner.py` | 12-line opt-in tool registration & dispatch next to `set_memory`. |
| `lsm_patcher.py` hook #3 (`is_dangerous`) | `src/local_shell_mcp/webui/approval_settings.py` | Treat read-only `memory_vault_search` / `memory_vault_status` as non-dangerous. |
| `lsm_patcher.py` hook #4 (`memory_vault_web_routes`) | `src/local_shell_mcp/main.py` | 5-line mount next to `memory_routes()`. |
| `tests/test_lsm_memory.py` + `test_lsm_bridge_and_patcher.py` | `tests/test_memory_vault*.py` | Runs under stdlib `unittest` or `pytest` with zero network/model downloads. |

## 4. Re-Deployment & Upgrade Workflow (Overlay Mode)

Until merged upstream, `lsm_memory` deploys cleanly as an overlay package into any LSM venv:

```bash
# Fresh or repeat install on any Linux host (1C1G+ ARM64 or x86_64)
sudo bash deploy/install.sh --mode daemon --patch-lsm

# After upgrading local-shell-mcp wheel (e.g. v6.0.3 -> v6.0.4)
/opt/local-shell-mcp/venv/bin/python -m lsm_memory.lsm_patcher status
/opt/local-shell-mcp/venv/bin/python -m lsm_memory.lsm_patcher apply

# Clean rollback at any time (restores byte-identical original LSM files)
/opt/local-shell-mcp/venv/bin/python -m lsm_memory.lsm_patcher revert
```
