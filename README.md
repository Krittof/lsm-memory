# LSMMemory

[![CI](https://github.com/Krittof/lsm-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/Krittof/lsm-memory/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-brightgreen.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **[English](README.md) | [简体中文](README.zh-CN.md)**

**A multi-tier, zero-dependency, cross-network persistent memory vault and WebUI split-injection bridge for [`local-shell-mcp` (LSM)](https://github.com/fwerkor/local-shell-mcp).**

`LSMMemory` provides a shared, auditable long-term memory backend for **LSM WebUI** and any external MCP client connected to LSM `/mcp` (such as Gemini Web, Claude, Codex, or Antigravity), implemented entirely in the **Python 3.11+ standard library** with ~14 MB idle RAM on 1C1G hosts.

---

## Features & Compatibility

`LSMMemory` extends LSM v6.0.3+ without modifying or breaking existing `webui_memories`, `/api/memories*`, `set_memory`/`delete_memory`, `memory_protection.py`, or `memory_agent.py`:

| Area | Legacy `webui_memories` | **LSMMemory (`v0.4.0`)** |
| :--- | :--- | :--- |
| **Scope & Sharing** | LSM WebUI only | Shared across LSM WebUI and all `/mcp` clients |
| **Retrieval Engine** | Full linear dump until token limit | **Multi-tier adaptive engine**: Tier 0 zero-model (~14 MB) to Tier 1/2 Hybrid RRF vectors |
| **Injection & Caching** | Appended to static system prompt | **Split-injection** inside `_project_model_messages()`: preserves `messages[0]` Prompt Cache prefix and leaves checkpoints clean |
| **Suspend / Resume** | Frozen in checkpoint on `AgentSuspended` | Live revalidation on every turn + stale recall tag stripping (zero leakage after suppress/forget/pause) |
| **Lifecycle & Governance** | Overwrite / hard delete | CAS locking, revision history, soft suppress/restore, tombstones, **per-machine include/exclude & role controls (`read_write` / `read_capture` / `read_only`)**, global pause |
| **WebUI Console** | Flat KV modal only | **Zero-dependency LSM-styled `/memories/vault` console** (reuses LSM auth): memory CRUD, **2D Canvas relation graph**, inbox manager, machine governance, hot-reload Tier 1A config & legacy import |
| **Offline Consolidation** | Per-turn extra LLM call only | Read-only (`mode=ro`) LSM SQLite collector + MCP work-packet tools (`pull`/`commit`) usable by any connected Agent |
| **Runtime Footprint** | Lightweight | Pure Python 3.11+ stdlib (**0 third-party dependencies**), ~14–16 MB cgroup RAM, `< 40 ms` warm search on 1,000 rows |

---

## Architecture & How It Works

```text
  [External MCP Clients: Gemini Web / Claude / Codex / CLI]     [LSM WebUI (AgentRunner)]
                             │  HTTPS /mcp (OAuth 2.1)                    │  _project_model_messages()
                             ▼                                            ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                        LSM Controller Host (1C1G+ Linux / ARM64 / x86_64)              │
  │                                                                                        │
  │   LSM Nested MCP (`lsm_memory`, stdio on-demand) ◄──► LSMMemory Core (Daemon / Embedded)│
  │                                                       • SQLite WAL (Single Writer)     │
  │   LSM Controller SQLite (read-only `mode=ro`) ──────► • Tier 0 / Tier 1-2 Retriever    │
  │                                                       • Compact `array('f')` Vec LRU   │
  └──────────────────────────┬───────────────────────────────────────────┬─────────────────┘
                             │ Pull/Commit work packets                  │ Missing/Upsert vectors
                             │ (HTTPS /mcp, no extra keys)               │ (HTTPS /mcp, no SSH)
                             ▼                                           ▼
              [Any LSM-Connected Agent]                 [Workstation GPU (`llama-server`)
              Distills atomic memories,                  or Cloud `/v1/embeddings` API]
              bilingual aliases & relations              (On-demand, 0 MB idle VRAM)
```

### 1. Storage, Consistency & Lifecycle (`store.py`, `policy.py`)
- **Single-Writer SQLite WAL**: Uses `BEGIN IMMEDIATE` transactions and CAS optimistic locking (`expected_version`) to prevent concurrent overwrites across clients.
- **Strict Separation of Suppress vs. Forget**:
  - `suppress` / `restore`: Soft-hides a memory from recall without deleting it and increments a monotonic `global_revision`.
  - `forget`: Marks the row forgotten and writes a `memory_tombstones` hash record so backups or sync workers cannot silently resurrect deleted data.
  - `global_pause`: Immediately halts automatic recall injection globally.
- **Content Safety**: Deterministic secret/PII checks block API keys (`sk-`, `Bearer `, private keys, tokens) prior to insertion; WebUI integration also enforces LSM's `configured_filters` / `memory_protection`.
- **1C1G Compact Vector Cache**: Precomputed embeddings are stored in a dimension-agnostic `embedding_cache` keyed by a cross-platform **logical model fingerprint** (`sha256(model_name + model_sha256 + dim)`) decoupled from OS binary hashes, and cached in memory as bounded `array('f')` buffers (9× smaller than Python float tuples).

### 2. Multi-Tier Adaptive Retrieval (`retrieval.py`, `embedder.py`)
- **Tier 0 — Zero-Model Default (~14 MB RAM)**:
  1. NFKC normalization + CJK bigram + ASCII tokenization with BM25 scoring.
  2. **3× weighting on `search_terms` and `topic`** so offline-expanded bilingual aliases bridge cross-lingual queries without a local model.
  3. Exact substring channel for short identifiers (ports, IPs, file paths, versions).
  4. Clause-level MaxSim reranking to prevent signal dilution in longer entries.
  5. **1-hop typed relation graph expansion** (`related`, `depends_on`, `supersedes`) to pull linked context automatically.
  6. Conservative NLI contradiction hints and trivial small-talk skipping.
- **Tier 1 / Tier 2 — Optional Hybrid Vector RRF**:
  - Enabled by configuring `LSM_MEMORY_EMBED_URL` (any OpenAI-compatible `/v1/embeddings` endpoint: cloud API or local `llama-server`).
  - Combines Tier 0 candidates and cosine-scored cached vectors via Reciprocal Rank Fusion (`k=60`) with a minimum cosine abstention floor.
  - Enforces a strict **750 ms wall-clock recall deadline**; on timeout, error, or outage, retrieval fails closed back to Tier 0 automatically.

### 3. WebUI Split-Injection & `AgentSuspended` Resume Safety (`lsm_bridge.py`, `lsm_patcher.py`)
- Hooks into `AgentRunner._project_model_messages()`, LSM's ephemeral provider projection layer:
  - **`messages[0]` is never mutated**, preserving static system prompt caching across turns.
  - **Checkpoints remain clean**: Ephemeral `<lsm-memory-recall scope="..." revision="..." untrusted="true">` blocks are attached only to a shallow copy of the latest user turn (compatible with OpenAI, Anthropic, and Gemini message schemas) and are never persisted into SQLite checkpoints.
  - **Resume Revalidation**: Because projection runs fresh before every model turn (including after resuming from `AgentSuspended`) and strips any stale `<lsm-memory-recall>` tags, memories suppressed, forgotten, or paused during suspension cannot leak on resume.
  - Obeys LSM's `effective_features["memory"]`, `user.memories`, `agent_partition(agent)` (`personal` vs. `agent:<id>`), and `approval_settings.is_dangerous()` (read-only vault tools skip dangerous-tool approval).

### 4. Offline Consolidation & Distributed Vector Backfill (`consolidator.py`, `vector_sync.py`)
- **Offline Consolidation**: Captures raw notes into `inbox_events` (`memory_capture`) and optionally reads LSM's Controller SQLite database strictly read-only (`mode=ro`) via a durable incremental cursor. Any LSM-connected Agent can pull a secret-scrubbed batch (`memory_consolidate_pull`), distill atomic memories with bilingual `search_terms` and relations, and commit atomically (`memory_consolidate_commit`) with zero extra API keys. An optional OpenAI-compatible chat distiller (`lsm-memory-consolidate run-openai`) is also included.
- **Distributed Vector Backfill**: `lsm-memory-sync` pulls memories missing vectors for a target logical fingerprint, computes normalized vectors using either a cloud `/v1/embeddings` API or an on-demand `ManagedLlamaEmbedServer` (starts loopback `llama-server --embeddings`, processes the batch, and terminates to release 100% VRAM), and uploads them via SQLite, loopback HTTP, or **HTTPS `/mcp` OAuth 2.1** (no SSH keys or extra open ports required).

---

## Retrieval Mode Comparison

| Mode | Query Vector Source | Document Vector Source | Controller RAM | Requirements |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0 (Default)** | None (lexical + aliases + MaxSim + 1-hop graph) | Offline Agent alias & relation enrichment | **~14 MB** | Python 3.11+ stdlib only (1C1G compatible) |
| **Tier 1A (Cloud API)** | OpenAI-compatible `/v1/embeddings` API | Same `/v1/embeddings` API via `vector_sync` | **~14 MB** | Cloud embedding API key (1C1G compatible) |
| **Tier 1B (Private Local)** | Controller local CPU `llama-server` (same GGUF) | Workstation GPU `llama-server` via HTTPS `/mcp` | ~750 MB (4G+ host) | Local GGUF model + `llama-server` |

---

## Benchmarks

Measured against an isolated benchmark corpus on an **ARM64 Linux controller (Ubuntu 22.04 `aarch64`)** and a **workstation GPU running `Qwen3-Embedding-0.6B-Q8_0.gguf`**, comparing legacy LSM `webui_memories` full-dump injection against `LSMMemory` Tier 0 split-injection:

### Token Savings & ARM64 Latency Scaling

| Library Size | Legacy Full Dump (est. BPE tokens/turn) | LSMMemory Avg (incl. small-talk skip) | Per-Turn Token Reduction | ARM64 Warm Search Latency |
| :---: | :---: | :---: | :---: | :---: |
| **20** | ~982 | ~224 | **77.2%** | `4.42 ms` |
| **50** | ~2,448 | ~225 | **90.8%** | `7.93 ms` |
| **100** | ~4,890 | ~226 | **95.4%** | `14.08 ms` |
| **200** | **6,890** (bilingual corpus) | **100.0** (`140.8` substantive / `0` small-talk) | **98.5%** | `25.36 ms` |
| **500** | ~24,526 | ~226 | **99.1%** | `62.13 ms` |
| **1,000** | ~49,073 (exceeds typical limit) | ~217 | **99.6%** | `121.26 ms` |

### 20-Turn Session & Quality Summary (200 Memories, 30-Query Bilingual Suite, Tier 0)

| Metric | Legacy Full Dump | **LSMMemory (Tier 0)** |
| :--- | :---: | :---: |
| **20-Turn Cumulative Memory Tokens** (15 substantive + 5 small-talk) | `137,800` | **`2,000`** (**135,800 saved**) |
| **Prompt Cache Prefix Invalidations** (2 mid-session edits/suppresses) | `2` (`~29,780` uncached recompute tokens) | **`0`** (`messages[0]` untouched) |
| **Recall@5 / MRR** (25 substantive queries: exact IDs + bilingual aliases + 1-hop) | N/A (99.5% noise ratio) | **`96.0%` / `0.940`** (`2.92` items/turn) |
| **1-Hop Associative Neighbor Recall@5** (5 linked pairs) | N/A | **`5/5 (100%)`** |
| **Small-Talk Skip Rate** / **`AgentSuspended` Resume Leakage** | `0%` / leaks frozen snapshot | **`100% (0 tokens)` / `0 leakage`** |
| **Loopback Daemon Latency & Idle Memory (ARM64)** | — | **`2.22 ms` avg**, **`~14.7 MB` cgroup RAM** |
| **Workstation GPU Vector Backfill (`Qwen3-Embedding-0.6B`, 1024-dim)** | — | **`152.8 items/s`** (50 items in `327 ms`, `0 MB` idle VRAM) |

---

## Installation & Deployment

### 1. Clone and Verify

```bash
git clone https://github.com/Krittof/lsm-memory.git
cd lsm-memory
python3 -B -m unittest discover -s tests -t . -v
```

### 2. Deploy on an LSM Controller Host (Linux ARM64 / x86_64)

```bash
# Daemon mode (recommended for shared WebUI + /mcp concurrency; 128M systemd MemoryMax)
sudo bash deploy/install.sh --mode daemon --patch-lsm
sudo systemctl restart local-shell-mcp.service

# Embedded mode (single process inside LSM, no extra systemd daemon)
sudo bash deploy/install.sh --mode embedded --patch-lsm
sudo systemctl restart local-shell-mcp.service
```

### 3. Register in LSM Nested MCP (`mcp_manage`)

Register `lsm_memory` as an on-demand `stdio` MCP server (0 MB idle RAM, exposes 12 tools):

```json
{
  "action": "register",
  "name": "lsm_memory",
  "transport": "stdio",
  "command": "/opt/local-shell-mcp/venv/bin/python",
  "args": [
    "-B", "-m", "lsm_memory.mcp_server",
    "--origin", "http://127.0.0.1:8775",
    "--token-file", "/var/lib/lsm-memory/daemon.token"
  ],
  "cwd": ".",
  "overwrite": true,
  "refresh": true
}
```

---

## Tools & CLI Usage

### MCP Tools (`lsm_memory:*`, 12 tools)
- **Core**: `memory_search`, `memory_add`, `memory_correct`, `memory_suppress`, `memory_forget`, `memory_relate`, `memory_capture`, `memory_status`
- **Offline Consolidation & Vector Sync**: `memory_consolidate_pull`, `memory_consolidate_commit`, `memory_vectors_missing`, `memory_vectors_upsert`

### Built-in LSM WebUI Tools & Console
- **WebUI Console (`/memories/vault`)**: Mounted directly under LSM's native WebUI authentication (`https://<your-lsm-domain>/memories/vault`) with a top-right launcher pill (`● Memory Vault` / `● 记忆库`), automatic language synchronization with LSM WebUI (`localStorage.lang`, with manual EN/ZH toggle), and inline `?` contextual help popovers across all controls.
- **WebUI Agent Tools (6 tools)**: `memory_vault_search` (read-only), `memory_vault_status` (read-only), `memory_vault_remember`, `memory_vault_suppress`, `memory_vault_forget`, `memory_vault_capture`

---

## WebUI Console Operations Manual

Once `LSMMemory` is deployed with `--patch-lsm` and `local-shell-mcp.service` is restarted, click the top-right **`● Memory Vault`** (or **`● 记忆库`** in Chinese UI) pill in LSM WebUI, or open `https://<your-lsm-domain>/memories/vault` directly.

### Header Bar & Language Sync
- **Status Badges**: Live version, active retrieval tier (`Tier 0` / `Tier 1`), daemon RSS memory, and active/suppressed/pending counts.
- **Global Pause (`⏸ Global Pause`)**: Immediately halts automatic recall injection globally (including on `AgentSuspended` resume) while keeping manual search and editing available.
- **Language Sync (`EN / 中文`)**: Automatically follows LSM WebUI's active display language (`localStorage.getItem('lang')`), supports manual override in one click, and updates all labels, tables, dialogs, and **`?` contextual help popovers** simultaneously.

### Tab 1 — Memories (CRUD, Lifecycle & Legacy Import)
1. **Search & Filter**: Query by keywords, bilingual aliases, ports, or paths; filter by `scope` (`personal` / `agent:<id>` / `*`), `machine`, and whether to include `Suppressed` or `Forgotten` entries.
2. **Create / CAS Edit (`+ Add Memory` / `Edit`)**: Add atomic facts with `kind`, `scope`, `topic`, `machine`, and comma-separated bilingual `search_terms` (3× BM25 weight boost). Edits use CAS optimistic locking (`expected_version`) and append an immutable revision record. API keys and sensitive tokens are blocked deterministically before write.
3. **Suppress vs. Forget**:
   - **Suppress / Restore**: Soft-hides a memory from automatic recall immediately without deleting data or history; reversible in one click.
   - **Forget**: Permanently marks the entry forgotten and writes a content-hash tombstone to `memory_tombstones` so backups or sync workers cannot resurrect it.
4. **Revision History (`History`)**: Inspect every `create`, `correct`, `suppress`, `restore`, and `forget` audit entry with timestamps and actors.
5. **Legacy Import (`Import Legacy Memories`)**: Appears automatically when legacy LSM `webui_memories` records exist; copies them non-destructively into `LSMMemory` with content deduplication.

### Tab 2 — Relation Graph (2D Canvas)
- **Visual Encoding**: Nodes are colored by `topic`; suppressed nodes appear faded with dashed outlines.
- **Interactive Inspection**: Drag nodes to rearrange, scroll to zoom, and click any node to highlight its **1-hop associative neighbors** and open the right-hand inspector.
- **Live Retrieval Links**: Use **Link** / **Unlink** in the inspector to manage `related`, `depends_on`, and `supersedes` edges. Edges actively participate in retrieval: matching any node automatically pulls its 1-hop linked neighbors into recall.

### Tab 3 — Inbox & Offline Consolidation
- **Staging Inbox**: Collect raw notes or drafts (`+ Add Draft` or via `memory_capture` from any client) before promoting them into long-term memory.
- **Zero-Extra-API-Key Consolidation**: Click **Copy Agent Consolidation Prompt** and paste it into any LSM-connected chat (WebUI, Gemini, Claude, or Codex). The Agent pulls pending drafts via `memory_consolidate_pull`, distills concise atomic memories with bilingual `search_terms` and relation edges, and commits atomically via `memory_consolidate_commit`.
- **Manual Triage**: Mark individual drafts as `consolidated` or `skipped` directly from the table.

### Tab 4 — Governance & Multi-Tier Retrieval
1. **Global Automation & Access Control**:
   - **WebUI Auto Split-Injection**: Master toggle for injecting `<lsm-memory-recall>` into WebUI projections (never mutates `messages[0]` or SQLite checkpoints).
   - **Include Newly Discovered Machines by Default**: Checked = open-by-default; unchecked = strict allowlist (new hosts stay excluded until explicitly enabled).
   - **External `/mcp` Default Access Role**: Enforced server-side (HTTP `403` on violation):
     - `read_write`: Full read/write management.
     - `read_capture`: Read + staging inbox drafts only (recommended for collaborative external agents).
     - `read_only`: Strictly read-only, preventing accidental deletion or prompt-injection writes.
   - **Offline Consolidation Mode & Interval**: Switch between `manual` (on-demand) and `auto` (scheduled interval policy, 5–1440 min).
2. **Per-Machine Include / Exclude & Roles**:
   - Automatically merges live online/offline LSM Workers (`remote_manager`) with machines recorded in memory entries.
   - Set **Excluded** on any host to hide its memories from search/injection and skip its inbox drafts non-destructively.
   - Assign a stricter per-machine role (`read_write` / `read_capture` / `read_only`) and per-machine `Auto-Consolidate` toggle.
3. **Multi-Tier Retrieval Engine (3-Mode Hot-Switch, No Restart Required)**:
   - **Tier 0 · Zero-Model Default (~14 MB RAM)**: Zero inputs required; 1-click activation. Also serves as the automatic 750 ms fail-closed fallback if any vector endpoint times out or goes offline.
   - **Tier 1A · Cloud Embeddings API (~14 MB Controller RAM)**: Enter `Endpoint URL`, `Model Name`, and `API Key` (saved `chmod 600`, never returned in plaintext), click **Test Connection & Dim**, **Save & Hot-Apply**, and **Backfill Missing Vectors** to populate vectors directly from the controller.
   - **Tier 1B · Worker / Local Model (`llama-server`)**:
     - **Step 1 (Offline Batch Backfill on Worker)**: Select an online LSM Worker (e.g., a GPU workstation) or `controller`, enter its local `llama-server` and `.gguf` model paths, and click **Run Backfill on Selected Worker**. The controller dispatches a self-contained stdlib task over LSM's encrypted Worker channel (zero SSH, zero secrets sent to Worker, zero `lsm_memory` install required on Worker), launches loopback `llama-server` on demand, uploads float32 vectors, and shuts down `llama-server` immediately (0 MB idle VRAM). An equivalent CLI command is also provided for large corpora.
     - **Step 2 (Query-Time Vector Source)**: Choose either **Offline Index Only (Keep Controller at Tier 0 · ~14 MB)** for 1C1G hosts, or **Local `llama-server` Hybrid RRF (4G+ Host)** with automatic vector fingerprint selection from cached fingerprints.

### Distributed Vector Backfill CLI

```bash
# On-demand workstation GPU via HTTPS /mcp (zero SSH)
export LOCAL_SHELL_MCP_OAUTH_ADMIN_PIN="<lsm-admin-pin>"
python3 -B -m lsm_memory.vector_sync \
  --lsm-url https://<your-lsm-domain> \
  --llama-server /path/to/llama-server \
  --model /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf

# Cloud OpenAI-compatible /v1/embeddings API
export OPENAI_API_KEY="<api-key>"
python3 -B -m lsm_memory.vector_sync \
  --lsm-url https://<your-lsm-domain> \
  --embed-url https://api.example.com/v1/embeddings \
  --embed-model BAAI/bge-m3
```

---

## Configuration Reference

All options are opt-in and follow LSM's `LOCAL_SHELL_MCP_*` environment variable convention:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED` | `0` (auto `1` if `ORIGIN` or `DB` set) | Master switch for WebUI bridge |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN` | `""` | Loopback daemon URL (`http://127.0.0.1:8775`); non-loopback rejected |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN_FILE` | `""` | Path to `chmod 600` Bearer token file |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_DB` | `""` | SQLite path for single-process embedded mode |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TIMEOUT_MS` | `750` | Fail-closed recall deadline (`50..3000` ms) |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_ITEMS` | `5` | Max recalled memories per turn (`1..10`) |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_CHARS` | `1200` | Max recall packet length (`200..4000` chars) |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOOLS` | `1` | Expose built-in `memory_vault_*` WebUI tools |
| `LSM_MEMORY_EMBED_URL` | `""` | Optional `/v1/embeddings` URL to activate Tier 1/2 Hybrid RRF |
| `LSM_MEMORY_EMBED_MODEL` | `"default"` | Logical model name matching `embedding_cache` |
| `LSM_MEMORY_EMBED_MODEL_SHA256` | `"api"` | Logical SHA-256 (or GGUF file SHA-256) for cross-platform reuse |
| `LSM_MEMORY_EMBED_API_KEY` | `""` | Optional Bearer key for cloud `/v1/embeddings` |
| `LSM_MEMORY_LSM_DB` | `""` | Optional LSM Controller SQLite path for read-only (`mode=ro`) consolidation |

---

## Upgrading LSM & Rollback

All WebUI hooks are marker-delimited, AST-verified before write, and backed up as `.lsm_memory_orig.bak`:

```bash
# Inspect patch state
/opt/local-shell-mcp/venv/bin/python -B -m lsm_memory.lsm_patcher status

# Re-apply cleanly after upgrading local-shell-mcp
/opt/local-shell-mcp/venv/bin/python -B -m lsm_memory.lsm_patcher apply
sudo systemctl restart local-shell-mcp.service

# Byte-exact rollback to original upstream LSM files
/opt/local-shell-mcp/venv/bin/python -B -m lsm_memory.lsm_patcher revert
sudo systemctl restart local-shell-mcp.service
```

See [docs/UPSTREAM_LSM_INTEGRATION.md](docs/UPSTREAM_LSM_INTEGRATION.md) for upstream `local-shell-mcp` PR file mapping and integration details.

## License

[MIT](LICENSE)
