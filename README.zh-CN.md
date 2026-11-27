# LSMMemory

[![CI](https://github.com/Krittof/lsm-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/Krittof/lsm-memory/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(stdlib%20only)-brightgreen.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **[English](README.md) | [简体中文](README.zh-CN.md)**

**面向 [`local-shell-mcp` (LSM)](https://github.com/fwerkor/local-shell-mcp) 的多档自适应、零第三方依赖、跨网络长期记忆库与 WebUI 分体注入桥接组件。**

`LSMMemory` 完全基于 **Python 3.11+ 标准库**实现（无第三方运行时依赖），为 **LSM WebUI** 以及所有接入 LSM `/mcp` 的外部客户端（Gemini Web、Claude、Codex、Antigravity 等）提供统一、可审计、可挂起/回滚的长期记忆能力，在 1C1G 低配主机上常驻内存仅约 **14 MB**。

---

## 核心特性与兼容性

本组件在 100% 保留原版 LSM v6.0.3+ 行为的前提下扩展记忆能力，不修改原有 `webui_memories` 表结构、`/api/memories*` 接口、`set_memory`/`delete_memory` 工具及 `memory_protection.py`：

| 维度 | 原版 `webui_memories` | **LSMMemory (`v0.4.0`)** |
| :--- | :--- | :--- |
| **跨端共享** | 仅 LSM WebUI 可见 | LSM WebUI 与全部 `/mcp` 外部客户端共享同一记忆库 |
| **检索与排序** | 无排序，全量拼接进 System Prompt | **多档自适应引擎**：Tier 0 零模型（~14 MB）至 Tier 1/2 混合向量 RRF |
| **注入与缓存** | 直接追加至静态 System Prompt | 依托 `_project_model_messages()` **分体注入**：不改 `messages[0]`（保 Prompt Cache），不污染持久化历史 |
| **挂起与恢复** | `AgentSuspended` 恢复时沿用旧快照 | 每轮实时重算并剥离陈旧标签；挂起期间抑制/遗忘/暂停后恢复 **0 泄漏** |
| **生命周期与权限治理** | 覆盖写 / 硬删除 | CAS 乐观锁、修订历史、软挂起/恢复、防复活墓碑、**按机器选入/剔除与权限档位（`read_write` / `read_capture` / `read_only`）**、全局暂停 |
| **WebUI 管理台** | 仅平面键值弹窗 | **零依赖 LSM 风格 `/memories/vault` 控制台**（复用 LSM 登录态）：记忆增删改查、**2D Canvas 关系星图**、收件箱、机器与权限开关、Tier 1A 热配置与原版记忆导入 |
| **离线整理** | 仅支持每轮额外调用模型 | 只读（`mode=ro`）增量采集 LSM 会话 + **MCP 整理工单工具**，可由已连接 Agent 直接提炼，无需额外 API Key |
| **资源占用** | 轻量 | 纯 Python 3.11+ stdlib（**0 第三方依赖**），常驻 ~14–16 MB，1,000 条记忆热检索 `< 40 ms` |

---

## 运行机制

```text
  [外部 MCP 客户端: Gemini Web / Claude / Codex / CLI]        [LSM WebUI (AgentRunner)]
                            │  HTTPS /mcp (OAuth 2.1)                    │  _project_model_messages()
                            ▼                                            ▼
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │                        LSM 主控主机 (1C1G+ Linux / ARM64 / x86_64)                      │
  │                                                                                        │
  │   LSM Nested MCP (`lsm_memory`, stdio 按需拉起) ◄──► LSMMemory 核心 (Daemon / Embedded) │
  │                                                      • SQLite WAL 单写入者成品库        │
  │   LSM 主控 SQLite (严格只读 `mode=ro`) ────────────► • Tier 0 / Tier 1-2 自适应检索引擎 │
  │                                                      • 紧凑 `array('f')` 有界向量 LRU   │
  └──────────────────────────┬───────────────────────────────────────────┬─────────────────┘
                             │ 拉取 / 提交整理包                          │ 拉取缺失 / 回推向量
                             │ (HTTPS /mcp，免额外 API Key)              │ (HTTPS /mcp，免 SSH)
                             ▼                                           ▼
              [LSM 已连接的任意 Agent]                     [本地电脑 GPU (`llama-server`)
              提炼原子记忆、双语别名与关系边                或云端 `/v1/embeddings` API]
                                                          (按需启停，空闲 0 MB 显存)
```

### 1. 存储、一致性与生命周期（`store.py`、`policy.py`）
- **单写入者 SQLite WAL**：通过 `BEGIN IMMEDIATE` 事务与 CAS 版本乐观锁（`expected_version`）防止多客户端并发覆盖；同时保留修订记录表 `memory_revisions`。
- **挂起与遗忘语义分离**：
  - `suppress` / `restore`：软挂起单条记忆，不删除原始数据，立即移出召回并递增单调版本号 `global_revision`。
  - `forget`：标记遗忘并在 `memory_tombstones` 写入内容哈希墓碑，防止备份恢复或跨端同步导致已删记忆复活。
  - `global_pause`：全局暂停自动召回注入。
- **内容安全校验**：写入前通过确定性规则拦截常见密钥与敏感凭证（`sk-`、`Bearer `、私钥头等）；WebUI 侧同时联动 LSM 原生 `configured_filters` / `memory_protection`。
- **跨平台紧凑向量缓存**：`embedding_cache` 为维度无关结构，以逻辑模型指纹 `sha256(model_name + model_sha256 + dim)` 索引（与操作系统可执行文件哈希解耦，Windows GPU 算出的向量可直接供 Linux ARM64/x86_64 复用）；内存中使用有界 `array('f')` LRU 缓存（相比 Python `tuple[float]` 节省约 9 倍内存）。

### 2. 多档自适应检索（`retrieval.py`、`embedder.py`）
- **Tier 0（默认零模型档，~14 MB 内存）**：
  1. NFKC 规范化 + 中文 2-gram + 英文词项 BM25 打分。
  2. **`search_terms` 与 `topic` 3 倍加权**：将语义扩展前置到离线整理阶段，由 Agent 补全中英文同义词与缩写，在无模型环境下实现跨语种高召回。
  3. 精确短标识符直通通道：针对端口、IP、路径、版本号等短标识做精确子串加权。
  4. 子句级 MaxSim 局部重排：按句切分计算局部最高相似度，避免长记忆稀释相关性。
  5. **1-hop 关系图谱扩展**：沿 `related` / `depends_on` / `supersedes` 边自动拉取关联记忆。
  6. 保守 NLI 矛盾标记与闲聊短句跳过（`is_small_talk`）。
- **Tier 1 / Tier 2（可选混合向量 RRF 档）**：
  - 配置 `LSM_MEMORY_EMBED_URL`（任意 OpenAI 兼容 `/v1/embeddings` 接口，可接云端 API 或本机 `llama-server`）后自动激活。
  - 将 Tier 0 词法候选与预计算向量余弦相似度通过 **Reciprocal Rank Fusion（RRF，`k=60`）** 融合，并设置最低余弦阈值防止无关记忆误召回。
  - 严格执行 **750 ms 召回死线**：向量接口超时、离线或异常时自动无感降级回 Tier 0，不阻塞对话。

### 3. WebUI 分体注入与 `AgentSuspended` 挂起恢复校验（`lsm_bridge.py`、`lsm_patcher.py`）
- 挂接在 LSM 原生投影层 `AgentRunner._project_model_messages()`：
  - **不修改 `messages[0]`**：静态 System Prompt 与前缀保持字节级不变，完整保留模型商 Prompt Cache。
  - **不污染 Checkpoint**：临时召回块 `<lsm-memory-recall scope="..." revision="..." untrusted="true">` 仅贴附于当轮 `user` 消息的浅拷贝副本（兼容 OpenAI / Anthropic / Gemini 协议），绝不写回持久化会话历史。
  - **挂起恢复实时重算**：每次模型调用前（含 `AgentSuspended` 审批暂停恢复后）均重新查询活跃记忆并剥离任何陈旧 `<lsm-memory-recall>` 标签，确保挂起期间被 `suppress`、`forget` 或 `global_pause` 的记忆在恢复后绝不泄露。
  - 完整遵守 LSM 会话开关（`effective_features["memory"]`）、用户偏好（`user.memories`）、Agent 分区隔离（`personal` / `agent:<id>`）以及审批分级（只读工具免危险审批）。

### 4. 离线整理与分布式向量回推（`consolidator.py`、`vector_sync.py`）
- **离线整理**：各端可通过 `memory_capture` 写入 `inbox_events` 暂存区，主控亦可通过只读连接（`mode=ro`）配合持久化游标增量扫描 LSM 会话数据库。已连接 LSM 的任意 Agent 可调用 `memory_consolidate_pull` 拉取脱敏整理包，提炼原子记忆、双语别名与关系边后通过 `memory_consolidate_commit` 原子提交（亦可选用 `lsm-memory-consolidate run-openai` 走独立接口无人值守整理）。
- **分布式向量回推**：`lsm-memory-sync` 拉取指定逻辑指纹下缺失向量的记忆，调用云端 `/v1/embeddings` 或通过 `ManagedLlamaEmbedServer` 按需拉起本地 GPU `llama-server --embeddings`（批处理完成后立即退出释放全部显存），经由本地 SQLite、环回 HTTP 或 **HTTPS `/mcp` OAuth 2.1** 批量写回主控 `embedding_cache`（无需 SSH 私钥，也无需在 VPS 额外开放端口）。

---

## 三档检索模式对比

| 模式 | 提问（Query）向量来源 | 记忆库向量来源 | 主控内存 | 依赖条件 |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0（默认）** | 不算向量（词法 + 别名 + MaxSim + 关系图谱） | 离线 Agent 提炼双语别名与关系边 | **~14 MB** | 仅 Python 3.11+ stdlib（兼容 1C1G） |
| **Tier 1A（云端 API）** | OpenAI 兼容 `/v1/embeddings` API | 同款 `/v1/embeddings` API 批量补齐 | **~14 MB** | 云端 Embedding API Key（兼容 1C1G） |
| **Tier 1B（全私有离线）** | 主控本机 CPU 跑同款 GGUF `llama-server` | 工作站 GPU 跑同款 GGUF 走 HTTPS `/mcp` 回推 | ~750 MB（4G+ 主机） | 本地 GGUF 模型 + `llama-server` |

---

## 实测性能与 Token 收益

基于独立基准语料在 **ARM64 Linux 主控（Ubuntu 22.04 `aarch64`）** 与 **工作站 GPU（`Qwen3-Embedding-0.6B-Q8_0.gguf`）** 实测，对比原版 LSM `webui_memories` 全量灌入与 `LSMMemory` Tier 0 分体注入：

### 不同记忆库规模下的 Token 降幅与 ARM64 检索延迟

| 记忆库规模 | 原版全量灌入（约合 BPE Token/轮） | LSMMemory 均值（含闲聊跳过） | 单轮 Token 节省 | ARM64 热检索耗时 |
| :---: | :---: | :---: | :---: | :---: |
| **20 条** | ~982 | ~224 | **77.2%** | `4.42 ms` |
| **50 条** | ~2,448 | ~225 | **90.8%** | `7.93 ms` |
| **100 条** | ~4,890 | ~226 | **95.4%** | `14.08 ms` |
| **200 条** | **6,890**（双语真实语料） | **100.0**（有效问 `140.8` / 闲聊 `0`） | **98.5%** | `25.36 ms` |
| **500 条** | ~24,526 | ~226 | **99.1%** | `62.13 ms` |
| **1,000 条** | ~49,073（原版常触发上限） | ~217 | **99.6%** | `121.26 ms` |

### 20 轮会话与 30-Query 召回质量汇总（200 条双语库 · Tier 0 零模型）

| 指标 | 原版全量灌入 | **LSMMemory（Tier 0）** |
| :--- | :---: | :---: |
| **20 轮累计记忆 Token**（15 轮提问 + 5 轮闲聊） | `137,800` | **`2,000`**（单场省 **`135,800 tok`**） |
| **Prompt Cache 前缀失效次数**（中途修改/挂起 2 条记忆） | `2 次`（额外重算约 `29,780 tok`） | **`0 次`**（`messages[0]` 字节不变） |
| **Recall@5 / MRR**（25 道有效题：精确标识 + 双语别名 + 1-hop） | —（上下文噪声比 `99.5%`） | **`96.0%` / `0.940`**（均注 `2.92` 条/轮） |
| **1-hop 关系邻居联想 Recall@5**（5 组关联对） | — | **`5/5 (100%)`** |
| **闲聊跳过率** / **`AgentSuspended` 恢复泄漏** | `0%` / 沿用旧快照 | **`100% (0 tok)` / `0 泄漏`** |
| **主控环回延迟与常驻内存（ARM64）** | — | 均值 **`2.22 ms`**，cgroup **`~14.7 MB`** |
| **工作站 GPU 批量回推（`Qwen3-Embedding-0.6B`，1024 维）** | — | **`152.8 条/秒`**（50 条 `327 ms`，空闲 `0 MB` 显存） |

---

## 安装与部署

### 1. 克隆并运行测试

```bash
git clone https://github.com/Krittof/lsm-memory.git
cd lsm-memory
python3 -B -m unittest discover -s tests -t . -v
```

### 2. 部署到 LSM 主控（Linux ARM64 / x86_64）

```bash
# 守护进程模式（推荐，供 WebUI 与 /mcp 共享单写入者，systemd MemoryMax=128M）
sudo bash deploy/install.sh --mode daemon --patch-lsm
sudo systemctl restart local-shell-mcp.service

# 单进程内嵌模式（无额外守护进程，直接嵌入 LSM 进程）
sudo bash deploy/install.sh --mode embedded --patch-lsm
sudo systemctl restart local-shell-mcp.service
```

### 3. 注册到 LSM Nested MCP（`mcp_manage`）

以 `stdio` 方式注册 `lsm_memory`（按需拉起，空闲 0 MB 内存，暴露 12 个工具）：

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

## 工具接口与命令行

### MCP 工具（`lsm_memory:*`，共 12 个）
- **核心操作**：`memory_search`、`memory_add`、`memory_correct`、`memory_suppress`、`memory_forget`、`memory_relate`、`memory_capture`、`memory_status`
- **离线整理与向量同步**：`memory_consolidate_pull`、`memory_consolidate_commit`、`memory_vectors_missing`、`memory_vectors_upsert`

### LSM WebUI 管理台与内置工具
- **WebUI 管理控制台（`/memories/vault`）**：挂载在 LSM 原生登录鉴权下（`https://<你的LSM域名>/memories/vault`），在 LSM 页面右上角提供 **`● 记忆库` / `● Memory Vault`** 快捷入口，自动跟随 LSM WebUI 语言设置（`localStorage.lang`，支持中/英一键切换），并在全部复杂配置与策略旁提供中英双语 **`?` 说明浮窗**。
- **WebUI 内置 Agent 工具（共 6 个）**：`memory_vault_search`（只读）、`memory_vault_status`（只读）、`memory_vault_remember`、`memory_vault_suppress`、`memory_vault_forget`、`memory_vault_capture`

---

## WebUI 控制台操作手册

完成 `--patch-lsm` 部署并重启 `local-shell-mcp.service` 后，在 LSM WebUI 页面右上角点击 **`● 记忆库`**（英文界面显示 **`● Memory Vault`**），或直接访问 `https://<你的LSM域名>/memories/vault` 即可进入控制台。

### 顶栏状态与语言同步
- **状态徽章**：实时展示组件版本、当前生效检索档位（`Tier 0` / `Tier 1`）、守护进程常驻内存（RSS）以及活跃/挂起/待整理条数。
- **全局暂停（`⏸ 全局暂停`）**：立即暂停全网自动记忆召回与 WebUI 分体注入（含 `AgentSuspended` 挂起恢复时），但仍可进行手动检索、审计与编辑。
- **语言同步（`中文 / EN`）**：自动读取 LSM WebUI 当前语言（`localStorage.getItem('lang')`），同时支持在控制台内手动切换；切换后全部标签、表格、弹窗及 **`?` 策略说明浮窗**同步更新。

### 标签页 1：记忆管理（增删改查、生命周期与原版导入）
1. **检索与筛选**：支持按关键词、中英文别名、端口、路径检索，并可按作用域（`personal` / `agent:<id>` / 全部）、来源机器及状态（是否显示“已挂起 / 已遗忘”）筛选。
2. **新建与 CAS 编辑（`+ 新建记忆` / `编辑`）**：录入原子事实并指定 `kind`、`scope`、`topic`、`machine` 与逗号分隔的中英文 `search_terms`（检索享 3 倍权重加成）。编辑采用 CAS 乐观锁（`expected_version`）并自动写入不可变修订记录；敏感密钥与凭证在入库前会被确定性拦截。
3. **软挂起与永久遗忘**：
   - **挂起 / 恢复**：将记忆临时移出自动召回，保留原始数据与修订历史，随时可一键恢复。
   - **遗忘**：永久标记删除并在 `memory_tombstones` 写入内容哈希墓碑，防止旧备份或同步脚本将其复活。
4. **修订历史（`历史`）**：查看每条记忆历次 `create`、`correct`、`suppress`、`restore`、`forget` 的版本、操作人与时间。
5. **原版记忆无损导入（`导入原版记忆`）**：检测到原版 LSM `webui_memories` 有数据时自动显示，点击即可按内容去重无损复制至 `LSMMemory`（不修改、不删除原版数据）。

### 标签页 2：关系星图（2D Canvas）
- **视觉编码**：节点按 `topic` 自动着色，已挂起节点以虚线半透明样式区分。
- **交互与检视**：支持拖拽节点、滚轮缩放；点击任意节点即可高亮其 **1-hop 关联邻居**，并在右侧面板直接编辑、挂起、查看历史。
- **维护检索关系边**：在右侧面板通过 **连边 / 删边** 管理 `related`、`depends_on`、`supersedes` 关系边。关系边直接参与检索引擎——任一节点命中时，其 1-hop 关联记忆会被自动联想召回。

### 标签页 3：收件箱与离线整理
- **暂存收件箱（Staging Inbox）**：各端通过 `memory_capture` 或控制台 **`+ 写入草稿`** 提交的原始碎片会先停留在收件箱，避免直接污染长期记忆库。
- **零额外 API Key 整理流程**：点击 **`复制 Agent 整理指令`**，粘贴到任意已接入 LSM 的对话（WebUI、Gemini、Claude 或 Codex），Agent 会调用 `memory_consolidate_pull` 拉取待整理草稿，提炼原子记忆、补全中英文 `search_terms` 与关系边后，通过 `memory_consolidate_commit` 原子入库。
- **人工处理**：也可直接在表格中将单条草稿标记为 `consolidated`（已整理）或 `skipped`（跳过）。

### 标签页 4：机器权限与多档配置
1. **自动化与权限总控**：
   - **开启 WebUI 自动分体注入**：控制是否在 WebUI 投影层注入 `<lsm-memory-recall>`（不改动 `messages[0]` 静态前缀，也不污染 SQLite Checkpoint）。
   - **新发现机器默认纳入**：勾选表示默认开放；取消勾选即进入严格白名单模式，新接入机器在手动设为“纳入”前不会参与召回与整理。
   - **外部 `/mcp` 默认权限档位**（服务端强制校验，越权返回 HTTP `403`）：
     - `read_write`：完全读写与管理权限。
     - `read_capture`：只读 + 允许提交草稿至收件箱（推荐用于多端协作）。
     - `read_only`：纯只读，防止外部客户端误删或提示词注入污染记忆库。
   - **离线整理模式与间隔**：在 `manual`（手动 / Agent 按需触发）与 `auto`（定时策略，间隔 5–1440 分钟）之间切换。
2. **按机器选入 / 剔除与权限治理**：
   - 自动汇总当前在线/离线的 LSM Worker（来自 `remote_manager`）及历史写入过记忆的机器。
   - 将单台机器设为 **剔除（Excluded）** 后，其记忆立即从全网检索与自动注入中隐藏，收件箱草稿也会被整理流程跳过（不删原数据，随时可恢复纳入）。
   - 支持按单机单独下调权限档位（取全局与单机中更严格的一方执行）及开关单机自动整理。
3. **多档检索引擎配置（三模式卡片热切换，无需重启服务）**：
   - **Tier 0 · 零模型默认（约 14 MB 内存）**：零输入项，一键启用；同时作为所有向量模式的 750 ms 超时安全降级兜底。
   - **Tier 1A · 云端向量 API（主控仍约 14 MB）**：仅需填写 `Endpoint URL`、`Model Name`、`API Key`（`chmod 600` 落盘且永不明文回显），依次点击 **`测试连通与维度`** → **`保存并立即热生效`** → **`一键补齐缺失向量`** 即可在浏览器内完成端到端配置与向量补齐。
   - **Tier 1B · Worker / 本地模型（`llama-server`）**：
     - **步骤 ①（在线 Worker 离线批量算向量）**：下拉选择一台在线 LSM Worker（如带独显的工作站）或 `controller`，填写该机器上的 `llama-server` 与 `.gguf` 模型路径，点击 **`一键在所选 Worker 补齐向量`**。主控通过 LSM 既有加密 Worker 通道下发纯标准库脚本（无需 SSH、无需在 Worker 预装 `lsm_memory`、不下发任何主控密钥或 PIN），按需拉起环回 `llama-server` 算完向量回传后立即退出（空闲 0 MB 显存）；下方同时折叠提供等效命令行供超大语料手动执行。
     - **步骤 ②（主控提问阶段向量来源）**：1C1G 低配主控可选择 **仅保留离线向量库（主控维持 Tier 0 · ~14 MB）**；4G+ 主控若本机常驻同款 GGUF `llama-server`，可选择 **主控本机 llama-server 混合检索**，并直接从下拉框自动匹配已有向量缓存指纹（无需手动复制 64 位 SHA-256）。

### 分布式向量回推命令

```bash
# 工作站 GPU 按需算向量，通过 HTTPS /mcp 远程回推（无需 SSH）
export LOCAL_SHELL_MCP_OAUTH_ADMIN_PIN="<你的LSM管理员PIN>"
python3 -B -m lsm_memory.vector_sync \
  --lsm-url https://<你的LSM域名> \
  --llama-server /path/to/llama-server \
  --model /path/to/Qwen3-Embedding-0.6B-Q8_0.gguf

# 调用云端 OpenAI 兼容 /v1/embeddings API
export OPENAI_API_KEY="<API_KEY>"
python3 -B -m lsm_memory.vector_sync \
  --lsm-url https://<你的LSM域名> \
  --embed-url https://api.example.com/v1/embeddings \
  --embed-model BAAI/bge-m3
```

---

## 配置项参考（环境变量）

所有配置默认关闭、显式启用，遵循 LSM `LOCAL_SHELL_MCP_*` 命名规范：

| 环境变量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED` | `0`（配置 ORIGIN/DB 时自动为 `1`） | WebUI 桥接总开关 |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN` | 空 | 守护进程地址（仅允许环回 `http://127.0.0.1:8775`） |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN_FILE` | 空 | Bearer Token 文件路径（`chmod 600`） |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_DB` | 空 | 单进程内嵌模式 SQLite 路径 |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TIMEOUT_MS` | `750` | 召回超时死线（`50..3000` ms） |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_ITEMS` | `5` | 每轮最大注入条数（`1..10`） |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_MAX_CHARS` | `1200` | 单次注入总字符上限（`200..4000`） |
| `LOCAL_SHELL_MCP_MEMORY_VAULT_TOOLS` | `1` | 是否暴露 WebUI 内置 `memory_vault_*` 工具 |
| `LSM_MEMORY_EMBED_URL` | 空 | 可选 `/v1/embeddings` 地址，配置后启用 Tier 1/2 混合 RRF |
| `LSM_MEMORY_EMBED_MODEL` | `default` | 逻辑模型名称（需与向量缓存一致） |
| `LSM_MEMORY_EMBED_MODEL_SHA256` | `api` | 逻辑模型哈希（云端 API 填 `api`，本地 GGUF 填文件 SHA-256） |
| `LSM_MEMORY_EMBED_API_KEY` | 空 | 云端 `/v1/embeddings` Bearer 密钥 |
| `LSM_MEMORY_LSM_DB` | 空 | 可选 LSM 主控 SQLite 路径（只读增量采集历史会话） |

---

## 升级 LSM 与回滚

WebUI 挂钩全部带显式标记块、写入前经 AST 语法检查，并自动保存 `.lsm_memory_orig.bak` 原文件备份：

```bash
# 查看补丁状态
/opt/local-shell-mcp/venv/bin/python -B -m lsm_memory.lsm_patcher status

# 升级 local-shell-mcp wheel 后幂等重打补丁
/opt/local-shell-mcp/venv/bin/python -B -m lsm_memory.lsm_patcher apply
sudo systemctl restart local-shell-mcp.service

# 字节级还原至原版 LSM 文件
/opt/local-shell-mcp/venv/bin/python -B -m lsm_memory.lsm_patcher revert
sudo systemctl restart local-shell-mcp.service
```

面向 `local-shell-mcp` 上游仓库合并的模块映射详见 [docs/UPSTREAM_LSM_INTEGRATION.md](docs/UPSTREAM_LSM_INTEGRATION.md)。

## License

[MIT](LICENSE)
