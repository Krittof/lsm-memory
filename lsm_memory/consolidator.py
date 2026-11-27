"""Offline memory consolidation pipeline for LSMMemory (1C1G-safe, pure stdlib).

Supports two zero-lock-in consolidation workflows:
1. LSM-Connected Agent Work-Packet Mode (`build_consolidation_pack` -> Agent distills
   -> `commit_consolidation_pack`): any Agent connected to LSM (Gemini Web, Claude,
   Codex, LSM WebUI Agent) pulls pending inbox notes and/or read-only incremental LSM
   conversation turns, extracts atomic memories + bilingual `search_terms` aliases +
   typed relations, and commits atomically via MCP or HTTP with zero extra API keys.
2. Optional Unattended OpenAI-Compatible Distiller (`distill_pack_with_openai`):
   calls any OpenAI-compatible `/v1/chat/completions` endpoint using pure `urllib`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import urllib.parse
import urllib.request
from typing import Any

from .governance import is_machine_auto_consolidate, load_governance
from .lsm_bridge import strip_recall_blocks
from .policy import ContentBlockedError, assert_content_safe
from .retrieval import is_small_talk
from .store import KINDS, RELATION_TYPES, MemoryStore


def _is_content_blocked(text: str) -> bool:
    try:
        assert_content_safe(text)
        return False
    except ContentBlockedError:
        return True

CONSOLIDATION_PROMPT = (
    "Distill durable, self-contained atomic memories from the raw notes/turns below. "
    "NEVER store secrets, tokens, passwords, or transient chitchat. "
    "For each memory, supply concise bilingual `search_terms` aliases (Chinese + English "
    "keywords/abbreviations) so zero-model 1C1G hosts achieve high recall. "
    "Return JSON: {\"memories\": [{\"content\": ..., \"kind\": \"episode|user_profile|persona|transient\", "
    "\"topic\": ..., \"search_terms\": [...]}], "
    "\"relations\": [{\"source_idx\": 0, \"target_idx\": 1, \"relation_type\": \"related|supersedes|depends_on\"}], "
    "\"inbox_ids\": [...], \"skip_inbox_ids\": [...]}"
)


def _extract_lsm_message_text(text_col: Any, content_col: Any) -> str:
    if isinstance(text_col, str) and text_col.strip():
        return strip_recall_blocks(text_col).strip()
    if isinstance(content_col, str) and content_col.strip():
        try:
            parsed = json.loads(content_col)
        except json.JSONDecodeError:
            return strip_recall_blocks(content_col).strip()
        if isinstance(parsed, list):
            parts = []
            for p in parsed:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            return strip_recall_blocks("\n".join(parts)).strip()
    return ""


def _lsm_cursor_key(lsm_db_path: str | Path) -> str:
    digest = hashlib.sha256(str(Path(lsm_db_path).resolve()).encode("utf-8")).hexdigest()[:12]
    return f"consolidator_lsm_{digest}"


def collect_lsm_turns(
    lsm_db_path: str | Path,
    *,
    since_us: int = 0,
    limit: int = 40,
    max_chars_per_turn: int = 600,
) -> dict[str, Any]:
    """Read incremental user/assistant turns from LSM Controller SQLite strictly read-only."""
    path = Path(lsm_db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LSM SQLite DB not found: {path}")
    uri = path.as_uri() + "?mode=ro"
    lim = max(1, min(int(limit), 200))
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "messages" not in tables:
            return {"turns": [], "next_cursor_us": since_us, "scanned": 0}
        has_convos = "conversations" in tables
        if has_convos:
            sql = (
                "SELECT m.id, m.conversation_id, m.role, m.text, m.content, m.created_at, "
                "c.title AS conversation_title "
                "FROM messages m LEFT JOIN conversations c ON c.id = m.conversation_id "
                "WHERE m.created_at > ? AND m.role IN ('user', 'assistant') "
                "ORDER BY m.created_at ASC, m.id ASC LIMIT ?"
            )
        else:
            sql = (
                "SELECT id, conversation_id, role, text, content, created_at, '' AS conversation_title "
                "FROM messages "
                "WHERE created_at > ? AND role IN ('user', 'assistant') "
                "ORDER BY created_at ASC, id ASC LIMIT ?"
            )
        rows = conn.execute(sql, (int(since_us), lim)).fetchall()
    finally:
        conn.close()

    turns: list[dict[str, Any]] = []
    max_ts = int(since_us)
    skipped_secret = 0
    skipped_trivial = 0
    for r in rows:
        ts = int(r["created_at"] or 0)
        if ts > max_ts:
            max_ts = ts
        body = _extract_lsm_message_text(r["text"], r["content"])
        if not body or is_small_talk(body):
            skipped_trivial += 1
            continue
        if _is_content_blocked(body):
            skipped_secret += 1
            continue
        if len(body) > max_chars_per_turn:
            body = body[: max_chars_per_turn - 3] + "..."
        turns.append({
            "message_id": str(r["id"]),
            "conversation_id": str(r["conversation_id"] or ""),
            "conversation_title": str(r["conversation_title"] or ""),
            "role": str(r["role"]),
            "text": body,
            "created_at_us": ts,
        })
    return {
        "turns": turns,
        "scanned": len(rows),
        "skipped_secret": skipped_secret,
        "skipped_trivial": skipped_trivial,
        "next_cursor_us": max_ts,
    }


def build_consolidation_pack(
    store: MemoryStore,
    *,
    scope: str = "personal",
    inbox_limit: int = 30,
    lsm_db_path: str | Path | None = None,
    lsm_limit: int = 40,
) -> dict[str, Any]:
    """Build a bounded, secret-scrubbed consolidation work packet for an Agent or LLM."""
    gov = load_governance(store)
    inbox_rows = store.list_inbox(status="pending", scope=scope, limit=inbox_limit)
    safe_inbox: list[dict[str, Any]] = []
    blocked_inbox_ids: list[int] = []
    skipped_machine_inbox = 0
    for row in inbox_rows:
        if not is_machine_auto_consolidate(gov, row.get("machine")):
            skipped_machine_inbox += 1
            continue
        if _is_content_blocked(str(row.get("content") or "")):
            blocked_inbox_ids.append(int(row["id"]))
            continue
        safe_inbox.append({
            "id": int(row["id"]),
            "content": row["content"],
            "source_client": row["source_client"],
            "machine": row["machine"],
            "session_ref": row["session_ref"],
            "created_at": row["created_at"],
        })

    lsm_section: dict[str, Any] = {
        "cursor_key": "",
        "prev_cursor_us": 0,
        "next_cursor_us": 0,
        "turns": [],
    }
    if lsm_db_path:
        ckey = _lsm_cursor_key(lsm_db_path)
        prev_raw = store.get_meta_value(ckey, "0")
        try:
            prev_us = int(prev_raw)
        except ValueError:
            prev_us = 0
        collected = collect_lsm_turns(lsm_db_path, since_us=prev_us, limit=lsm_limit)
        lsm_section = {
            "cursor_key": ckey,
            "prev_cursor_us": prev_us,
            "next_cursor_us": collected["next_cursor_us"],
            "scanned": collected["scanned"],
            "turns": collected["turns"],
        }

    existing_topics = sorted({
        str(m.get("topic") or "").strip()
        for m in store.list_memories(scope=scope, limit=200)
        if str(m.get("topic") or "").strip()
    })[:50]

    return {
        "scope": scope,
        "global_revision": store.global_revision(),
        "instructions": CONSOLIDATION_PROMPT,
        "existing_topics": existing_topics,
        "inbox_events": safe_inbox,
        "blocked_inbox_ids": blocked_inbox_ids,
        "lsm_conversations": lsm_section,
        "has_work": bool(safe_inbox or lsm_section["turns"]),
    }


def commit_consolidation_pack(
    store: MemoryStore,
    payload: dict[str, Any],
    *,
    default_scope: str = "personal",
    actor: str = "consolidator",
) -> dict[str, Any]:
    """Atomically commit distilled memories, relations, inbox marks, and LSM cursor."""
    if not isinstance(payload, dict):
        raise ValueError("consolidation commit payload must be a JSON object")

    memories_in = payload.get("memories") or []
    relations_in = payload.get("relations") or []
    inbox_ids = payload.get("inbox_ids") or []
    skip_inbox_ids = payload.get("skip_inbox_ids") or []
    cursor_key = str(payload.get("cursor_key") or "").strip()
    next_cursor_us = payload.get("next_cursor_us")

    created_ids: list[int] = []
    deduplicated_ids: list[int] = []
    rejected: list[dict[str, Any]] = []
    idx_to_id: dict[int, int] = {}

    for idx, item in enumerate(memories_in):
        if not isinstance(item, dict):
            rejected.append({"index": idx, "reason": "not_an_object"})
            continue
        content = str(item.get("content") or "").strip()
        terms = item.get("search_terms") if isinstance(item.get("search_terms"), list) else []
        terms = [str(t).strip() for t in terms if str(t).strip()]
        kind = str(item.get("kind") or "episode").strip()
        if kind not in KINDS:
            kind = "episode"
        scope = str(item.get("scope") or default_scope).strip() or default_scope
        topic = str(item.get("topic") or "").strip()
        source_client = str(item.get("source_client") or "consolidator").strip() or "consolidator"
        machine = str(item.get("machine") or "controller").strip() or "controller"
        session_ref = str(item.get("session_ref") or "").strip()

        try:
            assert_content_safe(content, terms)
            saved = store.add_memory(
                content,
                kind=kind,
                scope=scope,
                search_terms=terms,
                topic=topic,
                source_client=source_client,
                machine=machine,
                session_ref=session_ref,
                actor=actor,
            )
            mid = int(saved["id"])
            idx_to_id[idx] = mid
            if saved.get("deduplicated"):
                deduplicated_ids.append(mid)
            else:
                created_ids.append(mid)
        except (ContentBlockedError, ValueError, Exception) as exc:
            rejected.append({"index": idx, "error": type(exc).__name__, "message": str(exc)})

    relations_added = 0
    for rel in relations_in:
        if not isinstance(rel, dict):
            continue
        src_id = rel.get("source_id")
        dst_id = rel.get("target_id")
        if src_id is None and isinstance(rel.get("source_idx"), int):
            src_id = idx_to_id.get(rel["source_idx"])
        if dst_id is None and isinstance(rel.get("target_idx"), int):
            dst_id = idx_to_id.get(rel["target_idx"])
        rel_type = str(rel.get("relation_type") or "related").strip()
        if rel_type not in RELATION_TYPES:
            rel_type = "related"
        if isinstance(src_id, int) and isinstance(dst_id, int) and src_id != dst_id:
            try:
                store.add_relation(src_id, dst_id, rel_type, weight=float(rel.get("weight", 1.0)))
                relations_added += 1
            except Exception:
                pass

    marked_consolidated = store.mark_inbox([int(i) for i in inbox_ids if isinstance(i, int)], "consolidated")["updated"]
    marked_skipped = store.mark_inbox([int(i) for i in skip_inbox_ids if isinstance(i, int)], "skipped")["updated"]

    cursor_saved = False
    if cursor_key and isinstance(next_cursor_us, int) and next_cursor_us > 0:
        store.set_meta_value(cursor_key, str(next_cursor_us))
        cursor_saved = True

    return {
        "created_ids": created_ids,
        "deduplicated_ids": deduplicated_ids,
        "rejected": rejected,
        "relations_added": relations_added,
        "inbox_consolidated": marked_consolidated,
        "inbox_skipped": marked_skipped,
        "cursor_saved": cursor_saved,
        "global_revision": store.global_revision(),
    }


def distill_pack_with_openai(
    pack: dict[str, Any],
    *,
    base_url: str,
    model: str,
    api_key: str = "",
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Optional unattended LLM distiller via any OpenAI-compatible /v1/chat/completions API."""
    clean = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(clean)
    if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Plain HTTP chat endpoint must be loopback; use HTTPS for remote APIs")
    if not clean.endswith("/chat/completions"):
        clean = (clean + "/chat/completions") if clean.endswith("/v1") else (clean + "/v1/chat/completions")
    user_payload = json.dumps(
        {
            "scope": pack.get("scope"),
            "existing_topics": pack.get("existing_topics", []),
            "inbox_events": pack.get("inbox_events", []),
            "lsm_turns": (pack.get("lsm_conversations") or {}).get("turns", []),
        },
        ensure_ascii=False,
    )
    body = json.dumps(
        {
            "model": model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": CONSOLIDATION_PROMPT},
                {"role": "user", "content": user_payload},
            ],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(clean, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    content = raw["choices"][0]["message"]["content"]
    distilled = json.loads(content)
    if not isinstance(distilled, dict):
        raise ValueError("LLM consolidation response must be a JSON object")
    lsm_meta = pack.get("lsm_conversations") or {}
    if lsm_meta.get("cursor_key") and lsm_meta.get("next_cursor_us"):
        distilled.setdefault("cursor_key", lsm_meta["cursor_key"])
        distilled.setdefault("next_cursor_us", lsm_meta["next_cursor_us"])
    if pack.get("blocked_inbox_ids"):
        distilled.setdefault("skip_inbox_ids", []).extend(pack["blocked_inbox_ids"])
    return distilled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LSMMemory offline consolidation CLI")
    parser.add_argument("--db", required=True, help="Path to LSMMemory SQLite DB")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export-pack")
    p_exp.add_argument("--scope", default="personal")
    p_exp.add_argument("--inbox-limit", type=int, default=30)
    p_exp.add_argument("--lsm-db", default=None, help="Optional LSM Controller SQLite DB (read-only)")
    p_exp.add_argument("--lsm-limit", type=int, default=40)

    p_com = sub.add_parser("commit-pack")
    p_com.add_argument("--scope", default="personal")
    p_com.add_argument("--file", default="-", help="JSON commit payload file or '-' for stdin")

    p_run = sub.add_parser("run-openai")
    p_run.add_argument("--scope", default="personal")
    p_run.add_argument("--lsm-db", default=None)
    p_run.add_argument("--base-url", required=True)
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--api-key-env", default="OPENAI_API_KEY")

    args = parser.parse_args(argv)
    store = MemoryStore(args.db)
    if args.cmd == "export-pack":
        out = build_consolidation_pack(
            store,
            scope=args.scope,
            inbox_limit=args.inbox_limit,
            lsm_db_path=args.lsm_db,
            lsm_limit=args.lsm_limit,
        )
    elif args.cmd == "commit-pack":
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
        out = commit_consolidation_pack(store, json.loads(raw), default_scope=args.scope)
    elif args.cmd == "run-openai":
        pack = build_consolidation_pack(store, scope=args.scope, lsm_db_path=args.lsm_db)
        if not pack["has_work"]:
            out = {"skipped": True, "reason": "no_pending_work"}
        else:
            api_key = os.environ.get(args.api_key_env, "")
            distilled = distill_pack_with_openai(pack, base_url=args.base_url, model=args.model, api_key=api_key)
            out = commit_consolidation_pack(store, distilled, default_scope=args.scope)
    else:
        raise SystemExit(2)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
