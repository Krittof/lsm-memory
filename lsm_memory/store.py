"""Pure-stdlib SQLite WAL memory store optimized for 1C1G controllers.

Features:
- SQLite WAL + BEGIN IMMEDIATE transactions + CAS optimistic locking (version)
- Full revision audit log + zero-resurrection tombstones
- Soft suppression (suspend/restore individual memories without deletion)
- Cross-device/cross-client provenance (source_client, machine, session_ref)
- Offline semantic expansion fields (search_terms aliases, topic)
- 1-hop typed relations (related, supersedes, depends_on, contradicts, elaborates)
- Dimension-agnostic embedding cache with compact array('f') LRU (~4 KB / 1024-dim)
- Monotonic global_revision + global_pause kill-switch for suspend/resume checks
- Staging inbox_events table for offline Agent consolidation
"""

from __future__ import annotations

from array import array
from collections import OrderedDict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import uuid
from typing import Any

SCHEMA_VERSION = 1
KINDS = ("persona", "user_profile", "episode", "transient")
RELATION_TYPES = ("related", "supersedes", "depends_on", "contradicts", "elaborates")
INBOX_STATUSES = ("pending", "claimed", "consolidated", "skipped")

MAX_CONTENT = 20_000
MAX_SCOPE = 128
MAX_LABEL = 128
MAX_SEARCH_TERMS = 48
MAX_TERM_LEN = 80
MAX_VECTOR_DIM = 4096
MIN_VECTOR_DIM = 16
DEFAULT_VECTOR_CACHE_SIZE = 256  # ~1 MB RAM for 1024-dim array('f') on 1C1G hosts

_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS library_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT NOT NULL DEFAULT 'personal',
        kind TEXT NOT NULL CHECK(kind IN ('persona', 'user_profile', 'episode', 'transient')),
        content TEXT NOT NULL CHECK(length(content) BETWEEN 1 AND 20000),
        search_terms TEXT NOT NULL DEFAULT '[]',
        topic TEXT NOT NULL DEFAULT '',
        source_client TEXT NOT NULL DEFAULT 'manual',
        machine TEXT NOT NULL DEFAULT 'local',
        session_ref TEXT NOT NULL DEFAULT '',
        content_sha256 TEXT NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'forgotten')),
        suppressed INTEGER NOT NULL DEFAULT 0 CHECK(suppressed IN (0, 1)),
        suppressed_revision INTEGER NOT NULL DEFAULT 0,
        valid_from REAL,
        valid_until REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memories_scope_active
        ON memories(scope, status, suppressed, updated_at DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_memories_scope_sha
        ON memories(scope, content_sha256)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_revisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        memory_id INTEGER NOT NULL,
        version INTEGER NOT NULL,
        action TEXT NOT NULL,
        content TEXT NOT NULL,
        search_terms TEXT NOT NULL,
        actor TEXT NOT NULL DEFAULT 'system',
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_revisions_memory
        ON memory_revisions(memory_id, id DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_tombstones (
        memory_id INTEGER PRIMARY KEY,
        scope TEXT NOT NULL,
        content_sha256 TEXT NOT NULL,
        reason TEXT NOT NULL DEFAULT '',
        deleted_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_tombstones_scope_sha
        ON memory_tombstones(scope, content_sha256)
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL REFERENCES memories(id),
        target_id INTEGER NOT NULL REFERENCES memories(id),
        relation_type TEXT NOT NULL CHECK(
            relation_type IN ('related', 'supersedes', 'depends_on', 'contradicts', 'elaborates')
        ),
        weight REAL NOT NULL DEFAULT 1.0,
        confirmed INTEGER NOT NULL DEFAULT 1 CHECK(confirmed IN (0, 1)),
        created_at TEXT NOT NULL,
        UNIQUE(source_id, target_id, relation_type)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        memory_id INTEGER NOT NULL REFERENCES memories(id),
        content_version INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        dim INTEGER NOT NULL CHECK(dim BETWEEN 16 AND 4096),
        vector BLOB NOT NULL CHECK(length(vector) = dim * 4),
        updated_at TEXT NOT NULL,
        PRIMARY KEY(memory_id, fingerprint)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_embedding_fp
        ON embedding_cache(fingerprint, memory_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS inbox_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scope TEXT NOT NULL DEFAULT 'personal',
        source_client TEXT NOT NULL DEFAULT 'manual',
        machine TEXT NOT NULL DEFAULT 'local',
        session_ref TEXT NOT NULL DEFAULT '',
        event_key TEXT NOT NULL,
        content TEXT NOT NULL CHECK(length(content) BETWEEN 1 AND 20000),
        metadata TEXT NOT NULL DEFAULT '{}',
        content_sha256 TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending' CHECK(
            status IN ('pending', 'claimed', 'consolidated', 'skipped')
        ),
        claim_token TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(scope, source_client, session_ref, event_key)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_inbox_status
        ON inbox_events(status, scope, id ASC)
    """,
)


class StoreError(RuntimeError):
    """Base store error."""


class ConflictError(StoreError):
    """Raised on CAS version conflict or tombstone resurrection block."""


class NotFoundError(StoreError):
    """Raised when a memory or relation does not exist."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def validate_label(value: Any, name: str, *, allow_empty: bool = False, max_len: int = MAX_LABEL) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    cleaned = value.strip()
    if not allow_empty and not cleaned:
        raise ValueError(f"{name} must not be empty")
    if "\x00" in cleaned or len(cleaned) > max_len:
        raise ValueError(f"{name} is invalid or exceeds {max_len} chars")
    return cleaned


def validate_content(value: Any, name: str = "content") -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty string without NUL")
    cleaned = value.strip()
    if len(cleaned) > MAX_CONTENT:
        raise ValueError(f"{name} exceeds {MAX_CONTENT} characters")
    return cleaned


def validate_kind(value: Any) -> str:
    if not isinstance(value, str) or value not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    return value


def validate_search_terms(terms: Any) -> list[str]:
    if terms is None:
        return []
    if isinstance(terms, str):
        terms = [part.strip() for part in terms.split(",") if part.strip()]
    if not isinstance(terms, (list, tuple)):
        raise ValueError("search_terms must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        if not isinstance(raw, str):
            raise ValueError("each search_term must be a string")
        cleaned = raw.strip()
        if not cleaned or "\x00" in cleaned:
            continue
        if len(cleaned) > MAX_TERM_LEN:
            cleaned = cleaned[:MAX_TERM_LEN]
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            result.append(cleaned)
        if len(result) >= MAX_SEARCH_TERMS:
            break
    return result


def encode_vector(values: Any) -> tuple[int, bytes, array]:
    """Validate, L2-normalize, and pack a float vector into compact little-endian float32."""
    if isinstance(values, (str, bytes, dict)) or not hasattr(values, "__len__"):
        raise ValueError("vector must be a numeric sequence")
    dim = len(values)
    if not (MIN_VECTOR_DIM <= dim <= MAX_VECTOR_DIM):
        raise ValueError(f"vector dim must be between {MIN_VECTOR_DIM} and {MAX_VECTOR_DIM}")
    floats: list[float] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("vector elements must be numbers")
        fv = float(v)
        if not math.isfinite(fv):
            raise ValueError("vector elements must be finite")
        floats.append(fv)
    norm = math.hypot(*floats)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("vector norm must be finite and non-zero")
    arr = array("f", (fv / norm for fv in floats))
    raw = array("f", arr)
    if sys.byteorder != "little":
        raw.byteswap()
    return dim, raw.tobytes(), arr


def decode_vector(blob: bytes, dim: int) -> array | None:
    """Decode a little-endian float32 BLOB into an L2-normalized array('f')."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return None
    if len(blob) != dim * 4 or not (MIN_VECTOR_DIM <= dim <= MAX_VECTOR_DIM):
        return None
    arr = array("f")
    try:
        arr.frombytes(bytes(blob))
        if sys.byteorder != "little":
            arr.byteswap()
        norm = math.hypot(*arr)
    except (ValueError, OverflowError, TypeError):
        return None
    if not math.isfinite(norm) or norm == 0.0:
        return None
    if abs(norm - 1.0) > 1e-4:
        arr = array("f", (v / norm for v in arr))
    return arr


class MemoryStore:
    """Thread-safe, pure-stdlib SQLite WAL vault."""

    def __init__(self, db_path: str | Path, *, max_cached_vectors: int = DEFAULT_VECTOR_CACHE_SIZE):
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._max_cached_vectors = max(0, int(max_cached_vectors))
        # Key: (memory_id, fingerprint, content_version, blob_sha16) -> array('f')
        self._vec_cache: OrderedDict[tuple[int, str, int, bytes], array] = OrderedDict()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    @contextmanager
    def _read(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write(self):
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._write() as conn:
            for ddl in _SCHEMA_SQL:
                conn.execute(ddl)
            defaults = {
                "schema_version": str(SCHEMA_VERSION),
                "library_uuid": str(uuid.uuid4()),
                "global_revision": "1",
                "global_pause": "0",
            }
            for k, v in defaults.items():
                conn.execute(
                    "INSERT OR IGNORE INTO library_meta(key, value) VALUES(?, ?)", (k, v)
                )

    @staticmethod
    def _bump_revision(conn: sqlite3.Connection) -> int:
        row = conn.execute(
            "SELECT value FROM library_meta WHERE key='global_revision'"
        ).fetchone()
        rev = int(row["value"]) + 1 if row else 1
        conn.execute(
            "INSERT OR REPLACE INTO library_meta(key, value) VALUES('global_revision', ?)",
            (str(rev),),
        )
        return rev

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        try:
            terms = json.loads(row["search_terms"] or "[]")
            if not isinstance(terms, list):
                terms = []
        except json.JSONDecodeError:
            terms = []
        return {
            "id": int(row["id"]),
            "scope": row["scope"],
            "kind": row["kind"],
            "content": row["content"],
            "search_terms": terms,
            "topic": row["topic"],
            "source_client": row["source_client"],
            "machine": row["machine"],
            "session_ref": row["session_ref"],
            "content_sha256": row["content_sha256"],
            "version": int(row["version"]),
            "status": row["status"],
            "suppressed": bool(row["suppressed"]),
            "suppressed_revision": int(row["suppressed_revision"]),
            "valid_from": row["valid_from"],
            "valid_until": row["valid_until"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    # --- Metadata & Pause Control ---

    def meta(self) -> dict[str, Any]:
        with self._read() as conn:
            rows = conn.execute("SELECT key, value FROM library_meta").fetchall()
            data = {r["key"]: r["value"] for r in rows}
            counts = conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status='active' AND suppressed=0 THEN 1 ELSE 0 END) AS active,
                    SUM(CASE WHEN status='active' AND suppressed=1 THEN 1 ELSE 0 END) AS suppressed,
                    SUM(CASE WHEN status='forgotten' THEN 1 ELSE 0 END) AS forgotten
                FROM memories
                """
            ).fetchone()
            inbox_pending = conn.execute(
                "SELECT COUNT(*) FROM inbox_events WHERE status='pending'"
            ).fetchone()[0]
            rel_count = conn.execute("SELECT COUNT(*) FROM memory_relations").fetchone()[0]
            vec_rows = conn.execute(
                "SELECT fingerprint, dim, COUNT(*) AS cnt FROM embedding_cache GROUP BY fingerprint, dim"
            ).fetchall()
        return {
            "schema_version": int(data.get("schema_version", SCHEMA_VERSION)),
            "library_uuid": data.get("library_uuid", ""),
            "global_revision": int(data.get("global_revision", 0)),
            "global_pause": data.get("global_pause", "0") == "1",
            "counts": {
                "total": int(counts["total"] or 0),
                "active": int(counts["active"] or 0),
                "suppressed": int(counts["suppressed"] or 0),
                "forgotten": int(counts["forgotten"] or 0),
                "relations": int(rel_count or 0),
                "inbox_pending": int(inbox_pending or 0),
            },
            "vectors": [
                {"fingerprint": r["fingerprint"], "dim": int(r["dim"]), "count": int(r["cnt"])}
                for r in vec_rows
            ],
            "vector_cache_entries": len(self._vec_cache),
            "vector_cache_limit": self._max_cached_vectors,
        }

    def global_revision(self) -> int:
        with self._read() as conn:
            row = conn.execute(
                "SELECT value FROM library_meta WHERE key='global_revision'"
            ).fetchone()
            return int(row["value"]) if row else 0

    def is_paused(self) -> bool:
        with self._read() as conn:
            row = conn.execute(
                "SELECT value FROM library_meta WHERE key='global_pause'"
            ).fetchone()
            return bool(row and row["value"] == "1")

    def set_global_pause(self, paused: bool) -> dict[str, Any]:
        if type(paused) is not bool:
            raise ValueError("paused must be a boolean")
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO library_meta(key, value) VALUES('global_pause', ?)",
                ("1" if paused else "0",),
            )
            rev = self._bump_revision(conn)
        return {"global_pause": paused, "global_revision": rev}

    # --- Memory CRUD & Lifecycle ---

    def add_memory(
        self,
        content: str,
        *,
        kind: str = "episode",
        scope: str = "personal",
        search_terms: Any = None,
        topic: str = "",
        source_client: str = "manual",
        machine: str = "local",
        session_ref: str = "",
        valid_from: float | None = None,
        valid_until: float | None = None,
        actor: str = "agent",
        allow_resurrect: bool = False,
    ) -> dict[str, Any]:
        content = validate_content(content)
        kind = validate_kind(kind)
        scope = validate_label(scope, "scope", max_len=MAX_SCOPE)
        terms = validate_search_terms(search_terms)
        topic = validate_label(topic, "topic", allow_empty=True)
        source_client = validate_label(source_client, "source_client")
        machine = validate_label(machine, "machine")
        session_ref = validate_label(session_ref, "session_ref", allow_empty=True)
        actor = validate_label(actor, "actor")
        digest = _sha256_text(content)
        now = _utc_now()

        with self._write() as conn:
            if not allow_resurrect:
                tomb = conn.execute(
                    "SELECT memory_id, deleted_at FROM memory_tombstones WHERE scope=? AND content_sha256=?",
                    (scope, digest),
                ).fetchone()
                if tomb is not None:
                    raise ConflictError(
                        f"Memory matches tombstoned entry #{tomb['memory_id']} (deleted {tomb['deleted_at']})"
                    )
            dup = conn.execute(
                "SELECT * FROM memories WHERE scope=? AND content_sha256=? AND status='active' LIMIT 1",
                (scope, digest),
            ).fetchone()
            if dup is not None:
                # Merge any newly supplied search_terms idempotently without duplicating the memory.
                existing_terms = json.loads(dup["search_terms"] or "[]")
                merged = validate_search_terms([*existing_terms, *terms])
                if merged != existing_terms or (topic and not dup["topic"]):
                    new_ver = int(dup["version"]) + 1
                    conn.execute(
                        "UPDATE memories SET search_terms=?, topic=?, version=?, updated_at=? WHERE id=?",
                        (json.dumps(merged, ensure_ascii=False), topic or dup["topic"], new_ver, now, dup["id"]),
                    )
                    conn.execute(
                        "INSERT INTO memory_revisions(memory_id, version, action, content, search_terms, actor, created_at) "
                        "VALUES(?, ?, 'enrich_terms', ?, ?, ?, ?)",
                        (dup["id"], new_ver, dup["content"], json.dumps(merged, ensure_ascii=False), actor, now),
                    )
                    self._bump_revision(conn)
                    dup = conn.execute("SELECT * FROM memories WHERE id=?", (dup["id"],)).fetchone()
                out = self._row_to_dict(dup)
                out["deduplicated"] = True
                return out

            cur = conn.execute(
                """
                INSERT INTO memories(
                    scope, kind, content, search_terms, topic, source_client, machine,
                    session_ref, content_sha256, version, status, suppressed, suppressed_revision,
                    valid_from, valid_until, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'active', 0, 0, ?, ?, ?, ?)
                """,
                (
                    scope, kind, content, json.dumps(terms, ensure_ascii=False), topic,
                    source_client, machine, session_ref, digest, valid_from, valid_until, now, now,
                ),
            )
            mem_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO memory_revisions(memory_id, version, action, content, search_terms, actor, created_at) "
                "VALUES(?, 1, 'create', ?, ?, ?, ?)",
                (mem_id, content, json.dumps(terms, ensure_ascii=False), actor, now),
            )
            rev = self._bump_revision(conn)
            row = conn.execute("SELECT * FROM memories WHERE id=?", (mem_id,)).fetchone()
        out = self._row_to_dict(row)
        out["deduplicated"] = False
        out["global_revision"] = rev
        return out

    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("memory_id must be a positive integer")
        with self._read() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._row_to_dict(row) if row is not None else None

    def list_memories(
        self,
        *,
        scope: str | None = "personal",
        include_suppressed: bool = False,
        include_forgotten: bool = False,
        kind: str | None = None,
        machine: str | None = None,
        source_client: str | None = None,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope is not None:
            clauses.append("scope=?")
            params.append(validate_label(scope, "scope", max_len=MAX_SCOPE))
        if not include_forgotten:
            clauses.append("status='active'")
        if not include_suppressed:
            clauses.append("suppressed=0")
        if kind is not None:
            clauses.append("kind=?")
            params.append(validate_kind(kind))
        if machine is not None:
            clauses.append("machine=?")
            params.append(validate_label(machine, "machine"))
        if source_client is not None:
            clauses.append("source_client=?")
            params.append(validate_label(source_client, "source_client"))
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        lim = max(1, min(int(limit), 10_000))
        params.append(lim)
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def correct_memory(
        self,
        memory_id: int,
        content: str,
        expected_version: int,
        *,
        search_terms: Any = None,
        topic: str | None = None,
        actor: str = "agent",
    ) -> dict[str, Any]:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("memory_id must be a positive integer")
        if type(expected_version) is not int or expected_version < 1:
            raise ValueError("expected_version must be a positive integer")
        content = validate_content(content)
        actor = validate_label(actor, "actor")
        digest = _sha256_text(content)
        now = _utc_now()

        with self._write() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Memory #{memory_id} not found")
            if row["status"] != "active":
                raise ConflictError(f"Memory #{memory_id} is not active (status={row['status']})")
            if int(row["version"]) != expected_version:
                raise ConflictError(
                    f"CAS version mismatch for memory #{memory_id}: expected {expected_version}, got {row['version']}"
                )
            terms_json = (
                json.dumps(validate_search_terms(search_terms), ensure_ascii=False)
                if search_terms is not None
                else row["search_terms"]
            )
            new_topic = validate_label(topic, "topic", allow_empty=True) if topic is not None else row["topic"]
            new_ver = expected_version + 1
            conn.execute(
                "UPDATE memories SET content=?, search_terms=?, topic=?, content_sha256=?, version=?, updated_at=? WHERE id=?",
                (content, terms_json, new_topic, digest, new_ver, now, memory_id),
            )
            conn.execute(
                "INSERT INTO memory_revisions(memory_id, version, action, content, search_terms, actor, created_at) "
                "VALUES(?, ?, 'correct', ?, ?, ?, ?)",
                (memory_id, new_ver, content, terms_json, actor, now),
            )
            rev = self._bump_revision(conn)
            updated = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        out = self._row_to_dict(updated)
        out["global_revision"] = rev
        return out

    def set_suppressed(
        self, memory_id: int, suppressed: bool, *, actor: str = "agent"
    ) -> dict[str, Any]:
        """Soft-suspend or restore a memory without deleting it or altering its content version."""
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("memory_id must be a positive integer")
        if type(suppressed) is not bool:
            raise ValueError("suppressed must be a boolean")
        actor = validate_label(actor, "actor")
        now = _utc_now()

        with self._write() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Memory #{memory_id} not found")
            if row["status"] != "active":
                raise ConflictError(f"Memory #{memory_id} is forgotten and cannot be suppressed/restored")
            if row["kind"] == "persona" and suppressed:
                raise ConflictError("persona memories cannot be suppressed; correct or forget explicitly")
            new_srev = int(row["suppressed_revision"]) + 1
            conn.execute(
                "UPDATE memories SET suppressed=?, suppressed_revision=?, updated_at=? WHERE id=?",
                (1 if suppressed else 0, new_srev, now, memory_id),
            )
            conn.execute(
                "INSERT INTO memory_revisions(memory_id, version, action, content, search_terms, actor, created_at) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    int(row["version"]),
                    "suppress" if suppressed else "restore",
                    row["content"],
                    row["search_terms"],
                    actor,
                    now,
                ),
            )
            rev = self._bump_revision(conn)
            updated = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        out = self._row_to_dict(updated)
        out["global_revision"] = rev
        return out

    def forget_memory(self, memory_id: int, *, reason: str = "", actor: str = "agent") -> dict[str, Any]:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("memory_id must be a positive integer")
        reason = validate_label(reason, "reason", allow_empty=True, max_len=256)
        actor = validate_label(actor, "actor")
        now = _utc_now()

        with self._write() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Memory #{memory_id} not found")
            if row["status"] == "forgotten":
                out = self._row_to_dict(row)
                out["already_forgotten"] = True
                return out
            conn.execute(
                "UPDATE memories SET status='forgotten', updated_at=? WHERE id=?",
                (now, memory_id),
            )
            conn.execute("DELETE FROM embedding_cache WHERE memory_id=?", (memory_id,))
            conn.execute(
                "INSERT OR REPLACE INTO memory_tombstones(memory_id, scope, content_sha256, reason, deleted_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (memory_id, row["scope"], row["content_sha256"], reason, now),
            )
            conn.execute(
                "INSERT INTO memory_revisions(memory_id, version, action, content, search_terms, actor, created_at) "
                "VALUES(?, ?, 'forget', ?, ?, ?, ?)",
                (memory_id, int(row["version"]), row["content"], row["search_terms"], actor, now),
            )
            rev = self._bump_revision(conn)
            updated = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        with self._lock:
            for key in [k for k in self._vec_cache if k[0] == memory_id]:
                self._vec_cache.pop(key, None)
        out = self._row_to_dict(updated)
        out["global_revision"] = rev
        return out

    # --- 1-Hop Typed Relations ---

    def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str = "related",
        *,
        weight: float = 1.0,
        confirmed: bool = True,
    ) -> dict[str, Any]:
        if type(source_id) is not int or type(target_id) is not int or source_id < 1 or target_id < 1:
            raise ValueError("source_id and target_id must be positive integers")
        if source_id == target_id:
            raise ValueError("self-relations are not allowed")
        if relation_type not in RELATION_TYPES:
            raise ValueError(f"relation_type must be one of {RELATION_TYPES}")
        w = float(weight)
        if not math.isfinite(w) or not (0.0 < w <= 10.0):
            raise ValueError("weight must be in (0, 10]")
        now = _utc_now()
        with self._write() as conn:
            for mid in (source_id, target_id):
                if conn.execute("SELECT 1 FROM memories WHERE id=? AND status='active'", (mid,)).fetchone() is None:
                    raise NotFoundError(f"Active memory #{mid} not found")
            conn.execute(
                """
                INSERT INTO memory_relations(source_id, target_id, relation_type, weight, confirmed, created_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, relation_type) DO UPDATE SET
                    weight=excluded.weight, confirmed=excluded.confirmed
                """,
                (source_id, target_id, relation_type, w, 1 if confirmed else 0, now),
            )
            rev = self._bump_revision(conn)
            row = conn.execute(
                "SELECT * FROM memory_relations WHERE source_id=? AND target_id=? AND relation_type=?",
                (source_id, target_id, relation_type),
            ).fetchone()
        return {
            "id": int(row["id"]),
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "relation_type": row["relation_type"],
            "weight": float(row["weight"]),
            "confirmed": bool(row["confirmed"]),
            "created_at": row["created_at"],
            "global_revision": rev,
        }

    def get_neighbors(self, memory_ids: list[int], *, confirmed_only: bool = False) -> dict[int, list[dict[str, Any]]]:
        ids = sorted({int(mid) for mid in memory_ids if isinstance(mid, int) and mid > 0})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        conf_sql = "AND confirmed=1" if confirmed_only else ""
        with self._read() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_relations
                WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders})) {conf_sql}
                """,
                [*ids, *ids],
            ).fetchall()
        adj: dict[int, list[dict[str, Any]]] = {mid: [] for mid in ids}
        for r in rows:
            s, t = int(r["source_id"]), int(r["target_id"])
            if s in adj:
                adj[s].append(
                    {"neighbor_id": t, "relation_type": r["relation_type"], "direction": "out", "weight": float(r["weight"])}
                )
            if t in adj:
                adj[t].append(
                    {"neighbor_id": s, "relation_type": r["relation_type"], "direction": "in", "weight": float(r["weight"])}
                )
        return adj

    # --- Compact Dimension-Agnostic Vector Cache (1C1G-Safe) ---

    def save_vectors_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch upsert precomputed vectors (from workstation GPU or API)."""
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        now = _utc_now()
        accepted = 0
        skipped_stale = 0
        with self._write() as conn:
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("each vector item must be an object")
                mid = item.get("memory_id")
                ver = item.get("content_version")
                fp = validate_label(item.get("fingerprint", ""), "fingerprint", max_len=128)
                if type(mid) is not int or mid < 1 or type(ver) is not int or ver < 1:
                    raise ValueError("memory_id and content_version must be positive ints")
                dim, blob, arr = encode_vector(item.get("vector"))
                row = conn.execute(
                    "SELECT version, status FROM memories WHERE id=?", (mid,)
                ).fetchone()
                if row is None or row["status"] != "active" or int(row["version"]) != ver:
                    skipped_stale += 1
                    continue
                conn.execute(
                    """
                    INSERT INTO embedding_cache(memory_id, content_version, fingerprint, dim, vector, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id, fingerprint) DO UPDATE SET
                        content_version=excluded.content_version,
                        dim=excluded.dim,
                        vector=excluded.vector,
                        updated_at=excluded.updated_at
                    """,
                    (mid, ver, fp, dim, blob, now),
                )
                self._cache_put(mid, fp, ver, blob, arr)
                accepted += 1
        return {"accepted": accepted, "skipped_stale": skipped_stale}

    def list_missing_vectors(self, fingerprint: str, *, scope: str | None = None, limit: int = 256) -> list[dict[str, Any]]:
        fp = validate_label(fingerprint, "fingerprint", max_len=128)
        lim = max(1, min(int(limit), 2048))
        scope_sql = "AND m.scope=?" if scope is not None else ""
        params: list[Any] = [fp]
        if scope is not None:
            params.append(validate_label(scope, "scope", max_len=MAX_SCOPE))
        params.append(lim)
        with self._read() as conn:
            rows = conn.execute(
                f"""
                SELECT m.id, m.scope, m.kind, m.content, m.search_terms, m.topic, m.version
                FROM memories m
                LEFT JOIN embedding_cache e
                    ON e.memory_id = m.id AND e.fingerprint = ? AND e.content_version = m.version
                WHERE m.status='active' AND m.suppressed=0 AND e.memory_id IS NULL {scope_sql}
                ORDER BY m.id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return [
            {
                "memory_id": int(r["id"]),
                "scope": r["scope"],
                "kind": r["kind"],
                "content": r["content"],
                "search_terms": json.loads(r["search_terms"] or "[]"),
                "topic": r["topic"],
                "content_version": int(r["version"]),
            }
            for r in rows
        ]

    def _cache_put(self, memory_id: int, fp: str, ver: int, blob: bytes, arr: array) -> None:
        if self._max_cached_vectors <= 0:
            return
        key = (memory_id, fp, ver, hashlib.sha256(blob).digest()[:16])
        with self._lock:
            self._vec_cache[key] = arr
            self._vec_cache.move_to_end(key)
            while len(self._vec_cache) > self._max_cached_vectors:
                self._vec_cache.popitem(last=False)

    def get_vectors_batch(
        self, pairs: list[tuple[int, int]], fingerprint: str
    ) -> dict[int, array]:
        """Single-transaction batch load returning memory_id -> normalized array('f')."""
        fp = validate_label(fingerprint, "fingerprint", max_len=128)
        wanted = {int(mid): int(ver) for mid, ver in pairs}
        if not wanted:
            return {}
        out: dict[int, array] = {}
        ids = list(wanted)
        chunk_size = 400
        with self._read() as conn:
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT memory_id, content_version, dim, vector
                    FROM embedding_cache
                    WHERE fingerprint=? AND memory_id IN ({placeholders})
                    """,
                    [fp, *chunk],
                ).fetchall()
                for r in rows:
                    mid = int(r["memory_id"])
                    ver = int(r["content_version"])
                    if wanted.get(mid) != ver:
                        continue
                    blob = bytes(r["vector"])
                    digest = hashlib.sha256(blob).digest()[:16]
                    cache_key = (mid, fp, ver, digest)
                    arr = None
                    if self._max_cached_vectors > 0:
                        with self._lock:
                            arr = self._vec_cache.get(cache_key)
                            if arr is not None:
                                self._vec_cache.move_to_end(cache_key)
                    if arr is None:
                        arr = decode_vector(blob, int(r["dim"]))
                        if arr is None:
                            continue
                        self._cache_put(mid, fp, ver, blob, arr)
                    out[mid] = arr
        return out

    # --- Offline Consolidation Staging Inbox ---

    def enqueue_inbox(
        self,
        content: str,
        *,
        scope: str = "personal",
        source_client: str = "manual",
        machine: str = "local",
        session_ref: str = "",
        event_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        content = validate_content(content)
        scope = validate_label(scope, "scope", max_len=MAX_SCOPE)
        source_client = validate_label(source_client, "source_client")
        machine = validate_label(machine, "machine")
        session_ref = validate_label(session_ref, "session_ref", allow_empty=True)
        digest = _sha256_text(content)
        ev_key = validate_label(event_key or digest[:24], "event_key")
        meta_json = json.dumps(metadata or {}, ensure_ascii=False, separators=(",", ":"))
        now = _utc_now()

        with self._write() as conn:
            existing = conn.execute(
                "SELECT * FROM inbox_events WHERE scope=? AND source_client=? AND session_ref=? AND event_key=?",
                (scope, source_client, session_ref, ev_key),
            ).fetchone()
            if existing is not None:
                return {**dict(existing), "deduplicated": True}
            cur = conn.execute(
                """
                INSERT INTO inbox_events(
                    scope, source_client, machine, session_ref, event_key, content,
                    metadata, content_sha256, status, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (scope, source_client, machine, session_ref, ev_key, content, meta_json, digest, now, now),
            )
            row = conn.execute("SELECT * FROM inbox_events WHERE id=?", (cur.lastrowid,)).fetchone()
        return {**dict(row), "deduplicated": False}

    def list_inbox(self, *, status: str = "pending", scope: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if status not in INBOX_STATUSES:
            raise ValueError(f"status must be one of {INBOX_STATUSES}")
        lim = max(1, min(int(limit), 1000))
        clauses = ["status=?"]
        params: list[Any] = [status]
        if scope is not None:
            clauses.append("scope=?")
            params.append(validate_label(scope, "scope", max_len=MAX_SCOPE))
        params.append(lim)
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT * FROM inbox_events WHERE {' AND '.join(clauses)} ORDER BY id ASC LIMIT ?",
                params,
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            try:
                item["metadata"] = json.loads(item["metadata"] or "{}")
            except json.JSONDecodeError:
                item["metadata"] = {}
            out.append(item)
        return out

    def mark_inbox(self, inbox_ids: list[int], status: str) -> dict[str, Any]:
        if status not in ("consolidated", "skipped", "pending"):
            raise ValueError("status must be 'consolidated', 'skipped', or 'pending'")
        ids = [int(i) for i in inbox_ids if isinstance(i, int) and i > 0]
        if not ids:
            return {"updated": 0}
        now = _utc_now()
        placeholders = ",".join("?" for _ in ids)
        with self._write() as conn:
            cur = conn.execute(
                f"UPDATE inbox_events SET status=?, claim_token=NULL, updated_at=? WHERE id IN ({placeholders})",
                [status, now, *ids],
            )
        return {"updated": int(cur.rowcount)}

    def get_meta_value(self, key: str, default: str = "") -> str:
        k = validate_label(key, "meta_key", max_len=128)
        with self._read() as conn:
            row = conn.execute("SELECT value FROM library_meta WHERE key=?", (k,)).fetchone()
        return str(row["value"]) if row is not None else default

    def set_meta_value(self, key: str, value: str) -> str:
        k = validate_label(key, "meta_key", max_len=128)
        val = str(value)
        with self._write() as conn:
            conn.execute(
                "INSERT INTO library_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (k, val),
            )
        return val

    def list_revisions(self, memory_id: int, *, limit: int = 30) -> list[dict[str, Any]]:
        if type(memory_id) is not int or memory_id < 1:
            raise ValueError("memory_id must be a positive integer")
        lim = max(1, min(int(limit), 100))
        with self._read() as conn:
            rows = conn.execute(
                "SELECT id, memory_id, version, action, content, search_terms, actor, created_at "
                "FROM memory_revisions WHERE memory_id=? ORDER BY id DESC LIMIT ?",
                (memory_id, lim),
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            try:
                item["search_terms"] = json.loads(item["search_terms"] or "[]")
            except json.JSONDecodeError:
                item["search_terms"] = []
            out.append(item)
        return out

    def list_relations(self, *, memory_id: int | None = None, limit: int = 500) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit), 2000))
        with self._read() as conn:
            if memory_id is not None:
                rows = conn.execute(
                    "SELECT id, source_id, target_id, relation_type, weight, confirmed, created_at "
                    "FROM memory_relations WHERE source_id=? OR target_id=? ORDER BY id DESC LIMIT ?",
                    (int(memory_id), int(memory_id), lim),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, source_id, target_id, relation_type, weight, confirmed, created_at "
                    "FROM memory_relations ORDER BY id DESC LIMIT ?",
                    (lim,),
                ).fetchall()
        return [
            {
                "id": int(r["id"]),
                "source_id": int(r["source_id"]),
                "target_id": int(r["target_id"]),
                "relation_type": r["relation_type"],
                "weight": float(r["weight"]),
                "confirmed": bool(r["confirmed"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def delete_relation(self, source_id: int, target_id: int, relation_type: str | None = None) -> dict[str, Any]:
        with self._write() as conn:
            if relation_type:
                cur = conn.execute(
                    "DELETE FROM memory_relations WHERE source_id=? AND target_id=? AND relation_type=?",
                    (int(source_id), int(target_id), str(relation_type)),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM memory_relations WHERE source_id=? AND target_id=?",
                    (int(source_id), int(target_id)),
                )
            rev = self._bump_revision(conn) if cur.rowcount > 0 else self.global_revision()
        return {"deleted": int(cur.rowcount), "global_revision": rev}

    def list_known_machines(self) -> list[dict[str, Any]]:
        with self._read() as conn:
            mem_rows = conn.execute(
                "SELECT machine, COUNT(*) AS total, "
                "SUM(CASE WHEN status='active' AND suppressed=0 THEN 1 ELSE 0 END) AS active "
                "FROM memories WHERE machine != '' GROUP BY machine"
            ).fetchall()
            inbox_rows = conn.execute(
                "SELECT machine, COUNT(*) AS pending FROM inbox_events "
                "WHERE status='pending' AND machine != '' GROUP BY machine"
            ).fetchall()
        by_name: dict[str, dict[str, Any]] = {}
        for r in mem_rows:
            by_name[r["machine"]] = {
                "machine": r["machine"],
                "memories_total": int(r["total"]),
                "memories_active": int(r["active"]),
                "inbox_pending": 0,
            }
        for r in inbox_rows:
            entry = by_name.setdefault(
                r["machine"],
                {"machine": r["machine"], "memories_total": 0, "memories_active": 0, "inbox_pending": 0},
            )
            entry["inbox_pending"] = int(r["pending"])
        return sorted(by_name.values(), key=lambda x: x["machine"])

    def graph_snapshot(
        self,
        *,
        scope: str | None = None,
        include_suppressed: bool = True,
        limit: int = 300,
    ) -> dict[str, Any]:
        nodes = self.list_memories(
            scope=scope,
            include_suppressed=include_suppressed,
            include_forgotten=False,
            limit=limit,
        )
        node_ids = {int(n["id"]) for n in nodes}
        edges = [
            e
            for e in self.list_relations(limit=1500)
            if e["source_id"] in node_ids and e["target_id"] in node_ids
        ]
        return {
            "global_revision": self.global_revision(),
            "nodes": [
                {
                    "id": n["id"],
                    "scope": n["scope"],
                    "kind": n["kind"],
                    "topic": n["topic"] or "general",
                    "content": n["content"],
                    "search_terms": n["search_terms"],
                    "machine": n["machine"],
                    "source_client": n["source_client"],
                    "version": n["version"],
                    "suppressed": n["suppressed"],
                    "updated_at": n["updated_at"],
                }
                for n in nodes
            ],
            "edges": edges,
        }
