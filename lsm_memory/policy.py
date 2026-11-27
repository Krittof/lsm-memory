"""Pure content protection (secret/PII blocking) and 3-layer auto-write decision policy.

Zero I/O, zero external dependencies. Safe for 1C1G controllers and offline batch pipelines.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import re
from typing import Any

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_or_sk_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}")),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}", re.IGNORECASE)),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("lsm_worker_token", re.compile(r"\blsmcp_wk_[A-Za-z0-9_-]{16,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_pat", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
)

ROUTES = ("auto_commit", "queue_review", "defer_reflection", "skip_trivial")
LEVELS = ("conservative", "balanced", "proactive")
FACT_KINDS = ("user_profile", "episode", "transient")
INTENT_CUES = (
    "记住", "以后", "默认", "偏好", "不要", "约定", "规范", "红线",
    "remember", "always", "prefer", "never", "rule",
)
_NEAR_DUP = {
    "conservative": Decimal("0.85"),
    "balanced": Decimal("0.88"),
    "proactive": Decimal("0.92"),
}
_DURABILITY = {
    "user_profile": Decimal("1.0"),
    "episode": Decimal("0.70"),
    "transient": Decimal("0.35"),
}
_SCORE_BASE = {
    "conservative": Decimal("0.75"),
    "balanced": Decimal("0.55"),
    "proactive": Decimal("0.40"),
}


class ContentBlockedError(ValueError):
    """Raised when memory content contains blocked secrets or credentials."""

    def __init__(self, rule_id: str) -> None:
        self.rule_id = rule_id
        super().__init__(f"Content blocked by secret protection policy ({rule_id})")


def check_secret_safety(text: str) -> str | None:
    """Return the first matching secret rule id, or None if clean."""
    if not isinstance(text, str):
        return "invalid_type"
    for rule_id, pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return rule_id
    return None


def assert_content_safe(content: str, search_terms: list[str] | None = None) -> None:
    hit = check_secret_safety(content)
    if hit is not None:
        raise ContentBlockedError(hit)
    for term in search_terms or []:
        hit_t = check_secret_safety(term)
        if hit_t is not None:
            raise ContentBlockedError(hit_t)


def _q6(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


def decide_auto_write(context: dict[str, Any]) -> dict[str, Any]:
    """Deterministic 3-layer auto-write decision."""
    level = str(context.get("level") or "balanced")
    if level not in LEVELS:
        level = "conservative"
    kind = str(context.get("kind") or "episode")
    content = str(context.get("content") or "").strip()
    sim_raw = context.get("similarity")
    sim_dec = _q6(sim_raw) if isinstance(sim_raw, (int, float)) and not isinstance(sim_raw, bool) else None
    matched_id = context.get("matched_memory_id")

    def _emit(route: str, reason_code: str, needs_reflection: bool = False) -> dict[str, Any]:
        payload = {
            "v": 1,
            "route": route,
            "reason_code": reason_code,
            "kind": kind,
            "level": level,
            "matched_memory_id": matched_id,
            "needs_reflection": needs_reflection,
            "similarity": f"{sim_dec:.6f}" if sim_dec is not None else None,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "route": route,
            "reason_code": reason_code,
            "needs_reflection": needs_reflection,
            "similarity": float(sim_dec) if sim_dec is not None else None,
            "matched_memory_id": matched_id,
            "level": level,
            "decision_digest": digest,
        }

    if context.get("auto_write_enabled") is not True:
        return _emit("queue_review", "auto_write_disabled")
    if context.get("global_pause") is True:
        return _emit("queue_review", "global_pause")
    if len(content) < 6:
        return _emit("skip_trivial", "skip_trivial")
    if check_secret_safety(content) is not None:
        return _emit("queue_review", "hazard_blocked")
    if kind == "persona":
        return _emit("queue_review", "persona_human_only")
    if kind not in FACT_KINDS:
        return _emit("queue_review", "invalid_kind")
    if context.get("conflict") is True:
        return _emit("defer_reflection", "conflict", needs_reflection=True)
    if context.get("supersede") is True:
        return _emit("defer_reflection", "supersede", needs_reflection=True)
    if context.get("duplicate") is True:
        return _emit("skip_trivial", "duplicate")
    if sim_dec is not None and sim_dec >= _NEAR_DUP[level]:
        return _emit("queue_review", "semantic_near_dup")

    score = _DURABILITY[kind]
    low = content.lower()
    if any(cue in low for cue in INTENT_CUES):
        score += Decimal("0.20")
    if context.get("grounded") is True:
        score += Decimal("0.10")
    if score >= _SCORE_BASE[level]:
        return _emit("auto_commit", "auto_commit")
    return _emit("queue_review", "below_threshold")
