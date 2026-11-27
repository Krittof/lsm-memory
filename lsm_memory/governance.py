"""Server-enforced machine, client-role, and automation governance for LSMMemory.

Stored cleanly as compact JSON inside the existing `library_meta` table
(zero extra schema migrations, zero external dependencies, 1C1G safe).
"""

from __future__ import annotations

import json
from typing import Any

from .store import MemoryStore, validate_label

GOVERNANCE_META_KEY = "governance_policy_v1"
ROLES = ("read_write", "read_capture", "read_only")
CONSOLIDATE_MODES = ("manual", "auto")

# Operations mapped to minimum required role capability
READ_OPS = frozenset({"search", "status", "list", "graph", "revisions", "vectors_missing"})
CAPTURE_OPS = frozenset({"capture", "consolidate_pull"})
WRITE_OPS = frozenset({
    "add", "correct", "suppress", "restore", "forget",
    "relate", "delete_relation", "consolidate_commit", "vectors_upsert",
})


class GovernanceDeniedError(PermissionError):
    """Raised when a machine or client role is denied by server governance policy."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def default_governance_policy() -> dict[str, Any]:
    return {
        "auto_inject": True,
        "auto_consolidate_mode": "manual",  # "manual" | "auto"
        "auto_consolidate_interval_min": 60,
        "mcp_default_role": "read_write",  # "read_write" | "read_capture" | "read_only"
        "default_machine_included": True,  # True = opt-out, False = strict allowlist
        "machines": {},  # machine_name -> {"included": bool, "role": str, "auto_consolidate": bool, "note": str}
    }


def normalize_governance_policy(raw: dict[str, Any] | None) -> dict[str, Any]:
    base = default_governance_policy()
    if not isinstance(raw, dict):
        return base
    if type(raw.get("auto_inject")) is bool:
        base["auto_inject"] = raw["auto_inject"]
    mode = str(raw.get("auto_consolidate_mode") or base["auto_consolidate_mode"]).strip().lower()
    if mode in CONSOLIDATE_MODES:
        base["auto_consolidate_mode"] = mode
    try:
        iv = int(raw.get("auto_consolidate_interval_min", base["auto_consolidate_interval_min"]))
        base["auto_consolidate_interval_min"] = max(5, min(iv, 1440))
    except (TypeError, ValueError):
        pass
    role = str(raw.get("mcp_default_role") or base["mcp_default_role"]).strip().lower()
    if role in ROLES:
        base["mcp_default_role"] = role
    if type(raw.get("default_machine_included")) is bool:
        base["default_machine_included"] = raw["default_machine_included"]

    raw_machines = raw.get("machines")
    clean_machines: dict[str, dict[str, Any]] = {}
    if isinstance(raw_machines, dict):
        for name, rule in raw_machines.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(rule, dict):
                continue
            try:
                m_key = validate_label(name.strip(), "machine")
            except ValueError:
                continue
            m_role = str(rule.get("role") or "read_write").strip().lower()
            if m_role not in ROLES:
                m_role = "read_write"
            clean_machines[m_key] = {
                "included": bool(rule.get("included", True)),
                "role": m_role,
                "auto_consolidate": bool(rule.get("auto_consolidate", True)),
                "note": str(rule.get("note") or "")[:160],
            }
    base["machines"] = clean_machines
    return base


def load_governance(store: MemoryStore) -> dict[str, Any]:
    raw = store.get_meta_value(GOVERNANCE_META_KEY, "")
    if not raw:
        return default_governance_policy()
    try:
        return normalize_governance_policy(json.loads(raw))
    except json.JSONDecodeError:
        return default_governance_policy()


def save_governance(store: MemoryStore, updates: dict[str, Any], *, replace: bool = False) -> dict[str, Any]:
    current = default_governance_policy() if replace else load_governance(store)
    if not isinstance(updates, dict):
        raise ValueError("governance updates must be a JSON object")
    merged = dict(current)
    for k in (
        "auto_inject",
        "auto_consolidate_mode",
        "auto_consolidate_interval_min",
        "mcp_default_role",
        "default_machine_included",
    ):
        if k in updates:
            merged[k] = updates[k]
    if "machines" in updates and isinstance(updates["machines"], dict):
        if replace:
            merged["machines"] = updates["machines"]
        else:
            m_map = dict(current.get("machines") or {})
            for m_name, rule in updates["machines"].items():
                if rule is None:
                    m_map.pop(m_name, None)
                elif isinstance(rule, dict):
                    existing = dict(m_map.get(m_name) or {"included": True, "role": "read_write", "auto_consolidate": True, "note": ""})
                    existing.update(rule)
                    m_map[m_name] = existing
            merged["machines"] = m_map
    normalized = normalize_governance_policy(merged)
    store.set_meta_value(GOVERNANCE_META_KEY, json.dumps(normalized, ensure_ascii=False, separators=(",", ":")))
    return normalized


def is_machine_included(policy: dict[str, Any], machine: str | None) -> bool:
    name = (machine or "").strip()
    machines = policy.get("machines") or {}
    if name and name in machines:
        return bool(machines[name].get("included", True))
    return bool(policy.get("default_machine_included", True))


def is_machine_auto_consolidate(policy: dict[str, Any], machine: str | None) -> bool:
    if not is_machine_included(policy, machine):
        return False
    name = (machine or "").strip()
    machines = policy.get("machines") or {}
    if name and name in machines:
        return bool(machines[name].get("auto_consolidate", True))
    return True


def effective_role(
    policy: dict[str, Any],
    *,
    machine: str | None = None,
    channel: str = "admin",
) -> str:
    """Resolve strictest applicable role across channel (`mcp` vs `admin`/`webui`) and machine rule."""
    # Admin/WebUI management console operates with read_write unless restricted by machine rule on write target.
    rank = {"read_only": 0, "read_capture": 1, "read_write": 2}
    inv = {0: "read_only", 1: "read_capture", 2: "read_write"}
    base_role = policy.get("mcp_default_role", "read_write") if channel == "mcp" else "read_write"
    level = rank.get(base_role, 2)

    name = (machine or "").strip()
    machines = policy.get("machines") or {}
    if name and name in machines:
        m_role = machines[name].get("role", "read_write")
        level = min(level, rank.get(m_role, 2))
    return inv[level]


def enforce_operation(
    policy: dict[str, Any],
    operation: str,
    *,
    machine: str | None = None,
    channel: str = "admin",
) -> None:
    """Raise GovernanceDeniedError if machine is excluded or role forbids operation."""
    if channel != "admin" and machine and not is_machine_included(policy, machine):
        raise GovernanceDeniedError(
            "machine_excluded",
            f"Machine '{machine}' is excluded by memory governance policy.",
        )
    role = effective_role(policy, machine=machine if channel != "admin" else None, channel=channel)
    if operation in READ_OPS:
        return
    if operation in CAPTURE_OPS:
        if role in ("read_capture", "read_write"):
            return
        raise GovernanceDeniedError(
            "role_read_only",
            f"Operation '{operation}' denied: effective role is '{role}'.",
        )
    if operation in WRITE_OPS:
        if role == "read_write":
            return
        raise GovernanceDeniedError(
            "role_write_denied",
            f"Operation '{operation}' denied: effective role is '{role}' (requires 'read_write').",
        )
