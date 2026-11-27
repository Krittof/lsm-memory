"""Idempotent, reversible, AST-verified patcher for local-shell-mcp WebUI.

Wires `lsm_memory.lsm_bridge` into an installed `local_shell_mcp` package with
four minimal marker-delimited hooks:
  1. `webui/agent_runner.py` (`_context`): register opt-in `memory_vault_*` tools
  2. `webui/agent_runner.py` (`_invoke`): dispatch `memory_vault_*` tools
  3. `webui/agent_runner.py` (`_project_model_messages`): split-injection & resume revalidation
  4. `webui/approval_settings.py` (`is_dangerous`): treat read-only vault tools as non-dangerous

Every operation verifies Python syntax before writing, preserves an untouched
`.lsm_memory_orig.bak` backup, supports idempotent re-application after LSM wheel
upgrades, and reverts cleanly back to byte-identical upstream source.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any


@dataclass(frozen=True)
class HookSpec:
    rel_path: str
    hook_id: str
    anchor: str
    position: str  # "after" or "before"
    snippet: str

    @property
    def start_marker(self) -> str:
        return f"# >>> lsm-memory-vault:{self.hook_id}"

    @property
    def end_marker(self) -> str:
        return f"# <<< lsm-memory-vault:{self.hook_id}"


HOOKS: tuple[HookSpec, ...] = (
    HookSpec(
        rel_path="webui/agent_runner.py",
        hook_id="context-tools",
        anchor="            tools += tool_definitions(config.get('validKeys'))",
        position="after",
        snippet=(
            "            # >>> lsm-memory-vault:context-tools\n"
            "            try:\n"
            "                from lsm_memory.lsm_bridge import vault_tool_definitions as _lsm_vault_tools\n"
            "                tools += _lsm_vault_tools()\n"
            "            except Exception:\n"
            "                pass\n"
            "            # <<< lsm-memory-vault:context-tools"
        ),
    ),
    HookSpec(
        rel_path="webui/agent_runner.py",
        hook_id="invoke-tools",
        anchor='        if call.name in {"web_search", "web_fetch"}:',
        position="before",
        snippet=(
            "        # >>> lsm-memory-vault:invoke-tools\n"
            '        if call.name.startswith("memory_vault_"):\n'
            "            from .misc_routes import effective_features\n"
            "            from .storage import WorkspaceRepository\n"
            "            from lsm_memory.lsm_bridge import execute_vault_tool\n"
            "            _v_settings = WorkspaceRepository(self.database)\n"
            '            if (not effective_features(self.owner, self.authorization_conversation_id, repo=_v_settings)["memory"]\n'
            "                    or not memory_enabled(_v_settings, self.owner, tools=True)):\n"
            "                raise RuntimeSafetyError('Memory tools are disabled')\n"
            "            return execute_vault_tool(\n"
            "                call.name, call.arguments,\n"
            "                agent=state.context.agent,\n"
            "                filters=configured_filters(_v_settings, self.owner),\n"
            "                session_ref=self.authorization_conversation_id,\n"
            "            )\n"
            "        # <<< lsm-memory-vault:invoke-tools"
        ),
    ),
    HookSpec(
        rel_path="webui/agent_runner.py",
        hook_id="project-messages",
        anchor="        projected = project_faded_provider_messages(state.messages, state.context.fading_tier)",
        position="after",
        snippet=(
            "        # >>> lsm-memory-vault:project-messages\n"
            "        try:\n"
            "            from .misc_routes import effective_features as _v_eff_features\n"
            "            from .storage import WorkspaceRepository as _v_ws_repo_cls\n"
            "            from lsm_memory.lsm_bridge import project_memory_vault_messages as _v_project\n"
            "            _v_repo = _v_ws_repo_cls(self.database)\n"
            "            _v_mem_on = (\n"
            '                _v_eff_features(self.owner, self.authorization_conversation_id, repo=_v_repo)["memory"]\n'
            "                and memory_enabled(_v_repo, self.owner)\n"
            "            )\n"
            "            projected = _v_project(\n"
            "                projected,\n"
            "                agent=agent,\n"
            "                namespace=self.namespace,\n"
            "                memory_feature_enabled=_v_mem_on,\n"
            "                filters=configured_filters(_v_repo, self.owner),\n"
            "            )\n"
            "        except Exception:\n"
            "            pass\n"
            "        # <<< lsm-memory-vault:project-messages"
        ),
    ),
    HookSpec(
        rel_path="webui/approval_settings.py",
        hook_id="readonly-approval",
        anchor='    if name in _READ_TOOLS or name in {"web_search", "web_fetch", "ask_user_question", "subagent"}:\n        return False',
        position="after",
        snippet=(
            "    # >>> lsm-memory-vault:readonly-approval\n"
            '    if name in {"memory_vault_search", "memory_vault_status"}:\n'
            "        return False\n"
            "    # <<< lsm-memory-vault:readonly-approval"
        ),
    ),
    HookSpec(
        rel_path="main.py",
        hook_id="webui-console-routes",
        anchor="            routes[2:2] = memory_routes()",
        position="after",
        snippet=(
            "            # >>> lsm-memory-vault:webui-console-routes\n"
            "            try:\n"
            "                from lsm_memory.webui_console import memory_vault_web_routes as _lsm_vault_web_routes\n"
            "                routes[2:2] = _lsm_vault_web_routes()\n"
            "            except Exception:\n"
            "                pass\n"
            "            # <<< lsm-memory-vault:webui-console-routes"
        ),
    ),
    HookSpec(
        rel_path="webui/frontend.py",
        hook_id="webui-nav-inject",
        anchor='    return FileResponse(path, headers={"Cache-Control": cache}, media_type=media_type)',
        position="before",
        snippet=(
            "    # >>> lsm-memory-vault:webui-nav-inject\n"
            '    if path.name == "index.html" and request.method == "GET":\n'
            "        try:\n"
            "            from lsm_memory.webui_console import inject_vault_nav_into_index_html as _v_inj_nav\n"
            '            _v_raw = path.read_text(encoding="utf-8")\n'
            "            _v_html = _v_inj_nav(_v_raw)\n"
            "            if _v_html != _v_raw:\n"
            '                return Response(_v_html, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})\n'
            "        except Exception:\n"
            "            pass\n"
            "    # <<< lsm-memory-vault:webui-nav-inject"
        ),
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def discover_lsm_package_dir(explicit: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not (p / "webui" / "agent_runner.py").is_file():
            raise FileNotFoundError(f"Not a local_shell_mcp WebUI package dir: {p}")
        return p
    spec = importlib.util.find_spec("local_shell_mcp")
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Could not locate installed local_shell_mcp package; pass --target")
    p = Path(next(iter(spec.submodule_search_locations))).resolve()
    if not (p / "webui" / "agent_runner.py").is_file():
        raise FileNotFoundError(f"Installed local_shell_mcp at {p} has no webui/agent_runner.py")
    return p


def _strip_hook(text: str, hook: HookSpec, nl: str) -> str:
    pattern = re.compile(
        r"(?:\r?\n)?[ \t]*" + re.escape(hook.start_marker) + r".*?" + re.escape(hook.end_marker) + r"(?:\r?\n)?",
        re.DOTALL,
    )
    def _repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw.startswith(("\r\n", "\n")) and raw.endswith(("\r\n", "\n")):
            return nl
        return ""
    return pattern.sub(_repl, text)


def _apply_hook(text: str, hook: HookSpec, nl: str) -> str:
    clean = _strip_hook(text, hook, nl)
    anchor_norm = hook.anchor.replace("\n", nl)
    count = clean.count(anchor_norm)
    if count != 1:
        raise RuntimeError(
            f"Anchor mismatch for {hook.rel_path}:{hook.hook_id} (found {count} occurrences; upstream may have changed)"
        )
    snippet_norm = hook.snippet.replace("\n", nl)
    if hook.position == "after":
        replacement = anchor_norm + nl + snippet_norm
    elif hook.position == "before":
        replacement = snippet_norm + nl + anchor_norm
    else:
        raise ValueError(f"Unknown position: {hook.position}")
    return clean.replace(anchor_norm, replacement, 1)


def _hooks_by_file() -> dict[str, list[HookSpec]]:
    grouped: dict[str, list[HookSpec]] = {}
    for h in HOOKS:
        grouped.setdefault(h.rel_path, []).append(h)
    return grouped


def status(target_dir: str | Path | None = None) -> dict[str, Any]:
    root = discover_lsm_package_dir(target_dir)
    files: dict[str, Any] = {}
    all_applied = True
    for rel, hooks in _hooks_by_file().items():
        path = root / rel
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"
        hook_states = {}
        for h in hooks:
            present = h.start_marker in text and h.end_marker in text
            anchor_ok = h.anchor.replace("\n", nl) in _strip_hook(text, h, nl)
            hook_states[h.hook_id] = {"applied": present, "anchor_ok": anchor_ok}
            if not present:
                all_applied = False
        bak = path.with_name(path.name + ".lsm_memory_orig.bak")
        files[rel] = {
            "sha256": _sha256(raw_bytes),
            "backup_exists": bak.is_file(),
            "hooks": hook_states,
        }
    return {"target": str(root), "all_applied": all_applied, "files": files}


def apply(target_dir: str | Path | None = None) -> dict[str, Any]:
    root = discover_lsm_package_dir(target_dir)
    results: dict[str, Any] = {}
    for rel, hooks in _hooks_by_file().items():
        path = root / rel
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"
        bak = path.with_name(path.name + ".lsm_memory_orig.bak")
        had_any = any(h.start_marker in text for h in hooks)
        if not bak.is_file() and not had_any:
            shutil.copy2(path, bak)
        updated = text
        for h in hooks:
            updated = _apply_hook(updated, h, nl)
        # Validate AST before touching disk!
        ast.parse(updated, filename=str(path))
        new_bytes = updated.encode("utf-8")
        changed = new_bytes != raw_bytes
        if changed:
            tmp = path.with_name(path.name + ".lsm_memory_tmp")
            tmp.write_bytes(new_bytes)
            tmp.replace(path)
        results[rel] = {
            "changed": changed,
            "sha256": _sha256(new_bytes),
            "hooks_applied": [h.hook_id for h in hooks],
        }
    return {"action": "apply", "target": str(root), "files": results}


def revert(target_dir: str | Path | None = None) -> dict[str, Any]:
    root = discover_lsm_package_dir(target_dir)
    results: dict[str, Any] = {}
    for rel, hooks in _hooks_by_file().items():
        path = root / rel
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        nl = "\r\n" if "\r\n" in text else "\n"
        cleaned = text
        for h in hooks:
            cleaned = _strip_hook(cleaned, h, nl)
        ast.parse(cleaned, filename=str(path))
        new_bytes = cleaned.encode("utf-8")
        changed = new_bytes != raw_bytes
        if changed:
            tmp = path.with_name(path.name + ".lsm_memory_tmp")
            tmp.write_bytes(new_bytes)
            tmp.replace(path)
        results[rel] = {
            "changed": changed,
            "sha256": _sha256(new_bytes),
        }
    return {"action": "revert", "target": str(root), "files": results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Idempotent reversible patcher for local-shell-mcp WebUI.")
    parser.add_argument("action", choices=["status", "apply", "revert"])
    parser.add_argument("--target", default=None, help="Path to local_shell_mcp package directory")
    args = parser.parse_args(argv)
    if args.action == "status":
        out = status(args.target)
    elif args.action == "apply":
        out = apply(args.target)
    else:
        out = revert(args.target)
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
