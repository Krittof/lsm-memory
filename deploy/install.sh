#!/usr/bin/env bash
# Host-agnostic, idempotent installer for LSMMemory on Linux (ARM64 / x86_64, 1C1G-safe).
# Usage:
#   sudo bash deploy/install.sh [--mode daemon|embedded] [--source /path/to/LSMMemory] \
#        [--venv /opt/local-shell-mcp/venv] [--user lsm] [--port 8775] \
#        [--memory-max 128M] [--patch-lsm] [--env-file /etc/local-shell-mcp/env] [--uninstall]
set -euo pipefail

MODE="daemon"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LSM_VENV="/opt/local-shell-mcp/venv"
APP_DIR="/opt/lsm-memory"
DATA_DIR="/var/lib/lsm-memory"
RUN_USER="lsm"
RUN_GROUP="lsm"
PORT="8775"
MEMORY_MAX="128M"
MAX_CACHED_VECTORS="256"
ENV_FILE="/etc/local-shell-mcp/env"
PATCH_LSM=0
UNINSTALL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --source) SOURCE_DIR="$2"; shift 2 ;;
        --venv) LSM_VENV="$2"; shift 2 ;;
        --app-dir) APP_DIR="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --user) RUN_USER="$2"; RUN_GROUP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --memory-max) MEMORY_MAX="$2"; shift 2 ;;
        --max-cached-vectors) MAX_CACHED_VECTORS="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --patch-lsm) PATCH_LSM=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help)
            sed -n '2,6p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
    esac
done

PY_BIN="${LSM_VENV}/bin/python"
if [[ ! -x "${PY_BIN}" ]]; then
    PY_BIN="$(command -v python3)"
fi

if [[ "${UNINSTALL}" -eq 1 ]]; then
    echo "=== Reverting LSM patch (if applied) ==="
    if "${PY_BIN}" -c "import lsm_memory.lsm_patcher" >/dev/null 2>&1; then
        "${PY_BIN}" -B -m lsm_memory.lsm_patcher revert || true
    fi
    if systemctl list-unit-files | grep -q '^lsm-memory\.service'; then
        systemctl disable --now lsm-memory.service || true
    fi
    echo "Uninstall completed (data preserved at ${DATA_DIR})."
    exit 0
fi

if [[ "${MODE}" != "daemon" && "${MODE}" != "embedded" ]]; then
    echo "Invalid --mode: ${MODE} (expected daemon or embedded)" >&2
    exit 2
fi

if ! id -u "${RUN_USER}" >/dev/null 2>&1; then
    useradd -r -m -d "${DATA_DIR}" -s /bin/bash "${RUN_USER}"
fi

echo "=== 1. Stage package into ${APP_DIR} and install into ${PY_BIN} ==="
mkdir -p "${APP_DIR}/lsm_memory" "${DATA_DIR}"
cp -f "${SOURCE_DIR}"/lsm_memory/*.py "${APP_DIR}/lsm_memory/"
if [[ -f "${SOURCE_DIR}/pyproject.toml" ]]; then
    cp -f "${SOURCE_DIR}/pyproject.toml" "${APP_DIR}/"
fi
if [[ -f "${SOURCE_DIR}/README.md" ]]; then
    cp -f "${SOURCE_DIR}/README.md" "${APP_DIR}/"
fi
chown -R "${RUN_USER}:${RUN_GROUP}" "${DATA_DIR}"
chmod 0750 "${DATA_DIR}"
chmod -R a+rX "${APP_DIR}"

# Install directly into LSM's site-packages via a clean .pth file or pip/uv (zero network required)
SITE_PKG="$("${PY_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "${APP_DIR}" > "${SITE_PKG}/lsm_memory.pth"
chmod 0644 "${SITE_PKG}/lsm_memory.pth"
"${PY_BIN}" -m compileall -q "${APP_DIR}/lsm_memory"
"${PY_BIN}" -B -c "import lsm_memory, lsm_memory.lsm_bridge, lsm_memory.lsm_patcher; print('IMPORT_OK', lsm_memory.__version__)"

DB_PATH="${DATA_DIR}/memory.sqlite3"
TOKEN_FILE="${DATA_DIR}/daemon.token"

if [[ "${MODE}" == "daemon" ]]; then
    echo "=== 2. Install & start lsm-memory.service (1C1G-safe: MemoryMax=${MEMORY_MAX}) ==="
    cat > /etc/systemd/system/lsm-memory.service <<EOF
[Unit]
Description=LSMMemory Cross-Network Vault Daemon (1C1G-Safe Loopback)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
Group=${RUN_GROUP}
WorkingDirectory=${APP_DIR}
Environment=PYTHONDONTWRITEBYTECODE=1
ExecStart=${PY_BIN} -B -m lsm_memory.daemon --db ${DB_PATH} --host 127.0.0.1 --port ${PORT} --token-file ${TOKEN_FILE} --max-cached-vectors ${MAX_CACHED_VECTORS}
Restart=always
RestartSec=3
MemoryMax=${MEMORY_MAX}
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=${DATA_DIR}

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable lsm-memory.service
    systemctl restart lsm-memory.service
    sleep 2
    systemctl is-active lsm-memory.service
    curl -fsS "http://127.0.0.1:${PORT}/healthz"
    echo ""
fi

if [[ -f "${ENV_FILE}" ]]; then
    echo "=== 3. Update ${ENV_FILE} idempotently (backup created if absent) ==="
    if [[ ! -f "${ENV_FILE}.pre-lsm-memory.bak" ]]; then
        cp -p "${ENV_FILE}" "${ENV_FILE}.pre-lsm-memory.bak"
    fi
    "${PY_BIN}" - "${ENV_FILE}" "${MODE}" "${PORT}" "${TOKEN_FILE}" "${DB_PATH}" <<'PY'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
mode, port, token_file, db_path = sys.argv[2:6]
lines = env_path.read_text(encoding="utf-8").splitlines()
managed_prefixes = (
    "LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED=",
    "LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN=",
    "LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN_FILE=",
    "LOCAL_SHELL_MCP_MEMORY_VAULT_DB=",
)
kept = [ln for ln in lines if not ln.startswith(managed_prefixes)]
if mode == "daemon":
    kept.extend([
        "LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED=1",
        f"LOCAL_SHELL_MCP_MEMORY_VAULT_ORIGIN=http://127.0.0.1:{port}",
        f"LOCAL_SHELL_MCP_MEMORY_VAULT_TOKEN_FILE={token_file}",
    ])
else:
    kept.extend([
        "LOCAL_SHELL_MCP_MEMORY_VAULT_ENABLED=1",
        f"LOCAL_SHELL_MCP_MEMORY_VAULT_DB={db_path}",
    ])
env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
PY
    chmod 0640 "${ENV_FILE}" || true
fi

if [[ "${PATCH_LSM}" -eq 1 ]]; then
    echo "=== 4. Apply idempotent LSM WebUI bridge patch ==="
    "${PY_BIN}" -B -m lsm_memory.lsm_patcher apply
fi

echo "LSM_MEMORY_INSTALL_OK mode=${MODE} patch=${PATCH_LSM}"
