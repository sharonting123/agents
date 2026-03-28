#!/usr/bin/env bash
# Linux: smoke test — no model load; /api/chat may return 503 (same as .bat)
# chmod +x start_procurement_fastapi_smoke.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
PY="${PYTHON:-python3}"

exec "$PY" qa_fastapi.py --smoke --host 127.0.0.1 --port 7860
