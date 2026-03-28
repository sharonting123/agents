#!/usr/bin/env bash
# Linux: lite NL2SQL + stub chat (same intent as start_procurement_fastapi_lite.bat)
# chmod +x start_procurement_fastapi_lite.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export QA_LITE_NL2SQL_ONLY=1
export QA_LITE_NL2SQL_DEVICE="${QA_LITE_NL2SQL_DEVICE:-cuda}"
export QA_CHATGLM_DEVICE="${QA_CHATGLM_DEVICE:-cuda}"
export QA_CHATGLM_FP16="${QA_CHATGLM_FP16:-1}"
export CHATGLM_LOAD_IN_8BIT=0
export CHATGLM_LOAD_IN_4BIT=0
export QA_MODEL_DEVICE_SPLIT="${QA_MODEL_DEVICE_SPLIT:-0}"
export QA_LITE_FORCE_CLASS="${QA_LITE_FORCE_CLASS:-E}"
export LLM_MODEL_DIR="${LLM_MODEL_DIR:-$SCRIPT_DIR/data/pretrained_models/chatglm2-6b}"

PY="${PYTHON:-python3}"
echo "[LITE] NL2SQL + stub chat. http://127.0.0.1:7860  (QA_LITE_FORCE_CLASS=${QA_LITE_FORCE_CLASS})"
echo "Stop: Ctrl+C"
echo "OOM: QA_LITE_FORCE_CLASS=E, or QA_LITE_NL2SQL_DEVICE=cpu, or CHATGLM_LOAD_IN_4BIT=1"
echo ""

exec "$PY" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
