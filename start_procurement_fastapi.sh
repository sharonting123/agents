#!/usr/bin/env bash
# Linux: full ChatGLM + P-Tuning (same intent as start_procurement_fastapi.bat)
# chmod +x start_procurement_fastapi.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 默认三套 P-Tuning/底座均走 GPU；显存不足再设 QA_CLASSIFY_DEVICE=cpu 等
export QA_CLASSIFY_DEVICE="${QA_CLASSIFY_DEVICE:-cuda}"
export CHATGLM_LOAD_IN_8BIT="${CHATGLM_LOAD_IN_8BIT:-1}"
export QA_MODEL_DEVICE_SPLIT="${QA_MODEL_DEVICE_SPLIT:-0}"
export QA_CHATGLM_DEVICE="${QA_CHATGLM_DEVICE:-cuda}"
export QA_CHATGLM_FP16="${QA_CHATGLM_FP16:-1}"
# Override if needed: export LLM_MODEL_DIR=/path/to/chatglm2-6b
export LLM_MODEL_DIR="${LLM_MODEL_DIR:-$SCRIPT_DIR/data/pretrained_models/chatglm2-6b}"

PY="${PYTHON:-python3}"
echo "[Full] ChatGLM + P-Tuning. Not smoke."
echo "FastAPI + static page. Port 7860"
echo "Browser: http://127.0.0.1:7860"
echo ""
echo "Stop: Ctrl+C"
echo "LLM_MODEL_DIR=$LLM_MODEL_DIR (override env LLM_MODEL_DIR if needed)"
echo "If Loading checkpoint shards fails: pip install accelerate"
echo ""

exec "$PY" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
