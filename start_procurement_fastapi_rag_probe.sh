#!/usr/bin/env bash
# Linux: RAG probe — base ChatGLM only + retrieval/RAG (same intent as .bat)
# chmod +x start_procurement_fastapi_rag_probe.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONUNBUFFERED=1
export QA_RAG_PROBE=1
export LLM_MODEL_DIR="${LLM_MODEL_DIR:-$SCRIPT_DIR/data/pretrained_models/chatglm2-6b}"
# Optional: export POLICY_VECTOR_INDEX_DIR=/path/to/vector_index_policy

PY="${PYTHON:-python3}"
echo "[RAG probe] QA_RAG_PROBE=1 — base model + retrieval/RAG. Port 7860"
echo "Browser: http://127.0.0.1:7860"
echo ""

exec "$PY" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
