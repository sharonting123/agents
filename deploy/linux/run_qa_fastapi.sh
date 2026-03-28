#!/usr/bin/env bash
# 在「Code」目录的上一级调用，或先 cd 到 Code 再执行本脚本（需把路径改成你的部署路径）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_DIR}"

if [[ -f "${SCRIPT_DIR}/env.local" ]]; then
  # shellcheck source=/dev/null
  set -a && source "${SCRIPT_DIR}/env.local" && set +a
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export QA_MODEL_DEVICE_SPLIT="${QA_MODEL_DEVICE_SPLIT:-1}"
export QA_CHATGLM_FP16="${QA_CHATGLM_FP16:-1}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
GPU="${GPU:-0}"

exec python3 qa_fastapi.py --gpu "${GPU}" --host "${HOST}" --port "${PORT}" "$@"
