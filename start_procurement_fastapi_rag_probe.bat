@echo off
REM RAG probe mode: only load base ChatGLM (Nothing), test vector retrieval + RAG (F route). No classify / NL2SQL / MySQL.
chcp 65001 >nul
cd /d "%~dp0"
set LLM_MODEL_DIR=G:\Models\chatglm2-6b
set PYTHONUNBUFFERED=1
set QA_RAG_PROBE=1
REM Optional: policy FAISS index (default see cfg.POLICY_VECTOR_INDEX_DIR)
REM set POLICY_VECTOR_INDEX_DIR=D:\path\to\vector_index_policy
echo [RAG probe] QA_RAG_PROBE=1 — only base model + retrieval/RAG. Port 7860
echo Browser: http://127.0.0.1:7860
echo.
if exist D:\Python310\python.exe (set "PY=D:\Python310\python.exe") else (set "PY=python")
"%PY%" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
pause
