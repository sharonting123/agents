@echo off
REM ASCII-only: UTF-8 batch can break under cmd without BOM.
cd /d "%~dp0"
set LLM_MODEL_DIR=G:\Models\chatglm2-6b
set PYTHONUNBUFFERED=1
set QA_MODEL_DEVICE_SPLIT=0
set QA_CHATGLM_DEVICE=cuda
set QA_CHATGLM_FP16=1
if exist D:\Python310\python.exe (set "PY=D:\Python310\python.exe") else (set "PY=python")
echo Starting minimized window. http://127.0.0.1:7860
start "Procurement-FastAPI" /MIN "%PY%" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
echo Do not close the minimized window or the server stops.
