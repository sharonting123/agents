@echo off
REM ASCII-only. Smoke: no model load. /api/chat returns 503. Use start_procurement_fastapi.bat for full QA.
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
if exist D:\Python310\python.exe (set "PY=D:\Python310\python.exe") else (set "PY=python")
"%PY%" qa_fastapi.py --smoke --host 127.0.0.1 --port 7860
pause
