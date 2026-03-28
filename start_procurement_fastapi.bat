@echo off
REM ASCII-only: avoid UTF-8 batch garbling on Chinese Windows cmd.
chcp 65001 >nul
cd /d "%~dp0"
set LLM_MODEL_DIR=G:\Models\chatglm2-6b
set PYTHONUNBUFFERED=1
REM PyTorch CUDA allocator（Windows 上 expandable_segments 可能告警，可删本行）
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REM 默认分类+NL2SQL+底座均走 GPU；显存紧可 set QA_CLASSIFY_DEVICE=cpu 并保留 8bit
set QA_CLASSIFY_DEVICE=cuda
set CHATGLM_LOAD_IN_8BIT=1
REM 多模型分流标记；单卡全 GPU 时常用 0
set QA_MODEL_DEVICE_SPLIT=0
set QA_CHATGLM_DEVICE=cuda
set QA_CHATGLM_FP16=1
REM 若仍 OOM：set QA_CHATGLM_DEVICE=cpu（底座改 CPU，需大内存）或 QA_LITE_NL2SQL_ONLY=1
echo [Full] ChatGLM + P-Tuning. Not smoke.
echo FastAPI + static page. Port 7860
echo Browser: http://127.0.0.1:7860
echo.
echo Keep this window open. Stop: Ctrl+C
echo.
echo LLM_MODEL_DIR default G:\Models\chatglm2-6b. See config\cfg.py
echo If Loading checkpoint shards crashes: pip install accelerate
echo.
if exist D:\Python310\python.exe (set "PY=D:\Python310\python.exe") else (set "PY=python")
"%PY%" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" (
  echo [Exit] code=%EXITCODE% copy errors from this window for debug.
) else (
  echo [Exit] server stopped.
)
pause
