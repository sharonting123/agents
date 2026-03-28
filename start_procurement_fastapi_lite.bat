@echo off
REM ASCII-only lines: cmd.exe on Chinese Windows may mis-parse UTF-8 without BOM.
REM Use D:\Python310\python.exe if present, else python on PATH.
REM Policy RAG deps: D:\Python310\python.exe -m pip install sentence-transformers faiss-cpu
chcp 65001 >nul
cd /d "%~dp0"
set LLM_MODEL_DIR=G:\Models\chatglm2-6b
set PYTHONUNBUFFERED=1
REM Newer huggingface_hub may mis-treat Windows paths as Hub repo ids; keep local-only when models are on disk.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set QA_LITE_NL2SQL_ONLY=1
set QA_LITE_NL2SQL_DEVICE=cuda
set QA_CHATGLM_DEVICE=cuda
set QA_CHATGLM_FP16=1
REM P-Tuning+NL2SQL 勿与 8bit/device_map 混用（易 meta tensor）；精简仅 NL2SQL 时强制关量化
set CHATGLM_LOAD_IN_8BIT=0
set CHATGLM_LOAD_IN_4BIT=0
REM NL2SQL 仍报 meta tensor 时：set QA_NL2SQL_FP32=1（整模 FP32，更吃显存）
set QA_MODEL_DEVICE_SPLIT=0
REM Default E: one 6B on GPU (NL2SQL). F loads a SECOND full 6B for open chat / F-route (~24GB VRAM on same GPU).
REM If startup crashes around "Loading weights" (exit -1073741819): use E, or set QA_LITE_NL2SQL_DEVICE=cpu, or enable 4bit below.
set QA_LITE_FORCE_CLASS=E
REM NL2SQL 失败需 RAG 兜底时：set QA_LITE_FORCE_CLASS=F（需约双 6B 显存）
REM set QA_LITE_FORCE_CLASS=F
REM pip install bitsandbytes && set CHATGLM_LOAD_IN_4BIT=1
echo [LITE] NL2SQL + stub chat. http://127.0.0.1:7860  (QA_LITE_FORCE_CLASS=E; set =F for second 6B)
echo Keep window open until Uvicorn runs. Stop: Ctrl+C
echo OOM/crash: QA_LITE_FORCE_CLASS=E, or QA_LITE_NL2SQL_DEVICE=cpu, or CHATGLM_LOAD_IN_4BIT=1
echo.
if exist D:\Python310\python.exe (set "PY=D:\Python310\python.exe") else (set "PY=python")
"%PY%" qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
set EXITCODE=%ERRORLEVEL%
echo.
if "%EXITCODE%"=="0" goto OKEXIT
echo [Exit] code=%EXITCODE%  CUDA OOM, bad model path, or missing deps.
goto ENDPAUSE
:OKEXIT
echo [Exit] Stopped normally or Ctrl+C.
:ENDPAUSE
pause
