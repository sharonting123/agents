#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政府采购投标助手 — FastAPI + 静态 HTML/CSS/JS。

启动（在 Code 目录下）:
  pip install fastapi uvicorn[standard]

完整加载 ChatGLM（可正常 POST /api/chat，勿加 --smoke）:
  python qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
  或双击 start_procurement_fastapi.bat（已设 QA_MODEL_DEVICE_SPLIT、QA_CHATGLM_FP16）

浏览器: http://127.0.0.1:7860
API: POST /api/chat  JSON {"message": "..."}，可选 agent_mode: "auto"|"nl2sql"|"policy"

仅测页面/连通性（不加载模型，聊天会 503）:
  python qa_fastapi.py --smoke --host 127.0.0.1 --port 7860
  浏览器同样打开: http://127.0.0.1:7860

完整模式默认「分类 + NL2SQL + 底座」均在 GPU（见 start_procurement_fastapi.sh 的 QA_CLASSIFY_DEVICE=cuda）；显存不足再设 QA_CLASSIFY_DEVICE=cpu 等。建议 QA_CHATGLM_FP16=1，必要时 CHATGLM_LOAD_IN_8BIT=1 或精简模式 QA_LITE_NL2SQL_ONLY=1。

Linux 部署：见 deploy/linux/README.md（路径用 CODE_BASE_DIR / LLM_MODEL_DIR 等环境变量）。

仅测 NL2SQL（少加载两套模型、启动更快）：设 QA_LITE_NL2SQL_ONLY=1，可选 QA_LITE_NL2SQL_DEVICE=cuda。

RAG 探针（精简模式 2）：仅测向量检索+RAG 链路，设 QA_RAG_PROBE=1（只加载底座，不要数据库）。见 start_procurement_fastapi_rag_probe.bat。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from user_backend.deps import get_current_user_optional
from user_backend.models import User
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

WEB_DIR = Path(__file__).resolve().parent / "web"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户问题")
    agent_mode: str | None = Field(
        default=None,
        description="auto/空=自动分类；nl2sql=强制库表 NL2SQL；policy=强制政策咨询 F",
    )
    conversation_id: int | None = Field(
        default=None,
        description="登录用户多轮会话时传入上次返回的 conversation_id；未登录忽略",
    )


class ChatResponse(BaseModel):
    answer: str
    strategy: str = ""
    route: str = ""
    raw_classification: str = ""
    normalized_class: str = ""
    error: str | None = None
    # NL2SQL 数据分析：ECharts option（仅当问题含图表/可视化等意图且可绘图时）
    chart: dict | None = None
    # 政策/F 类检索：从参考材料解析的引用链接（title + url）
    references: list[dict] | None = None
    # 已登录且落库成功时返回，供下一轮 conversation_id 续聊
    conversation_id: int | None = None
    assistant_message_id: int | None = None


def build_app(gpu: int, *, smoke: bool = False) -> FastAPI:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from loguru import logger

        try:
            from user_backend.database import init_db, user_app_backend_label

            init_db()
            logger.info("用户应用库: {}", user_app_backend_label())
        except Exception as e:
            logger.warning("用户应用库初始化失败（注册/审计可能不可用）: {}", e)

        app.state.chat_lock = asyncio.Lock()
        app.state.smoke = smoke

        if smoke:
            logger.warning(
                "--smoke：不加载 ChatGLM，仅验证 Web/API；问答请用完整启动（勿带 --smoke）"
            )
            app.state.cls_model = None  # 完整模式为分类 P-Tuning；smoke 不加载
            app.state.sql_model = None
            app.state.chat_model = None
            app.state.run_one_round = None
            app.state.sql_cursor = None
            app.state.models_loaded = False
            yield
            return

        from company_table import get_sql_search_cursor, load_company_table
        from config import cfg
        from chatglm_ptuning import build_qa_models
        from qa_chat import run_one_round

        d_cls, d_sql, d_chat = cfg.qa_chatglm_device_split()
        if getattr(cfg, "QA_RAG_PROBE", False):
            logger.info("QA_RAG_PROBE：仅加载底座（RAG 探针）device: {}", d_chat)
        else:
            logger.info("加载 P-Tuning 模型（分类 / NL2SQL / 底座）device: {}, {}, {}", d_cls, d_sql, d_chat)
        cls_model, sql_model, chat_model = build_qa_models()
        app.state.cls_model = cls_model
        app.state.sql_model = sql_model
        app.state.chat_model = chat_model
        app.state.run_one_round = run_one_round
        # 与 DB 解耦：仅因 MySQL/CSV 失败时不应让前端误判为 smoke 或未加载模型
        app.state.models_loaded = True

        if getattr(cfg, "QA_RAG_PROBE", False):
            logger.info("QA_RAG_PROBE：跳过数据库（仅测 RAG）")
            app.state.sql_cursor = None
        else:
            logger.info("连接数据库并预热列信息…")
            try:
                app.state.sql_cursor = get_sql_search_cursor()
            except Exception as e:
                logger.error("get_sql_search_cursor 失败（问答中需 SQL 的路径将不可用）: {}", e)
                app.state.sql_cursor = None
            try:
                _ = list(load_company_table().columns)
            except Exception as e:
                logger.warning("load_company_table 失败（SQL 纠错可能受限）: {}", e)

        yield

        cls_m = getattr(app.state, "cls_model", None)
        sql_m = getattr(app.state, "sql_model", None)
        if cls_m is not None:
            cls_m.unload_model()
        if sql_m is not None:
            sql_m.unload_model()
        chat_m = getattr(app.state, "chat_model", None)
        if chat_m is not None:
            chat_m.unload_model()
        logger.info("已卸载模型。")
        # 不在此处将 models_loaded 置 False，避免关闭过程中仍有 /api/health 请求时误判为「未加载」

    app = FastAPI(title="甄甄 · 政府采购投标助手", lifespan=lifespan)

    from user_backend.routers import router as user_app_router

    app.include_router(user_app_router)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        from config import cfg

        loaded = getattr(app.state, "models_loaded", False)
        cur = getattr(app.state, "sql_cursor", None)
        return {
            "ok": True,
            "models_loaded": bool(loaded),
            "smoke": bool(getattr(app.state, "smoke", False)),
            "sql_cursor_ok": cur is not None,
            "qa_lite_nl2sql_only": bool(getattr(cfg, "QA_LITE_NL2SQL_ONLY", False)),
            "qa_rag_probe": bool(getattr(cfg, "QA_RAG_PROBE", False)),
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(
        req: ChatRequest,
        current_user: User | None = Depends(get_current_user_optional),
    ) -> ChatResponse:
        if not getattr(app.state, "models_loaded", False):
            raise HTTPException(
                status_code=503,
                detail="模型未加载（若使用 --smoke 启动则仅用于连通性测试，请去掉 --smoke 完整启动）",
            )

        from config import cfg

        if (
            getattr(cfg, "QA_LITE_NL2SQL_ONLY", False)
            and not getattr(cfg, "QA_RAG_PROBE", False)
            and getattr(app.state, "sql_cursor", None) is None
        ):
            raise HTTPException(
                status_code=503,
                detail="数据库未连接：精简模式需要执行 NL2SQL。请配置 MySQL（USE_MYSQL_FOR_SQL）或在 data 下放置 CompanyTable.csv。",
            )

        msg = (req.message or "").strip()
        if not msg:
            raise HTTPException(status_code=400, detail="message 为空")

        from loguru import logger

        from user_backend.chat_service import persist_chat_turn_safe

        lock: asyncio.Lock = app.state.chat_lock
        t0 = time.perf_counter()
        async with lock:
            loop = asyncio.get_event_loop()

            def _run():
                return app.state.run_one_round(
                    msg,
                    app.state.cls_model,
                    app.state.sql_model,
                    app.state.chat_model,
                    app.state.sql_cursor,
                    agent_mode=getattr(req, "agent_mode", None),
                )

            try:
                answer, strategy, route, raw_preview, qt, chart, refs = await loop.run_in_executor(
                    None, _run
                )
            except Exception as e:
                logger.exception("问答失败")
                raise HTTPException(status_code=500, detail=str(e)) from e

            latency_ms = int((time.perf_counter() - t0) * 1000)

            logger.info(
                "[chat] 规整类别={} · 路由={} · 策略={} · 分类原始（截断）: {}",
                qt or "",
                route or "",
                strategy or "",
                (raw_preview or "")[:500],
            )

        conv_id: int | None = None
        asst_msg_id: int | None = None
        if current_user is not None:
            r = persist_chat_turn_safe(
                user_id=current_user.id,
                conversation_id=getattr(req, "conversation_id", None),
                user_text=msg,
                assistant_text=answer or "",
                raw_preview=raw_preview or "",
                qt=qt or "",
                strategy=strategy or "",
                route=route or "",
                agent_mode=getattr(req, "agent_mode", None),
                latency_ms=latency_ms,
                chart=chart,
                refs=refs if refs else None,
            )
            if r:
                conv_id = r.get("conversation_id")
                asst_msg_id = r.get("assistant_message_id")

        return ChatResponse(
            answer=answer or "(空)",
            strategy=strategy or "",
            route=route or "",
            raw_classification=raw_preview or "",
            normalized_class=qt or "",
            chart=chart,
            references=refs if refs else None,
            conversation_id=conv_id,
            assistant_message_id=asst_msg_id,
        )

    @app.get("/")
    async def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/login")
    async def login_page():
        return FileResponse(WEB_DIR / "login.html")

    app.mount(
        "/static",
        StaticFiles(directory=WEB_DIR / "static"),
        name="static",
    )

    return app


def _parse_args():
    p = argparse.ArgumentParser(description="FastAPI 政府采购投标助手")
    p.add_argument("--gpu", type=int, default=0, help="CUDA 设备序号")
    p.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    p.add_argument("--port", type=int, default=7860, help="端口")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="不加载 ChatGLM，仅验证服务与静态页（/api/chat 返回 503）",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    import uvicorn

    if args.smoke:
        print(
            f"[smoke] 本机浏览器打开: http://127.0.0.1:{args.port}/\n"
            f"        监听 {args.host}:{args.port}；首页与 /static 可用，/api/chat 未加载模型会 503",
            flush=True,
        )

    app = build_app(args.gpu, smoke=args.smoke)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
