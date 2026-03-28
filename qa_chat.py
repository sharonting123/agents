#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式问答（联调测试）：完整模式为 P-Tuning 分类（A–E→NL2SQL，F→检索+RAG）；精简模式 1 为 QA_LITE_*；RAG 探针（QA_RAG_PROBE）仅测向量检索+RAG。

用法（在 Code 目录下，使用已安装 torch/transformers 的环境）:
  python qa_chat.py
  python qa_chat.py --gpu 0
  python qa_chat.py -q "2024年上海有哪些中标公告？"

浏览器界面：FastAPI + HTML/CSS/JS 见 qa_fastapi.py（推荐）；Gradio 见 qa_gradio.py。

依赖：与主项目一致；MySQL 时需 pymysql（cfg.USE_MYSQL_FOR_SQL）。
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple


def run_one_round(
    q: str,
    classify_model: Any,
    sql_model: Any,
    chat_model: Any,
    sql_cursor: Any,
    agent_mode: Optional[str] = None,
) -> Tuple[str, str, str, str, str, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """
    单轮问答：完整模式下先 P-Tuning 分类，A–E→NL2SQL，F→检索；NL2SQL 异常可兜底 F。
    精简模式下 classify_model 传 None，内部走规则/QA_LITE_FORCE_CLASS。

    agent_mode（仅完整模式且非 RAG 探针时生效）:
        None / auto — 走 P-Tuning 分类；
        nl2sql / sql / ae — 跳过分类，直接 NL2SQL（库表查询）；
        policy / f — 跳过分类，直接向量检索 + 生成（政策咨询 F）。

    Returns:
        answer, strategy, route, raw_classification_snippet, normalized_class, chart_option_or_none, references_or_none
        references：政策/F 类检索时从材料中解析的 {{title, url}}，供前端展示链接。
    """
    from config import cfg
    from generate_answer_with_classify import (
        _rule_only_class,
        normalize_ptuning_classify_label,
    )
    from answers.constants import OPEN_QA_CLASS, SQL_TRIGGER_CLASSES
    from answers import retrieval
    from answers.sql_answer import answer_with_nl2sql_model
    from loguru import logger

    def _nl2sql_fail_user_msg(exc: BaseException) -> str:
        return (
            "NL2SQL 调用失败且未加载底座，无法兜底。"
            f" 错误: {exc} "
            "可选：设置 QA_LITE_FORCE_CLASS=F（需第二套底座显存，失败时走 RAG）；"
            "或设 QA_NL2SQL_FP32=1 再试；确认 CHATGLM_LOAD_IN_8BIT=0。"
        )

    q = (q or "").strip()
    if not q:
        return "", "", "", "", "", None, None

    if getattr(cfg, "QA_RAG_PROBE", False):
        if getattr(chat_model, "model", None) is None:
            return (
                "RAG 探针模式需要已加载底座模型（ChatGLM Nothing）。请检查 LLM_MODEL_DIR。",
                "rag_probe_no_chat",
                "QA_RAG_PROBE 未加载底座",
                "",
                OPEN_QA_CLASS,
                None,
                None,
            )
        answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
        return (
            answer,
            strategy,
            "RAG探针（QA_RAG_PROBE）→ 向量检索 + 生成",
            "(跳过分类与 NL2SQL，仅测 F 链路)",
            OPEN_QA_CLASS,
            None,
            policy_refs or None,
        )

    if getattr(cfg, "QA_LITE_NL2SQL_ONLY", False):
        force = getattr(cfg, "QA_LITE_FORCE_CLASS", "E")
        if force not in ("A", "B", "C", "D", "E", "F"):
            force = "A"
        if force in ("B", "C", "D", "E"):
            force = "A"
        rule_qt = _rule_only_class(q)

        def _has_chat_base() -> bool:
            return getattr(chat_model, "model", None) is not None

        # 规则先验为 F：开放问答（已加载底座时走检索+RAG，否则提示）
        if rule_qt == OPEN_QA_CLASS:
            route = "精简模式 → 规则先验 F"
            raw_preview = "(规则先验→F)"
            if _has_chat_base():
                answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                return answer, strategy, route, raw_preview, "F", None, policy_refs or None
            answer = (
                "你好，我是政府采购投标助手。当前为精简模式，未加载 6B 底座，无法进行开放问答。"
                "请设置 QA_LITE_FORCE_CLASS=F 并重启（见 start_procurement_fastapi_lite.bat），"
                "或使用完整启动 start_procurement_fastapi.bat。"
            )
            return answer, "rule_f_lite_no_sql", route, "(规则先验 F，未加载底座)", "F", None, None

        # 规则先验为 A（库表）：始终走 NL2SQL（与 QA_LITE_FORCE_CLASS 无关）；失败则兜底 F
        if rule_qt is not None and rule_qt in SQL_TRIGGER_CLASSES:
            route = "精简模式 → 规则先验 A（库表）→ NL2SQL"
            raw_preview = f"(规则先验→{rule_qt})"
            try:
                answer, strategy, chart = answer_with_nl2sql_model(
                    q, sql_model, chat_model, sql_cursor
                )
            except Exception as e:
                logger.warning("精简模式 NL2SQL 失败，兜底 F: {}", e)
                raw_preview = f"{raw_preview} | NL2SQL异常: {str(e)[:300]}"
                if _has_chat_base():
                    answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                    return (
                        answer,
                        strategy,
                        "精简模式 → NL2SQL失败→兜底F（向量检索）",
                        raw_preview,
                        OPEN_QA_CLASS,
                        None,
                        policy_refs or None,
                    )
                return (
                    _nl2sql_fail_user_msg(e),
                    "nl2sql_error_no_chat_fallback",
                    route,
                    raw_preview,
                    OPEN_QA_CLASS,
                    None,
                    None,
                )
            display = answer if answer else "(空)"
            return display, strategy, route, raw_preview, rule_qt, chart, None

        # 规则未命中：按 QA_LITE_FORCE_CLASS
        if force == OPEN_QA_CLASS:
            route = "精简模式 → 固定 F（开放问答）"
            raw_preview = "(精简模式：QA_LITE_FORCE_CLASS=F)"
            if _has_chat_base():
                answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                return answer, strategy, route, raw_preview, "F", None, policy_refs or None
            answer = (
                "精简模式已设 QA_LITE_FORCE_CLASS=F，但未成功加载底座模型，无法开放问答。"
                "请检查 LLM_MODEL_DIR 与显存。"
            )
            return answer, "rule_f_lite_no_sql", route, raw_preview, "F", None, None

        qt = force
        if qt not in SQL_TRIGGER_CLASSES and qt != OPEN_QA_CLASS:
            qt = "A"
        raw_preview = f"(精简模式，固定意图={qt}，未调用底座意图模型)"
        route = "精简模式 → NL2SQL + 执行 SQL"
        try:
            answer, strategy, chart = answer_with_nl2sql_model(
                q, sql_model, chat_model, sql_cursor
            )
        except Exception as e:
            logger.warning("精简模式 NL2SQL 失败，兜底 F: {}", e)
            raw_preview = f"{raw_preview} | NL2SQL异常: {str(e)[:300]}"
            if _has_chat_base():
                answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                return (
                    answer,
                    strategy,
                    "精简模式 → NL2SQL失败→兜底F（向量检索）",
                    raw_preview,
                    OPEN_QA_CLASS,
                    None,
                    policy_refs or None,
                )
            return (
                _nl2sql_fail_user_msg(e),
                "nl2sql_error_no_chat_fallback",
                route,
                raw_preview,
                OPEN_QA_CLASS,
                None,
                None,
            )
        display = answer if answer else "(空)"
        return display, strategy, route, raw_preview, qt, chart, None

    # 前端「智能体」：强制库表 NL2SQL 或政策咨询（F），不经过 P-Tuning 分类
    if not getattr(cfg, "QA_LITE_NL2SQL_ONLY", False) and not getattr(
        cfg, "QA_RAG_PROBE", False
    ):
        am = (agent_mode or "").strip().lower()
        if am in ("auto", "none", ""):
            am = ""
        if am in ("nl2sql", "sql", "ae", "knowledge", "kb"):
            raw_preview = "(手动：库表查询 / NL2SQL)"
            route = "前端指定 → 库表查询（NL2SQL）"
            qt = "A"
            if sql_cursor is None:
                return (
                    "数据库未连接，无法执行库表查询。请配置 MySQL（USE_MYSQL_FOR_SQL）或在 data 下放置 CompanyTable.csv。",
                    "db_unavailable",
                    route,
                    raw_preview,
                    qt,
                    None,
                    None,
                )
            try:
                answer, strategy, chart = answer_with_nl2sql_model(
                    q, sql_model, chat_model, sql_cursor
                )
            except Exception as e:
                logger.warning("前端指定 NL2SQL 失败，兜底 F: {}", e)
                raw_preview = f"{raw_preview} | NL2SQL异常: {str(e)[:300]}"
                if getattr(chat_model, "model", None) is not None:
                    route = "前端指定 NL2SQL → 失败，兜底 F（向量检索）"
                    answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                    display = answer if answer else "(空)"
                    return display, strategy, route, raw_preview, OPEN_QA_CLASS, None, policy_refs or None
                return (
                    _nl2sql_fail_user_msg(e),
                    "nl2sql_error_no_chat_fallback",
                    route,
                    raw_preview,
                    OPEN_QA_CLASS,
                    None,
                    None,
                )
            display = answer if answer else "(空)"
            return display, strategy, route, raw_preview, qt, chart, None
        if am in ("policy", "f", "rag"):
            raw_preview = "(手动：政策咨询 / 向量检索)"
            route = "前端指定 → 政策咨询（F）"
            if getattr(chat_model, "model", None) is None:
                return (
                    "当前未加载底座模型，无法进行政策咨询。请完整启动并确认显存与 LLM_MODEL_DIR。",
                    "policy_no_chat",
                    route,
                    raw_preview,
                    OPEN_QA_CLASS,
                    None,
                    None,
                )
            answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
            display = answer if answer else "(空)"
            return display, strategy, route, raw_preview, OPEN_QA_CLASS, None, policy_refs or None

    # 完整模式：先 P-Tuning 分类；A–E → NL2SQL；F（或其它）→ 向量检索 + 生成
    chart: Optional[Dict[str, Any]] = None
    raw = ""
    if classify_model is None or not getattr(classify_model, "isClassify", False):
        logger.error("完整模式需要分类 P-Tuning：请关闭 QA_LITE_NL2SQL_ONLY 并确认 CLASSIFY_CHECKPOINT_PATH")
        return (
            "配置错误：未加载分类 P-Tuning。",
            "config_error",
            "未加载 classify",
            "",
            OPEN_QA_CLASS,
            None,
            None,
        )
    try:
        raw = classify_model.classify(q) or ""
        raw_preview = (raw or "").strip().replace("\n", " ")[:500]
    except Exception as e:
        logger.warning("分类 P-Tuning 调用失败，尝试规则先验→NL2SQL（若仍失败再兜底 F）: {}", e)
        raw_preview = f"(分类异常) {str(e)[:300]}"
        rule_qt = _rule_only_class(q)
        if rule_qt is not None and rule_qt in SQL_TRIGGER_CLASSES:
            route = f"分类异常 → 规则先验 {rule_qt} → NL2SQL"
            try:
                answer, strategy, chart = answer_with_nl2sql_model(
                    q, sql_model, chat_model, sql_cursor
                )
            except Exception as e2:
                logger.warning("规则补路径 NL2SQL 失败，兜底 F: {}", e2)
                raw_preview = f"{raw_preview} | NL2SQL异常: {str(e2)[:300]}"
                if getattr(chat_model, "model", None) is not None:
                    route = "分类失败→规则A–E→NL2SQL仍失败 → 兜底 F（向量检索）"
                    answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                    display = answer if answer else "(空)"
                    return display, strategy, route, raw_preview, OPEN_QA_CLASS, None, policy_refs or None
                return (
                    f"分类与 NL2SQL 均失败。分类: {e}; NL2SQL: {e2}",
                    "classify_then_nl2sql_error_no_chat",
                    route,
                    raw_preview,
                    OPEN_QA_CLASS,
                    None,
                    None,
                )
            display = answer if answer else "(空)"
            return display, strategy, route, raw_preview, rule_qt, chart, None
        if rule_qt == OPEN_QA_CLASS and getattr(chat_model, "model", None) is not None:
            route = "分类失败 → 规则先验 F → 向量检索"
            answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
            display = answer if answer else "(空)"
            return display, strategy, route, raw_preview, OPEN_QA_CLASS, None, policy_refs or None
        if getattr(chat_model, "model", None) is not None:
            route = "分类失败 → 兜底 F（向量检索）"
            answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
            display = answer if answer else "(空)"
            return display, strategy, route, raw_preview, OPEN_QA_CLASS, None, policy_refs or None
        return (
            f"分类模型调用失败，且未加载底座无法兜底。错误: {e}",
            "classify_error_no_chat",
            "分类失败",
            raw_preview,
            OPEN_QA_CLASS,
            None,
            None,
        )

    qt = normalize_ptuning_classify_label(raw)
    policy_refs: Optional[List[Dict[str, Any]]] = None
    if qt in SQL_TRIGGER_CLASSES:
        route = f"P-Tuning 分类 {qt}（A–E）→ NL2SQL"
        try:
            answer, strategy, chart = answer_with_nl2sql_model(
                q, sql_model, chat_model, sql_cursor
            )
        except Exception as e:
            logger.warning("NL2SQL 失败，兜底 F: {}", e)
            raw_preview = f"{raw_preview} | NL2SQL异常: {str(e)[:300]}"
            if getattr(chat_model, "model", None) is not None:
                route = "NL2SQL 失败 → 兜底 F（向量检索）"
                answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
                qt = OPEN_QA_CLASS
                chart = None
            else:
                answer = _nl2sql_fail_user_msg(e)
                strategy = "nl2sql_error_no_chat_fallback"
                qt = OPEN_QA_CLASS
                chart = None
                policy_refs = None
    else:
        route = "P-Tuning 分类 F → 向量检索 + 生成"
        answer, strategy, policy_refs = retrieval.answer_via_retrieval(q, chat_model)
        qt = OPEN_QA_CLASS

    display = answer if answer else "(空)"
    return display, strategy, route, raw_preview, qt, chart, policy_refs


def _parse_args():
    p = argparse.ArgumentParser(description="交互式问答测试")
    p.add_argument("--gpu", type=int, default=0, help="CUDA 设备序号")
    p.add_argument(
        "-q",
        "--question",
        type=str,
        default=None,
        help="只问一条后退出（非交互）；不设则进入多轮输入",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from loguru import logger

    from config import cfg
    from chatglm_ptuning import build_qa_models
    from company_table import get_sql_search_cursor, load_company_table

    d_cls, d_sql, d_chat = cfg.qa_chatglm_device_split()
    logger.info(
        "加载模型（分类 P-Tuning + NL2SQL + 底座），device: {}, {}, {}",
        d_cls,
        d_sql,
        d_chat,
    )
    cls_model, sql_model, chat_model = build_qa_models()

    logger.info("连接数据库并预热列信息…")
    sql_cursor = get_sql_search_cursor()
    try:
        _ = list(load_company_table().columns)
    except Exception as e:
        logger.warning("load_company_table 失败（SQL 纠错可能受限）: {}", e)

    def one_round(q: str) -> None:
        if not (q or "").strip():
            return
        answer, strategy, route, raw_preview, qt, chart, _refs = run_one_round(
            q, cls_model, sql_model, chat_model, sql_cursor
        )
        print("\n[意图] 模型输出: {}  →  规整: {}".format(raw_preview[:80], qt))
        print("[路由]", route)
        print("[策略]", strategy)
        print("[回答]", answer if answer else "(空)")
        if chart:
            print("[图表]", "已生成 ECharts 配置（Web 端展示）")

    if args.question:
        one_round(args.question)
        return

    print("就绪。输入问题，空行退出。Ctrl+C 结束。\n")
    while True:
        try:
            line = input("问题> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            break
        one_round(line)

    if cls_model is not None:
        cls_model.unload_model()
    if sql_model is not None:
        sql_model.unload_model()
    chat_model.unload_model()
    logger.info("已卸载模型。")


if __name__ == "__main__":
    main()
