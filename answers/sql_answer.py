"""A（知识库/SQL，含历史 B–E 标签）：NL2SQL → 执行 → 可选纠错。"""
import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from loguru import logger

import prompt_util
import sql_correct_util
from config import cfg
from answers.constants import SQL_TRIGGER_CLASSES


def _default_answer_placeholder(ori_question: str) -> str:
    return "经查询，无法回答{}".format(ori_question)


def _execute_sql_with_correction(
    ori_question: str,
    sql: str,
    model: Any,
    sql_cursor: Any,
) -> Tuple[Optional[str], str, Optional[str]]:
    """执行 SQL，失败时按与批量答题相同的逻辑尝试字段/模型纠错（兼容 SQLite 与 MySQL 报错文案）。
    第三项为最后一次成功执行所用的 SQL（用于 exc_sql_rows 出图）。"""
    sql = sql_correct_util.correct_sql_number(sql, ori_question)
    answer, exec_log = sql_correct_util.exc_sql(ori_question, sql, sql_cursor)
    last_err = exec_log
    last_good_sql: Optional[str] = sql if answer is not None else None

    if answer is None:
        try:
            low = (exec_log or "").lower()
            if "no such column" in low or "unknown column" in low:
                sql = sql_correct_util.correct_sql_field(sql, ori_question, model)
                answer, exec_log = sql_correct_util.exc_sql(ori_question, sql, sql_cursor)
                last_err = exec_log or last_err
                if answer is not None:
                    last_good_sql = sql
            else:
                logger.info("模型纠正前sql：{}".format(sql.replace("<>", "")))
                key_words = prompt_util.build_sql_column_catalog()
                correct_sql_answer = model(
                    prompt_util.prompt_sql_correct.format(key_words, sql, exec_log)
                )
                logger.info("模型纠正sql结果：{}".format(correct_sql_answer.replace("<>", "")))
                sql_blocks = re.findall("```sql([\\s\\S]+)```", correct_sql_answer)
                if len(sql_blocks) > 0:
                    sql = sql_blocks[0].replace("\n", "").strip()
                logger.info("模型纠正后sql：{}".format(sql.replace("<>", "")))
                answer, exec_log = sql_correct_util.exc_sql(ori_question, sql, sql_cursor)
                last_err = exec_log or last_err
                if answer is not None:
                    last_good_sql = sql
        except Exception as e:
            logger.error("纠正SQL[{}]错误! {}".format(sql.replace("<>", ""), e))

    if answer is None:
        logger.warning(
            "SQL 执行失败且无可用结果，请向上翻看 ERROR「执行SQL[...]」或此处 last_err={}",
            last_err,
        )

    logger.opt(colors=True).info("<green>{}</>".format(str(sql).replace("<>", "")))
    logger.opt(colors=True).info("<magenta>{}</>".format(str(answer).replace("<>", "")))
    return answer, "sql", last_good_sql


def answer_with_nl2sql_model(
    question_text: str,
    nl2sql_model: Any,
    chat_model: Any,
    sql_cursor: Any,
) -> Tuple[Optional[str], str, Optional[Dict[str, Any]]]:
    """
    交互/单条测试：不读 data/sql/*.csv，直接用 NL2SQL 模型生成 SQL 再执行。
    第三项为可选 ECharts option（用户问题含图表/可视化等意图且结果可绘图时）。
    """
    from answers.sql_chart import build_echarts_option, question_wants_chart

    ori_question = re.sub("[\\(\\)（）]", "", question_text)
    sql = nl2sql_model.nl2sql(question_text)
    if not sql or not str(sql).strip():
        return _default_answer_placeholder(ori_question), "sql", None
    answer, _, final_sql = _execute_sql_with_correction(ori_question, sql, chat_model, sql_cursor)
    chart: Optional[Dict[str, Any]] = None
    if (
        getattr(cfg, "SQL_CHART_ENABLED", True)
        and sql_cursor is not None
        and final_sql
        and answer
        and answer != "查询无结果"
        and question_wants_chart(question_text)
    ):
        rows, cols, err = sql_correct_util.exc_sql_rows(ori_question, final_sql, sql_cursor)
        if rows is not None and cols and not err:
            chart = build_echarts_option(rows, cols, question_text)
    return answer, "sql", chart


def compute_sql_branch_answer(
    question: dict,
    question_type: str,
    model: Any,
    sql_cursor: Any,
) -> Tuple[Optional[str], str, Optional[Dict[str, Any]]]:
    """
    若 question_type 不在 SQL_TRIGGER_CLASSES，返回 (None, 'skip', None)。
    否则返回 (answer 字符串, 'sql', 可选图表)；失败时 answer 可能为占位句。
    """
    from answers.sql_chart import build_echarts_option, question_wants_chart

    if question_type not in SQL_TRIGGER_CLASSES:
        return None, "skip", None

    ori_question = re.sub("[\\(\\)（）]", "", question["question"])
    answer: Optional[str] = _default_answer_placeholder(ori_question)

    sql_csv = os.path.join(cfg.DATA_PATH, "sql", "{}.csv".format(question["id"]))
    if not os.path.exists(sql_csv):
        logger.warning("SQL 文件不存在: {}", sql_csv)
        return answer, "sql", None

    with open(sql_csv, "r", encoding="utf-8") as f:
        sql_result = json.load(f)
    sql = sql_result.get("sql")
    if sql is None:
        return answer, "sql", None

    answer, _, final_sql = _execute_sql_with_correction(ori_question, sql, model, sql_cursor)
    chart: Optional[Dict[str, Any]] = None
    qtext = question.get("question") or ""
    if (
        getattr(cfg, "SQL_CHART_ENABLED", True)
        and sql_cursor is not None
        and final_sql
        and answer
        and answer != "查询无结果"
        and question_wants_chart(qtext)
    ):
        rows, cols, err = sql_correct_util.exc_sql_rows(ori_question, final_sql, sql_cursor)
        if rows is not None and cols and not err:
            chart = build_echarts_option(rows, cols, qtext)
    return answer, "sql", chart
