"""
端到端「答题」编排：
- A–E：SQL 检索链路（见 sql_answer）
- F：信息检索 + 生成（见 retrieval）；无知识库时回退纯生成
"""
import copy
import json
import os
import re
from datetime import datetime

from loguru import logger

import re_util
from config import cfg
from procurement_questions import load_test_questions
from company_table import get_sql_search_cursor, load_company_table

from answers.constants import OPEN_QA_CLASS, SQL_TRIGGER_CLASSES
from answers import sql_answer
from answers import retrieval


def generate_answer(model):
    """与旧版 generate_answer 等价，便于 03 脚本直接调用。"""
    logger.info("Generate answers (A–E: SQL, F: retrieval+RAG)...")
    test_questions = load_test_questions()

    sql_cursor = get_sql_search_cursor()
    _ = list(load_company_table().columns)
    logger.info("company_table columns loaded for SQL branch")

    answer_dir = os.path.join(cfg.DATA_PATH, "answers")
    if not os.path.exists(answer_dir):
        os.mkdir(answer_dir)

    for question in test_questions:
        class_csv = os.path.join(cfg.DATA_PATH, "classify", "{}.csv".format(question["id"]))
        if os.path.exists(class_csv):
            with open(class_csv, "r", encoding="utf-8") as f:
                class_result = json.load(f)
                question_type = class_result["class"]
        else:
            logger.warning("分类文件不存在!")
            question_type = OPEN_QA_CLASS

        answer_csv = os.path.join(answer_dir, "{}.csv".format(question["id"]))
        ori_question = re.sub("[\\(\\)（）]", "", question["question"])
        answer = "经查询，无法回答{}".format(ori_question)
        strategy = "none"

        logger.opt(colors=True).info(
            "<blue>Start process question {} {}</>".format(question["id"], question["question"].replace("<", ""))
        )
        logger.opt(colors=True).info("<cyan>问题类型 {}</>".format(question_type.replace("<", "")))

        try:
            if question_type in SQL_TRIGGER_CLASSES:
                logger.info("路由: A–E → SQL 检索")
                answer, strategy, _ = sql_answer.compute_sql_branch_answer(
                    question, question_type, model, sql_cursor
                )
                if answer is None:
                    answer = "经查询，无法回答{}".format(ori_question)

            elif question_type == OPEN_QA_CLASS:
                logger.info("路由: F → 信息检索 + 生成")
                answer, strategy, _refs = retrieval.answer_via_retrieval(ori_question, model)
            else:
                logger.warning("未知类别 {}，按 F 处理", question_type)
                answer, strategy, _refs = retrieval.answer_via_retrieval(ori_question, model)

        except Exception as e:
            logger.exception(e)

        result = copy.deepcopy(question)
        result["answer"] = answer if answer is not None else ""
        result["answer_strategy"] = strategy

        with open(answer_csv, "w", encoding="utf-8") as f:
            try:
                json.dump(result, f, ensure_ascii=False)
            except Exception:
                result["answer"] = ""
                json.dump(result, f, ensure_ascii=False)


def make_answer():
    """汇总 answers/*.csv → result_YYYYMMDD.json（与旧版一致）。"""
    answers = []
    test_questions = load_test_questions()
    answer_dir = os.path.join(cfg.DATA_PATH, "answers")

    for question in test_questions:
        answer_csv = os.path.join(answer_dir, "{}.csv".format(question["id"]))
        if os.path.exists(answer_csv):
            with open(answer_csv, "r", encoding="utf-8") as f:
                answer = json.load(f)
                question = answer
        else:
            question["answer"] = ""

        question["answer"] = re_util.rewrite_answer(question["answer"])
        answers.append(question)

    save_path = os.path.join(cfg.DATA_PATH, "result_{}.json".format(datetime.now().strftime("%Y%m%d")))
    with open(save_path, "w", encoding="utf-8") as f:
        for answer in answers:
            try:
                line = json.dumps(answer, ensure_ascii=False).encode("utf-8").decode() + "\n"
            except Exception:
                answer["answer"] = ""
                line = json.dumps(answer, ensure_ascii=False).encode("utf-8").decode() + "\n"
            f.write(line)
