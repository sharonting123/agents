# [03] 阅读顺序编号文件，对应原文件: generate_answer_with_classify.py
"""
批量评测管线：分类 P-Tuning → classify/*.csv → 关键词（规则）→ sql/*.csv → answers。

在线完整模式仅用 classify_model.classify() + normalize_ptuning_classify_label；
精简模式仍用 _rule_only_class（与批量分类独立）。
"""
import os
import json
import copy
import re
from typing import Optional
from loguru import logger
from config import cfg
from procurement_questions import load_test_questions
from chatglm_ptuning import ChatGLM_Ptuning
from answers.constants import SQL_TRIGGER_CLASSES
from answers.orchestrator import generate_answer, make_answer

# 在线意图合并为 A（库表/SQL）与 F（开放问答）；批量 classify/*.csv 仍可读历史 B–E 标签。


def _zh_field_names_for_keywords() -> list:
    """与 cfg.SQL_EN_TO_ZH_COLUMNS 中文注释一致，用于从用户问题里抽字段关键词（问题仍为中文）。"""
    cmap = getattr(cfg, "SQL_EN_TO_ZH_COLUMNS", None) or {}
    return [v for v in cmap.values() if v]


def _dedupe_keep_order(items):
    seen = set()
    result = []
    for item in items:
        token = (item or '').strip()
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def _parse_model_keywords(raw: str):
    """将 keywords() 模型输出拆成列表（支持顿号/逗号/分号等分隔）。"""
    if not raw:
        return []
    s = raw.strip()
    for sep in ["\n", "、", "，", ",", ";", "；", "|"]:
        s = s.replace(sep, ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return _dedupe_keep_order(parts)


def _build_procurement_keywords(question: str, question_type: str):
    # 政府采购专用关键词模板：字段级对齐 shggzy_bid_result 中文字段名（按 A–F 分类）。
    class_templates = {
        "A": [
            "项目编号",
            "公告标题",
            "发布日期",
            "招标人/采购单位",
            "招标人联系人",
            "招标人联系方式",
            "招标代理机构",
            "招标代理机构联系方式",
            "详情页链接",
        ],
        "B": [
            "项目编号",
            "中标人",
            "中标人联系人",
            "中标人联系方式",
            "中标金额(元)",
            "项目地点",
            "招标文件",
        ],
        "C": [
            "项目编号",
            "公告标题",
            "招标人/采购单位",
            "招标代理机构",
            "中标人",
            "项目地点",
            "发布日期",
            "中标金额(元)",
        ],
        "D": [
            "中标金额(元)",
            "发布日期",
            "招标人/采购单位",
            "中标人",
            "同比",
            "环比",
            "占比",
            "百分比",
        ],
        "E": [
            "项目编号",
            "中标金额(元)",
            "发布日期",
            "招标人/采购单位",
            "中标人",
            "统计",
            "计数",
            "求和",
            "平均",
        ],
        "F": ["开放问答", "政策解读", "流程说明", "名词定义", "业务规则", "政府采购"],
    }
    keywords = ["TYPE_{}".format(question_type)]
    keywords.extend(class_templates.get(question_type, class_templates["F"]))

    alias_to_field = {
        "采购人": "招标人/采购单位",
        "招标人": "招标人/采购单位",
        "采购单位": "招标人/采购单位",
        "招标代理": "招标代理机构",
        "代理机构": "招标代理机构",
        "代理联系人": "招标代理机构联系人",
        "代理联系方式": "招标代理机构联系方式",
        "中标金额": "中标金额(元)",
        "成交金额": "中标金额(元)",
        "金额": "中标金额(元)",
        "代理服务费": "代理服务收费金额(元)",
        "代理服务收费": "代理服务收费金额(元)",
        "代理费": "代理服务收费金额(元)",
        "公告链接": "详情页链接",
        "链接": "详情页链接",
    }
    for field_name in _zh_field_names_for_keywords():
        if field_name in question:
            keywords.append(field_name)
    for alias, field_name in alias_to_field.items():
        if alias in question:
            keywords.append(field_name)

    extra_hints = ["公开招标", "邀请招标", "竞争性谈判", "竞争性磋商", "询价", "单一来源"]
    for hint in extra_hints:
        if hint in question:
            keywords.append(hint)

    date_tokens = re.findall(r"\d{4}年(?:\d{1,2}月)?|\d{4}-\d{1,2}(?:-\d{1,2})?", question)
    keywords.extend(date_tokens)

    money_tokens = re.findall(r"\d+(?:\.\d+)?(?:亿|万|元|万元)", question)
    keywords.extend(money_tokens)

    regions = ["上海", "北京", "广州", "深圳", "杭州", "南京", "苏州", "天津", "重庆", "武汉", "成都", "西安"]
    for city in regions:
        if city in question:
            keywords.append(city)

    return _dedupe_keep_order(keywords)[:20]


def normalize_ptuning_classify_label(raw: str) -> str:
    """
    从分类 P-Tuning 的 `classify()` 文本输出中解析 A–F；无有效字母时视为 F（走开放检索）。
    """
    if not raw:
        return "F"
    s = str(raw).strip().upper()
    for ch in s:
        if ch in "ABCDE":
            return ch
        if ch == "F":
            return "F"
    return "F"


def _rule_only_class(question: str) -> Optional[str]:
    """
    仅规则先验：命中则返回 A（库表）或 F（开放问答）；无命中返回 None。
    单轮问答里若命中，则不再调用底座意图模型。
    """
    question = (question or "").strip()
    if not question:
        return None

    # 简短闲聊/问候：先走 F（检索+生成），避免误走 NL2SQL
    if len(question) <= 32 and not re.search(
        r"(项目编号|中标|招标|采购|公告|SQL|统计|金额|数量|几家|Top|排名|代理|标段|有多少|哪几)",
        question,
        re.I,
    ):
        if re.match(
            r"^(你好|您好|hi|hello|在吗|谢谢|辛苦了|拜拜|再见|好的|ok|OK|早上好|晚上好)[\s!！？。，,]*$",
            question,
            re.I,
        ):
            return "F"
        # 「聊天吗」「闲聊吗」等纯闲聊意图
        if re.match(
            r"^[^。]{0,24}(聊天|闲聊|唠嗑|聊聊)[吗呢嘛吧呀]?[？!！。…,\s]*$",
            question,
            re.I,
        ):
            return "F"

    # 明显开放问答
    if re.findall(r"(状况|简要介绍|简要分析|概述|具体描述|分析|影响)", question):
        return "F"
    if re.findall(r"(什么是|指什么|什么意思|定义|含义|为什么)", question):
        return "F"

    # 原 D/E/B/A/C 等均走结构化库表 → 规整为 A
    if re.findall(r"(增长率|比率|占比|同比|环比|百分比|公式|计算)", question):
        return "A"

    if re.findall(r"(多少个|几家|总计|合计|总额|平均|排名|最高|最低|前\d+|top\d+)", question.lower()):
        return "A"

    if re.findall(r"(中标|成交|中标人|中标金额|代理服务收费)", question):
        return "A"

    if re.findall(r"(项目编号|公告标题|发布日期|采购人|招标人|代理机构|联系人|联系电话|地址|详情链接|招标文件)", question):
        return "A"

    return None


def do_classification(classify_model: ChatGLM_Ptuning):
    """使用分类 P-Tuning `classify()` 写出 classify/*.csv，标签为 A–F（与 normalize_ptuning_classify_label 一致）。"""
    if not getattr(classify_model, "isClassify", False):
        raise RuntimeError("do_classification 需要 ChatGLM_Ptuning(PtuningType.Classify)")
    logger.info("Do classification (P-Tuning classify A–F)...")
    test_questions = load_test_questions()

    classify_dir = os.path.join(cfg.DATA_PATH, "classify")
    if not os.path.exists(classify_dir):
        os.mkdir(classify_dir)

    for question in test_questions:
        class_csv = os.path.join(classify_dir, "{}.csv".format(question["id"]))
        logger.opt(colors=True).info(
            "<blue>Start process question {} {}</>".format(question["id"], question["question"])
        )
        qtext = question["question"]
        try:
            raw_result = classify_model.classify(qtext) or ""
        except Exception as e:
            logger.warning("分类 P-Tuning 调用失败: {}", e)
            raw_result = ""
        result = normalize_ptuning_classify_label(raw_result)

        logger.info(result.replace('<', ''))

        with open(class_csv, 'w', encoding='utf-8') as f:
            save_result = copy.deepcopy(question)
            save_result['class'] = result

            json.dump(save_result, f, ensure_ascii=False)


def do_gen_keywords(model):
    logger.info("Do gen keywords...")
    test_questions = load_test_questions()

    keywords_dir = os.path.join(cfg.DATA_PATH, "keywords")
    if not os.path.exists(keywords_dir):
        os.mkdir(keywords_dir)

    use_kw_model = getattr(model, "isKeywords", False)
    if not use_kw_model:
        logger.warning("未加载 Keywords P-Tuning，批量关键词仅使用规则模板")

    for question in test_questions:
        keywords_csv = os.path.join(keywords_dir, '{}.csv'.format(question['id']))
        class_csv = os.path.join(cfg.DATA_PATH, 'classify', '{}.csv'.format(question['id']))
        question_type = 'F'
        if os.path.exists(class_csv):
            with open(class_csv, 'r', encoding='utf-8') as f:
                class_result = json.load(f)
                question_type = class_result.get('class', 'F')
        logger.opt(colors=True).info('<blue>Start process question {} {}</>'.format(question['id'], question['question']))

        rule_kw = _build_procurement_keywords(question["question"], question_type)
        model_kw = []
        if use_kw_model:
            raw_kw = model.keywords(question["question"])
            model_kw = _parse_model_keywords(raw_kw)
            logger.info("keywords模型原始输出: {}".format(raw_kw))

        # 有模型输出时：模型词优先，再与政府采购模板合并去重；否则仅用模板
        if model_kw:
            result = _dedupe_keep_order(model_kw + rule_kw)[:40]
        else:
            if use_kw_model:
                logger.warning("关键词模型无有效输出，回退规则模板: {}", question["question"][:80])
            result = rule_kw

        logger.info(result)

        with open(keywords_csv, 'w', encoding='utf-8') as f:
            save_result = copy.deepcopy(question)
            if len(result) == 0:
                logger.warning('问题{}的关键词为空'.format(question['question']))
                result = [question['question']]
            save_result['keywords'] = result

            json.dump(save_result, f, ensure_ascii=False)



def do_sql_generation(model: ChatGLM_Ptuning):
    logger.info("Do sql generation...")
    test_questions = load_test_questions()

    sql_dir = os.path.join(cfg.DATA_PATH, 'sql')
    if not os.path.exists(sql_dir):
        os.mkdir(sql_dir)

    for question in test_questions:

        sql_csv = os.path.join(sql_dir, '{}.csv'.format(question['id']))

        sql = None
        class_csv = os.path.join(cfg.DATA_PATH, 'classify', '{}.csv'.format(question['id']))
        if os.path.exists(class_csv):
            with open(class_csv, "r", encoding="utf-8") as f:
                class_result = json.load(f)
                question_type = class_result.get("class", "F")

            if question_type in SQL_TRIGGER_CLASSES:
                logger.opt(colors=True).info('<blue>Start process question {} {}</>'.format(question['id'], question['question'].replace('<', '')))
                sql = model.nl2sql(question['question'])
                logger.info(sql.replace('<>', ''))

        with open(sql_csv, 'w', encoding='utf-8') as f:
            save_result = copy.deepcopy(question)
            save_result['sql'] = sql
            json.dump(save_result, f, ensure_ascii=False)


# generate_answer / make_answer 已迁移至 answers 包（SQL vs 检索路由）
