# -*- coding: utf-8 -*-
"""
答案文本后处理：汇总/落盘前做简单清洗与规整。
（原独立工具文件；若项目中有更复杂的规则可在此扩展。）
"""
from __future__ import annotations

import re
from typing import Any


def rewrite_answer(text: Any) -> str:
    """将模型或 SQL 返回的 answer 规整为适合写入 JSON 的字符串。"""
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    # 合并连续空白，避免多余换行影响一行一条 JSON
    s = re.sub(r"\s+", " ", s)
    return s
