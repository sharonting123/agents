# -*- coding: utf-8 -*-
"""
政府采购投标助手：批量评测用问题列表（替代原 file.load_test_questions）。

默认读取 `data/test_questions.json`，格式为 JSON 数组：
[{"id": 1, "question": "..."}, ...]
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from config import cfg


def load_test_questions() -> List[Dict[str, Any]]:
    path = os.path.join(cfg.DATA_PATH, "test_questions.json")
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return list(data["questions"])
    return []
