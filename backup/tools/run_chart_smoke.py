# -*- coding: utf-8 -*-
"""不经过 answers 包初始化，直接加载 answers/sql_chart.py 做自检。"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PATH = _ROOT / "answers" / "sql_chart.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("sql_chart_mod", _PATH)
    if spec is None or spec.loader is None:
        print("无法加载", _PATH, file=sys.stderr)
        sys.exit(1)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rows = [
        {"region": "上海", "amt": 1200000},
        {"region": "北京", "amt": 980000},
    ]
    cols = ["region", "amt"]
    q = "用柱状图可视化各地区中标金额"
    print("question_wants_chart:", mod.question_wants_chart(q))
    opt = mod.build_echarts_option(rows, cols, q)
    print(json.dumps(opt, ensure_ascii=False, indent=2))
    print("--- pie ---")
    pie = mod.build_echarts_option(rows, cols, "各地区占比饼图")
    print(json.dumps(pie, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
