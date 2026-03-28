# -*- coding: utf-8 -*-
"""
NL2SQL 结果 → ECharts option：根据用户问题关键词选择柱状/折线/饼图等。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

_CHART_HINT = re.compile(
    r"(图表|柱状|柱图|条形|折线|饼图|饼状|环形|可视化|统计图|绘图|画图|画个图|数据分析|趋势图|占比|比例图)",
    re.I,
)


def question_wants_chart(question_text: str) -> bool:
    q = (question_text or "").strip()
    return bool(q and _CHART_HINT.search(q))


def _cell_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _pick_category_and_value(
    rows: List[Dict[str, Any]], cols: List[str]
) -> Tuple[List[str], List[float], str, str]:
    """第一列类别 + 第一个可解析为数值的列作为序列。"""
    if not rows or not cols:
        return [], [], "", ""

    def first_numeric_from(start: int) -> Optional[int]:
        for j in range(start, len(cols)):
            name = cols[j]
            vals = [_cell_float(r.get(name)) for r in rows]
            if any(x is not None for x in vals):
                return j
        return None

    if len(cols) == 1:
        c0 = cols[0]
        nums = [_cell_float(r.get(c0)) for r in rows]
        if all(x is not None for x in nums):
            cats = [str(i + 1) for i in range(len(rows))]
            return cats, [x for x in nums if x is not None], c0, c0
        return [], [], "", ""

    cat_idx = 0
    val_idx = first_numeric_from(1)
    if val_idx is None:
        val_idx = first_numeric_from(0)
        if val_idx is not None and val_idx == 0:
            nums = [_cell_float(r.get(cols[0])) for r in rows]
            if all(x is not None for x in nums):
                cats = [str(i + 1) for i in range(len(rows))]
                return cats, nums, cols[0], cols[0]
        return [], [], "", ""

    cname = cols[cat_idx]
    vname = cols[val_idx]
    cats: List[str] = []
    vals: List[float] = []
    for r in rows:
        raw = r.get(cname)
        nv = _cell_float(r.get(vname))
        if nv is None:
            continue
        cats.append(str(raw) if raw is not None else "")
        vals.append(nv)
    return cats, vals, cname, vname


def build_echarts_option(
    rows: List[Dict[str, Any]],
    cols: List[str],
    question_text: str,
) -> Optional[Dict[str, Any]]:
    """将二维表转为 ECharts option；无法绘图时返回 None。"""
    if not rows or not cols:
        return None

    q = (question_text or "").strip()
    title = q[:48] + ("…" if len(q) > 48 else "")

    want_pie = bool(re.search(r"(饼图|饼状|环形|占比|比例)", q))
    want_line = bool(re.search(r"(折线|趋势)", q))

    cats, vals, cname, vname = _pick_category_and_value(rows, cols)
    if not vals:
        return None

    if want_pie:
        pie_data = [{"name": cats[i] if i < len(cats) else str(i), "value": vals[i]} for i in range(len(vals))]
        return {
            "title": {
                "text": title,
                "left": "center",
                "textStyle": {"fontSize": 14, "color": "#e6edf3"},
            },
            "tooltip": {"trigger": "item"},
            "series": [
                {
                    "type": "pie",
                    "radius": "58%",
                    "data": pie_data,
                    "emphasis": {"itemStyle": {"shadowBlur": 10}},
                }
            ],
        }

    stype = "line" if want_line else "bar"
    axis_muted = "#8b9cb3"
    axis_line = "#3d4f66"
    return {
        "title": {
            "text": title,
            "left": "center",
            "textStyle": {"fontSize": 14, "color": "#e6edf3"},
        },
        "tooltip": {"trigger": "axis"},
        "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": cats,
            "axisLabel": {"rotate": 30 if len(cats) > 10 else 0, "color": axis_muted},
            "axisLine": {"lineStyle": {"color": axis_line}},
        },
        "yAxis": {
            "type": "value",
            "name": vname or "值",
            "nameTextStyle": {"color": axis_muted},
            "axisLabel": {"color": axis_muted},
            "splitLine": {"lineStyle": {"color": "#2d3a4d"}},
        },
        "series": [
            {
                "name": vname or "数值",
                "type": stype,
                "data": vals,
                "smooth": want_line,
            }
        ],
    }


if __name__ == "__main__":
    import json

    _rows = [
        {"region": "上海", "amt": 1200000},
        {"region": "北京", "amt": 980000},
    ]
    _cols = ["region", "amt"]
    _q = "用柱状图可视化各地区中标金额"
    print("question_wants_chart:", question_wants_chart(_q))
    _opt = build_echarts_option(_rows, _cols, _q)
    print(json.dumps(_opt, ensure_ascii=False, indent=2))
    _pie = build_echarts_option(_rows, _cols, "各地区占比饼图")
    print("--- pie ---")
    print(json.dumps(_pie, ensure_ascii=False, indent=2))
