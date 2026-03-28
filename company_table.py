# -*- coding: utf-8 -*-
"""
NL2SQL 执行与纠错：
- load_company_table()：纠错用 **cfg.SQL_EN_TO_ZH_COLUMNS 的中文注释（值）**，与 normalize 替换一致。
- get_sql_search_cursor()：执行 SQL 仍连 MySQL（cfg.MYSQL_SQL_TABLE），或回退 CSV→SQLite。
"""
from __future__ import annotations

import csv
import os
import sqlite3
from typing import Any, List

from loguru import logger

from config import cfg


def _zh_columns_from_cfg() -> List[str]:
    """英文列名对应的中文注释（用于字段纠错候选）。"""
    cmap = getattr(cfg, "SQL_EN_TO_ZH_COLUMNS", None) or {}
    return [v for v in cmap.values() if v]


class _TableMeta:
    """兼容 load_company_table().columns 的轻量对象。"""

    __slots__ = ("columns",)

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns


def load_company_table() -> _TableMeta:
    """
    返回带 .columns 的对象，供 NL2SQL **字段纠错**。
    始终使用中文注释（cfg.SQL_EN_TO_ZH_COLUMNS 的值），与库表英文键成对配置。
    """
    return _TableMeta(_zh_columns_from_cfg())


def get_sql_search_cursor() -> Any:
    """返回支持 execute(sql) / fetchall() 的游标（执行层，连真实库）。"""
    if getattr(cfg, "USE_MYSQL_FOR_SQL", False):
        try:
            import pymysql

            conn = pymysql.connect(**cfg.MYSQL_CONFIG)
            return conn.cursor()
        except Exception as e:
            logger.error("MySQL 连接失败: {}", e)

    csv_path = os.path.join(cfg.DATA_PATH, "CompanyTable.csv")
    if os.path.isfile(csv_path):
        try:
            conn = sqlite3.connect(":memory:")
            cur = conn.cursor()
            with open(csv_path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                if not header:
                    raise ValueError("CSV 无表头")
                placeholders = ",".join(["?"] * len(header))
                cols_sql = ",".join(f'"{c}" TEXT' for c in header)
                cur.execute(f"CREATE TABLE company_table ({cols_sql})")
                for row in reader:
                    if len(row) < len(header):
                        row = row + [""] * (len(header) - len(row))
                    cur.execute(
                        f"INSERT INTO company_table VALUES ({placeholders})",
                        row[: len(header)],
                    )
                conn.commit()
            return conn.cursor()
        except Exception as e:
            logger.error("SQLite 加载 CompanyTable.csv 失败: {}", e)

    raise RuntimeError(
        "无法创建 SQL 游标：请配置可用的 MySQL（USE_MYSQL_FOR_SQL=1 且 MYSQL_* 正确），"
        "或在 data/CompanyTable.csv 提供本地表数据。"
    )
