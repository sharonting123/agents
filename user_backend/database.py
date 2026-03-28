from __future__ import annotations

import os
import re
from urllib.parse import quote_plus

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import cfg
from user_backend.models import Base


def _mysql_user_app_database_name() -> str:
    mc = cfg.MYSQL_CONFIG
    return getattr(cfg, "USER_APP_MYSQL_DATABASE", None) or mc["database"]


def ensure_mysql_database_exists() -> None:
    """
    若库不存在则 CREATE DATABASE（需账号具备建库权限）。
    在创建 SQLAlchemy engine 之前调用，避免连向不存在的库。
    """
    if not getattr(cfg, "USER_APP_USE_MYSQL", True):
        return
    import pymysql

    mc = cfg.MYSQL_CONFIG
    db_name = _mysql_user_app_database_name()
    if not re.match(r"^[A-Za-z0-9_]+$", db_name):
        raise ValueError(f"非法数据库名: {db_name!r}")
    conn = pymysql.connect(
        host=mc["host"],
        port=int(mc["port"]),
        user=mc["user"],
        password=mc["password"],
        charset=mc.get("charset", "utf8mb4"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.commit()
    finally:
        conn.close()


def _build_engine():
    use_mysql = getattr(cfg, "USER_APP_USE_MYSQL", True)
    if use_mysql:
        ensure_mysql_database_exists()
        mc = cfg.MYSQL_CONFIG
        db_name = _mysql_user_app_database_name()
        user = quote_plus(str(mc["user"]))
        password = quote_plus(str(mc["password"]))
        host = mc["host"]
        port = int(mc["port"])
        url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4"
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )

    _db_path = getattr(cfg, "USER_APP_DB_PATH", "")
    if _db_path:
        os.makedirs(os.path.dirname(os.path.abspath(_db_path)), exist_ok=True)
    return create_engine(
        f"sqlite:///{_db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )


engine = _build_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """在目标库中创建 qa_users / qa_conversations / qa_messages / qa_model_calls / qa_evaluations。"""
    Base.metadata.create_all(bind=engine)


def user_app_backend_label() -> str:
    """供启动日志展示。"""
    if getattr(cfg, "USER_APP_USE_MYSQL", True):
        mc = cfg.MYSQL_CONFIG
        db = _mysql_user_app_database_name()
        return f"MySQL {mc['host']}:{mc['port']}/{db} (qa_* 表)"
    return f"SQLite {getattr(cfg, 'USER_APP_DB_PATH', '')}"
