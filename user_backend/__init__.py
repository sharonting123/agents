"""用户管理、对话历史、模型调用与评估指标（默认 MySQL 与 MYSQL_CONFIG 同库 qa_* 表；可改 USER_APP_USE_MYSQL=0 用 SQLite）。"""

from user_backend.database import SessionLocal, engine, init_db

__all__ = ["SessionLocal", "engine", "init_db"]
