from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt
from passlib.context import CryptContext

from config import cfg

# 仅用于校验「旧库」里已存的纯 bcrypt(明文) 哈希（明文须 ≤72 字节）。
_legacy_bcrypt = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """
    先对 UTF-8 口令做 SHA256，再对 64 字符 hex 做 bcrypt，避免 bcrypt 对「原始口令」72 字节上限
    （长中文口令会触发 password cannot be longer than 72 bytes）。
    """
    digest = hashlib.sha256(password.encode("utf-8")).hexdigest().encode("ascii")
    return bcrypt.hashpw(digest, bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """新哈希：bcrypt(sha256hex)；旧哈希：bcrypt(明文) 且明文 ≤72 字节。"""
    if not plain or not hashed:
        return False
    try:
        hb = hashed.encode("ascii")
        digest = hashlib.sha256(plain.encode("utf-8")).hexdigest().encode("ascii")
        try:
            if bcrypt.checkpw(digest, hb):
                return True
        except ValueError:
            pass
        if len(plain.encode("utf-8")) <= 72:
            try:
                return _legacy_bcrypt.verify(plain, hashed)
            except Exception:
                return False
        return False
    except Exception:
        return False


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=int(getattr(cfg, "ACCESS_TOKEN_EXPIRE_MINUTES", 10080)))
    )
    to_encode.update({"exp": expire})
    return jwt.encode(
        to_encode,
        getattr(cfg, "JWT_SECRET_KEY", "change-me"),
        algorithm=getattr(cfg, "JWT_ALGORITHM", "HS256"),
    )


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(
            token,
            getattr(cfg, "JWT_SECRET_KEY", "change-me"),
            algorithms=[getattr(cfg, "JWT_ALGORITHM", "HS256")],
        )
    except JWTError:
        return None
