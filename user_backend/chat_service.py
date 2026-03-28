from __future__ import annotations

from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy.orm import Session

from user_backend.database import SessionLocal
from user_backend.models import Conversation, Message, ModelCall


def persist_chat_turn(
    db: Session,
    *,
    user_id: int,
    conversation_id: int | None,
    user_text: str,
    assistant_text: str,
    raw_preview: str,
    qt: str,
    strategy: str,
    route: str,
    agent_mode: str | None,
    latency_ms: int,
    chart: dict[str, Any] | None,
    refs: list[dict[str, Any]] | None,
) -> dict[str, int]:
    """
    写入用户消息、助手消息与一次 ModelCall。成功返回 conversation_id / model_call_id / assistant_message_id。
    """
    if conversation_id is None:
        conv = Conversation(user_id=user_id, title=(user_text[:200] if user_text else "新对话"))
        db.add(conv)
        db.flush()
    else:
        conv = (
            db.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .first()
        )
        if conv is None:
            raise ValueError("conversation_not_found")

    um = Message(
        conversation_id=conv.id,
        role="user",
        content=user_text,
        meta_json={"agent_mode": agent_mode},
    )
    db.add(um)
    db.flush()

    am_meta: dict[str, Any] = {
        "strategy": strategy,
        "route": route,
        "normalized_class": qt,
        "raw_classification_preview": (raw_preview or "")[:2000],
        "has_chart": bool(chart),
        "refs_count": len(refs) if refs else 0,
    }
    am = Message(
        conversation_id=conv.id,
        role="assistant",
        content=assistant_text or "",
        meta_json=am_meta,
    )
    db.add(am)
    db.flush()

    mc = ModelCall(
        user_id=user_id,
        conversation_id=conv.id,
        user_message_id=um.id,
        assistant_message_id=am.id,
        agent_mode=agent_mode,
        normalized_class=qt,
        strategy=strategy,
        route=route,
        latency_ms=latency_ms,
        model_name="qa_pipeline",
        extra_json={"has_chart": bool(chart), "refs_count": len(refs) if refs else 0},
    )
    db.add(mc)
    conv.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(conv)
    return {
        "conversation_id": conv.id,
        "model_call_id": mc.id,
        "assistant_message_id": am.id,
    }


def persist_chat_turn_safe(
    *,
    user_id: int,
    conversation_id: int | None,
    user_text: str,
    assistant_text: str,
    raw_preview: str,
    qt: str,
    strategy: str,
    route: str,
    agent_mode: str | None,
    latency_ms: int,
    chart: dict[str, Any] | None,
    refs: list[dict[str, Any]] | None,
) -> dict[str, int] | None:
    """独立 Session，失败只打日志，不影响问答。"""
    db = SessionLocal()
    try:
        return persist_chat_turn(
            db,
            user_id=user_id,
            conversation_id=conversation_id,
            user_text=user_text,
            assistant_text=assistant_text,
            raw_preview=raw_preview,
            qt=qt,
            strategy=strategy,
            route=route,
            agent_mode=agent_mode,
            latency_ms=latency_ms,
            chart=chart,
            refs=refs,
        )
    except Exception as e:
        logger.warning("对话落库失败（不影响本次回答）: {}", e)
        return None
    finally:
        db.close()
