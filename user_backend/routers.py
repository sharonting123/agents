from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from user_backend.deps import get_current_user, get_db
from user_backend.models import Conversation, Evaluation, Message, ModelCall, User
from user_backend.schemas import (
    AnalyticsSummary,
    ConversationDetailOut,
    ConversationOut,
    EvaluationCreate,
    EvaluationOut,
    LoginBody,
    MessageOut,
    ModelCallOut,
    Token,
    UserCreate,
    UserOut,
)
from user_backend.security import create_access_token, get_password_hash, verify_password

router = APIRouter(prefix="/api/v1", tags=["用户与审计"])


@router.post("/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate, db: Session = Depends(get_db)) -> User:
    if db.query(User).filter(User.username == body.username.strip()).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    if body.email and db.query(User).filter(User.email == body.email.strip()).first():
        raise HTTPException(status_code=400, detail="邮箱已被注册")
    u = User(
        username=body.username.strip(),
        email=body.email.strip() if body.email else None,
        full_name=body.full_name,
        hashed_password=get_password_hash(body.password),
    )
    db.add(u)
    try:
        db.commit()
        db.refresh(u)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="用户名已存在")
    return u


@router.post("/auth/login", response_model=Token)
def login(body: LoginBody, db: Session = Depends(get_db)) -> Token:
    u = db.query(User).filter(User.username == body.username.strip()).first()
    if u is None or not verify_password(body.password, u.hashed_password):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    if not u.is_active:
        raise HTTPException(status_code=403, detail="账号已禁用")
    token = create_access_token(data={"sub": str(u.id), "username": u.username})
    return Token(access_token=token)


@router.get("/users/me", response_model=UserOut)
def read_me(user: User = Depends(get_current_user)) -> User:
    return user


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Conversation]:
    q = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(q.all())


@router.get("/conversations/{cid}", response_model=ConversationDetailOut)
def get_conversation(
    cid: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ConversationDetailOut:
    c = (
        db.query(Conversation)
        .filter(Conversation.id == cid, Conversation.user_id == user.id)
        .first()
    )
    if c is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    msgs = (
        db.query(Message)
        .filter(Message.conversation_id == cid)
        .order_by(Message.id.asc())
        .all()
    )
    return ConversationDetailOut(
        id=c.id,
        user_id=c.user_id,
        title=c.title,
        created_at=c.created_at,
        updated_at=c.updated_at,
        messages=[MessageOut.model_validate(m) for m in msgs],
    )


@router.delete("/conversations/{cid}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    cid: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    c = (
        db.query(Conversation)
        .filter(Conversation.id == cid, Conversation.user_id == user.id)
        .first()
    )
    if c is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.delete(c)
    db.commit()


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalyticsSummary:
    if user.is_superuser:
        total_calls = db.query(ModelCall).count()
        total_conv = db.query(Conversation).count()
        total_ev = db.query(Evaluation).count()
        avg_lat = db.query(func.avg(ModelCall.latency_ms)).scalar()
        since = datetime.utcnow() - timedelta(hours=24)
        calls_24h = db.query(ModelCall).filter(ModelCall.created_at >= since).count()
    else:
        uid = user.id
        total_calls = db.query(ModelCall).filter(ModelCall.user_id == uid).count()
        total_conv = db.query(Conversation).filter(Conversation.user_id == uid).count()
        total_ev = db.query(Evaluation).filter(Evaluation.user_id == uid).count()
        avg_lat = (
            db.query(func.avg(ModelCall.latency_ms)).filter(ModelCall.user_id == uid).scalar()
        )
        since = datetime.utcnow() - timedelta(hours=24)
        calls_24h = (
            db.query(ModelCall)
            .filter(ModelCall.user_id == uid, ModelCall.created_at >= since)
            .count()
        )

    avg_out = float(avg_lat) if avg_lat is not None else None
    return AnalyticsSummary(
        total_model_calls=total_calls,
        total_conversations=total_conv,
        total_evaluations=total_ev,
        avg_latency_ms=avg_out,
        calls_last_24h=calls_24h,
    )


@router.get("/analytics/model-calls", response_model=list[ModelCallOut])
def list_model_calls(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ModelCall]:
    q = db.query(ModelCall).order_by(ModelCall.id.desc())
    if not user.is_superuser:
        q = q.filter(ModelCall.user_id == user.id)
    return list(q.offset(skip).limit(limit).all())


@router.post("/evaluations", response_model=EvaluationOut, status_code=status.HTTP_201_CREATED)
def create_evaluation(
    body: EvaluationCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Evaluation:
    am = db.query(Message).filter(Message.id == body.assistant_message_id).first()
    if am is None or am.role != "assistant":
        raise HTTPException(status_code=400, detail="无效的 assistant_message_id")
    conv = db.query(Conversation).filter(Conversation.id == am.conversation_id).first()
    if conv is None or conv.user_id != user.id:
        raise HTTPException(status_code=403, detail="无权评价该条消息")
    ev = Evaluation(
        user_id=user.id,
        conversation_id=body.conversation_id or conv.id,
        assistant_message_id=body.assistant_message_id,
        rating=body.rating,
        feedback_text=body.feedback_text,
        metrics_json=body.metrics_json,
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    return ev


@router.get("/evaluations", response_model=list[EvaluationOut])
def list_evaluations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[Evaluation]:
    q = db.query(Evaluation).filter(Evaluation.user_id == user.id).order_by(Evaluation.id.desc())
    return list(q.offset(skip).limit(limit).all())


@router.post("/admin/bootstrap-superuser", response_model=UserOut)
def bootstrap_superuser(
    body: UserCreate,
    db: Session = Depends(get_db),
) -> User:
    """
    若系统中尚无任何用户，则创建首个用户并设为超级管理员；否则 403。
    便于首次部署，无需手改数据库。
    """
    if db.query(User).count() > 0:
        raise HTTPException(status_code=403, detail="已有用户，请使用注册接口或数据库授权")
    u = User(
        username=body.username.strip(),
        email=body.email.strip() if body.email else None,
        full_name=body.full_name,
        hashed_password=get_password_hash(body.password),
        is_superuser=True,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u
