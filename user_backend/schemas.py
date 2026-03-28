from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=6, max_length=12)
    email: str | None = None
    full_name: str | None = None

    @field_validator("username", mode="before")
    @classmethod
    def username_strip(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


class UserOut(BaseModel):
    id: int
    username: str
    email: str | None
    full_name: str | None
    is_active: bool
    is_superuser: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginBody(BaseModel):
    username: str
    password: str = Field(..., min_length=6, max_length=12)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    meta_json: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: int
    user_id: int
    title: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut] = Field(default_factory=list)


class ModelCallOut(BaseModel):
    id: int
    user_id: int | None
    conversation_id: int | None
    agent_mode: str | None
    normalized_class: str | None
    strategy: str | None
    route: str | None
    latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    model_name: str | None
    error: str | None
    extra_json: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvaluationCreate(BaseModel):
    assistant_message_id: int
    rating: int | None = Field(None, ge=1, le=5)
    feedback_text: str | None = None
    metrics_json: dict[str, Any] | None = None
    conversation_id: int | None = None

    @field_validator("rating")
    @classmethod
    def rating_ok(cls, v: int | None) -> int | None:
        if v is not None and not (1 <= v <= 5):
            raise ValueError("rating must be 1–5")
        return v


class EvaluationOut(BaseModel):
    id: int
    user_id: int
    conversation_id: int | None
    assistant_message_id: int
    rating: int | None
    feedback_text: str | None
    metrics_json: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalyticsSummary(BaseModel):
    total_model_calls: int
    total_conversations: int
    total_evaluations: int
    avg_latency_ms: float | None
    calls_last_24h: int
