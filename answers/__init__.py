"""
answers 包：按问题类别路由答题策略。
- A–E：结构化 SQL（shggzy_bid_result）
- F：政策法规 FAISS 向量 + 可选 DeepSeek 补充 + ChatGLM；可选 FAISS+LCEL（cfg.USE_LANGCHAIN_LCEL_RAG）或 stuff 链（cfg.USE_LANGCHAIN_RAG）

注意：勿在包初始化时 import orchestrator（会拉取 re_util 等），否则
`from answers import retrieval` 也会失败。generate_answer / make_answer 惰性加载。
"""
from __future__ import annotations

from typing import Any

__all__ = ["generate_answer", "make_answer"]


def __getattr__(name: str) -> Any:
    if name == "generate_answer":
        from answers.orchestrator import generate_answer

        return generate_answer
    if name == "make_answer":
        from answers.orchestrator import make_answer

        return make_answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
