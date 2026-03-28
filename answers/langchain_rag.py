# -*- coding: utf-8 -*-
"""
F 类：基于 LangChain 的 RAG（stuff combine_documents + 自定义 Prompt + ChatGLM）。

依赖项目中的 langchain==0.0.x。
若链构建失败，retrieval.answer_via_retrieval 会回退为「拼 prompt + model()」。
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from loguru import logger

# LangChain Prompt：与 retrieval.build_rag_prompt 语义一致
RAG_TEMPLATE = (
    "你是政府采购与招投标领域的助手。请仅根据下面「参考材料」回答用户问题；"
    "材料中没有的信息请明确说明无法从材料中得出，不要编造。\n\n"
    "【参考材料】\n"
    "{context}\n\n"
    "【用户问题】\n"
    "{question}"
)


def _make_llm_adapter(chat_model: Any):
    """将 ChatGLM_Ptuning（Nothing）包装为 LangChain LLM（闭包，避免 Pydantic 字段问题）。"""
    try:
        from langchain.llms.base import LLM
    except ImportError as e:
        raise ImportError("需要安装 langchain") from e

    class ChatGLM_PtuningLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "chatglm_ptuning"

        def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
        ) -> str:
            return chat_model(prompt)

    return ChatGLM_PtuningLLM()


def answer_via_langchain_stuff_chain(
    question: str,
    evidence_chunks: List[str],
    strategy_tag: str,
    chat_model: Any,
) -> Tuple[str, str]:
    """
    使用 LangChain `load_qa_chain(..., chain_type="stuff")` 基于检索片段生成答案。

    Returns:
        (answer_text, strategy_tag) — strategy_tag 与入参一致（retrieval_rag_web | retrieval_rag）
    """
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document

    llm = _make_llm_adapter(chat_model)
    prompt = PromptTemplate(template=RAG_TEMPLATE, input_variables=["context", "question"])

    try:
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    except TypeError:
        chain = load_qa_chain(
            llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
        )

    docs = [Document(page_content=c, metadata={"source": "rag"}) for c in evidence_chunks if c.strip()]
    if not docs:
        return chat_model(question.strip()), strategy_tag

    try:
        out = chain.run(input_documents=docs, question=question.strip())
    except Exception as e:
        logger.warning("LangChain stuff 链 run 失败，尝试 __call__: {}", e)
        out = chain({"input_documents": docs, "question": question.strip()})
        if isinstance(out, dict):
            out = out.get("output_text") or out.get("answer") or str(out)
    text = (out or "").strip() if isinstance(out, str) else str(out).strip()
    return text if text else chat_model(question.strip()), strategy_tag
