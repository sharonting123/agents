# -*- coding: utf-8 -*-
"""
F 类：参考「RunnableParallel + FAISS retriever + Prompt + LLM」的 RAG 管线
（与 f:\\Machinelearning\\Deeplearning\\RAG\\rag.py 中 llm_chain / chunk2vector 思路一致）。

依赖（需单独安装，与旧版 langchain==0.0.247 可并存，本模块只 import 新栈）:
  pip install langchain-core langchain-community faiss-cpu sentence-transformers

嵌入默认使用 cfg.TEXT2VEC_MODEL_DIR（sentence-transformers 兼容目录）。
若不可用，retrieval 会回退 answers.langchain_rag 的 stuff 链或拼 prompt。
"""
from __future__ import annotations

import os
from typing import Any, List, Tuple

from loguru import logger

from config import cfg


def _prompt_to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    t = getattr(x, "text", None)
    if t is not None:
        return str(t)
    return str(x)


def _build_embeddings():
    """HuggingFace 句向量，路径与 text2vec 本地目录一致（需 sentence-transformers）。"""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    emb_path = getattr(cfg, "TEXT2VEC_MODEL_DIR", None)
    if not emb_path or not os.path.isdir(emb_path):
        raise FileNotFoundError(f"TEXT2VEC_MODEL_DIR 无效: {emb_path}")

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return HuggingFaceEmbeddings(
        model_name=emb_path,
        model_kwargs={"device": device},
    )


def answer_via_lcel_faiss_rag(
    question: str,
    evidence_chunks: List[str],
    strategy_tag: str,
    chat_model: Any,
    top_k: int = 5,
) -> Tuple[str, str]:
    """
    将已排序的证据片段写入内存 FAISS，按参考脚本方式构建链并 invoke(question)。

    Returns:
        (answer_text, strategy_tag)
    """
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

    docs = [
        Document(page_content=c.strip(), metadata={"source": "evidence"})
        for c in evidence_chunks
        if c and str(c).strip()
    ]
    if not docs:
        return chat_model(question.strip()), strategy_tag

    embeddings = _build_embeddings()
    vector = FAISS.from_documents(documents=docs, embedding=embeddings)
    k = max(1, min(top_k, len(docs)))
    retriever = vector.as_retriever(search_kwargs={"k": k})

    def format_docs(docs_in: List[Document]) -> str:
        return "\n\n".join(d.page_content for d in docs_in)

    # 与参考 rag.py 的 RunnableParallel | prompt | llm 一致；此处用 PromptTemplate 便于接 ChatGLM 字符串推理
    template = (
        "你是政府采购与招投标领域的助手。请仅根据下面「参考材料」回答用户问题；"
        "材料中没有的信息请明确说明无法从材料中得出，不要编造。\n\n"
        "【参考材料】\n{context}\n\n【用户问题】\n{question}"
    )
    prompt = PromptTemplate.from_template(template)

    def glm_generate(prompt_value: Any) -> str:
        s = _prompt_to_str(prompt_value)
        return chat_model(s)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | RunnableLambda(glm_generate)
    )

    q = (question or "").strip() or "hello"
    out = chain.invoke(q)
    text = (out or "").strip() if isinstance(out, str) else str(out).strip()
    return (text if text else chat_model(q), strategy_tag)


def lcel_rag_available() -> bool:
    try:
        import langchain_core  # noqa: F401
        from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
        from langchain_community.vectorstores import FAISS  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
