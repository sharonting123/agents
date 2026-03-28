"""
F 类补充材料：DeepSeek Chat（OpenAI 兼容接口）。
无真实「联网检索」，由模型根据问题生成可并入 RAG 的要点式文本；与本地政策向量等合并后由主模型生成答案。

需：环境变量 DEEPSEEK_API_KEY；可选 DEEPSEEK_BASE_URL、DEEPSEEK_MODEL。
"""
from __future__ import annotations

from typing import Any, Dict

import httpx
from loguru import logger

from config import cfg


def deepseek_supplement_raw(question: str) -> str:
    """
    调用 DeepSeek chat/completions，返回 assistant 文本，供 split_for_rag_chunks 切分重排。
    """
    key = (getattr(cfg, "DEEPSEEK_API_KEY", "") or "").strip()
    if not key:
        raise ValueError("DEEPSEEK_API_KEY 未配置（请设置环境变量）")

    base = (getattr(cfg, "DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com").rstrip("/")
    model = getattr(cfg, "DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat"
    url = f"{base}/v1/chat/completions"

    system = (
        "你是政府采购与招投标政策法规领域的助手。请针对用户问题，输出简明、可检索的要点式摘要，"
        "便于作为 RAG 参考材料；可列出相关法规名称或关键词；不确定处请说明「需以最新官方文本为准」，不要编造。"
    )
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question.strip()},
        ],
        "temperature": 0.3,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            logger.error("DeepSeek HTTP {}: {}", r.status_code, r.text[:2000])
            r.raise_for_status()
        data = r.json()

    err = data.get("error")
    if isinstance(err, dict) and err.get("message"):
        raise RuntimeError(str(err.get("message", err)))

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek 响应无 choices")

    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        raise RuntimeError("DeepSeek 返回空 content")
    return content
