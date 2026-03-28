"""
政策法规 FAISS 索引本地加载（与 policy.faiss + policy_chunks.json 配套）。
不依赖仓库外的 policy_vector_retrieval 模块；嵌入与构建索引时一致：text2vec-base-chinese（768 维）。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from config import cfg


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


class PolicyVectorIndex:
    """policy.faiss + policy_chunks.json，检索与 build 脚本对齐（IndexFlatIP + 归一化向量）。"""

    def __init__(self, index_dir: str):
        self.index_dir = os.path.abspath(index_dir)
        self._index = None
        self._chunks: List[Dict[str, Any]] = []
        self._encoder = None

    def load(self) -> None:
        import faiss  # noqa: PLC0415

        chunks_path = os.path.join(self.index_dir, "policy_chunks.json")
        faiss_path = os.path.join(self.index_dir, "policy.faiss")
        with open(chunks_path, encoding="utf-8") as f:
            self._chunks = json.load(f)
        self._index = faiss.read_index(faiss_path)
        if len(self._chunks) != int(self._index.ntotal):
            raise ValueError(
                f"policy_chunks.json 条数 ({len(self._chunks)}) 与 FAISS ntotal ({self._index.ntotal}) 不一致"
            )

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        import torch  # noqa: PLC0415

        path = getattr(cfg, "TEXT2VEC_MODEL_DIR", None)
        if not path or not os.path.isdir(path):
            raise RuntimeError(f"TEXT2VEC_MODEL_DIR 无效: {path}")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            from text2vec import SentenceModel  # noqa: PLC0415

            self._encoder = SentenceModel(model_name_or_path=path, device=device)
        except ImportError:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            st_device = "cuda" if torch.cuda.is_available() else "cpu"
            self._encoder = SentenceTransformer(path, device=st_device)
        return self._encoder

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip() or self._index is None or not self._chunks:
            return []
        enc = self._get_encoder()
        q_emb = np.asarray(enc.encode([query.strip()]), dtype=np.float32)
        q_emb = _l2_normalize(q_emb)
        k = min(top_k, int(self._index.ntotal))
        scores, indices = self._index.search(q_emb, k)
        out: List[Dict[str, Any]] = []
        for rank in range(k):
            idx = int(indices[0][rank])
            if idx < 0:
                continue
            row = dict(self._chunks[idx])
            row["_score"] = float(scores[0][rank])
            out.append(row)
        return out


def build_rag_evidence_blocks(question: str, hits: List[Dict[str, Any]]) -> List[str]:
    """
    条 1 优先；若检索第 5 条与「联合体」相关且与第 1 条不同，追加联合体补充块。
    """
    if not hits:
        return []
    q = question.strip()

    def _block(h: Dict[str, Any], tag: str) -> str:
        title = h.get("title") or ""
        url = h.get("detail_url") or ""
        body = (h.get("text") or "").strip()
        return f"【{tag}】\n《{title}》\n参考链接：{url}\n\n{body}"

    blocks = [_block(hits[0], "政策法规·优先依据")]
    if len(hits) >= 5:
        fifth = hits[4]
        same_chunk = hits[0].get("policy_id") == fifth.get("policy_id") and hits[0].get(
            "chunk_index"
        ) == fifth.get("chunk_index")
        if not same_chunk and (
            "联合体" in q or "联合体" in (fifth.get("text") or "")
        ):
            blocks.append(_block(fifth, "联合体相关补充"))
    return blocks
