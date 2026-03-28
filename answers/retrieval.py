"""
F 类：信息检索 + 生成。
1) 政策法规：`cfg.POLICY_VECTOR_INDEX_DIR` 下 FAISS（`policy_vector_retrieval` 从库表 body_text 构建），
   由 `_get_policy_rag_evidence_blocks` 注入，置于参考材料最前。
2) 补充：`get_ranked_evidence_chunks` 在配置 DeepSeek 时拉模型摘要，经 text2vec/jieba 重排后与政策合并。


生成步优先级：
1) `cfg.USE_LANGCHAIN_LCEL_RAG`：`answers.rag_lcel`（FAISS + RunnableParallel）
2) `cfg.USE_LANGCHAIN_RAG`：`answers.langchain_rag`（stuff QA 链）
3) 否则「拼 prompt + model()」
"""
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from config import cfg

try:
    import jieba
except ImportError:
    jieba = None

try:
    from text2vec import SentenceModel, semantic_search
except ImportError:
    SentenceModel = None  # type: ignore
    semantic_search = None  # type: ignore

_sentence_model = None
_policy_vector_index = None
_policy_vector_index_dir: Optional[str] = None
_policy_deps_warned = False


def _policy_vector_deps_installed() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    if SentenceModel is not None:
        return True
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _get_policy_rag_evidence_blocks(question: str) -> List[str]:
    """
    从 docs/policy_vector_retrieval 的 FAISS 索引取 Top5，按 build_rag_evidence_blocks 规则
    生成「条1 优先 + 必要时条5 联合体」文本块，置于 RAG 材料最前。
    """
    global _policy_deps_warned, _policy_vector_index, _policy_vector_index_dir
    if not getattr(cfg, "POLICY_VECTOR_RAG_ENABLED", True):
        return []
    idx_dir = getattr(cfg, "POLICY_VECTOR_INDEX_DIR", "") or ""
    idx_dir = os.path.abspath(idx_dir)
    if not idx_dir or not os.path.isfile(os.path.join(idx_dir, "policy.faiss")):
        return []
    if not _policy_vector_deps_installed():
        if not _policy_deps_warned:
            _policy_deps_warned = True
            logger.warning(
                "政策法规向量 RAG 已跳过：需要 faiss-cpu，以及 text2vec 或 sentence-transformers（与 TEXT2VEC_MODEL_DIR 一致）。"
                "请执行: pip install faiss-cpu text2vec"
            )
        return []
    try:
        try:
            from policy_vector_retrieval import PolicyVectorIndex, build_rag_evidence_blocks
        except ImportError:
            from answers.policy_faiss_local import PolicyVectorIndex, build_rag_evidence_blocks

        if _policy_vector_index is None or _policy_vector_index_dir != idx_dir:
            _policy_vector_index = PolicyVectorIndex(index_dir=idx_dir)
            _policy_vector_index.load()
            _policy_vector_index_dir = idx_dir
        hits = _policy_vector_index.search(question.strip(), top_k=5)
        return build_rag_evidence_blocks(question, hits)
    except Exception as e:
        logger.warning("政策法规向量 RAG 不可用（索引路径、模型目录或依赖）: {}", e)
        return []


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model
    if SentenceModel is None:
        return None
    path = getattr(cfg, "TEXT2VEC_MODEL_DIR", None)
    if not path or not os.path.isdir(path):
        logger.warning("TEXT2VEC_MODEL_DIR 无效或未找到: {}", path)
        return None
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _sentence_model = SentenceModel(model_name_or_path=path, device=device)
    return _sentence_model


def _score_chunks_semantic(question: str, chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    sm = _get_sentence_model()
    if sm is None or semantic_search is None or not chunks:
        return []
    q_emb = sm.encode([question])
    c_emb = sm.encode(chunks)
    hits = semantic_search(q_emb, c_emb, top_k=min(top_k, len(chunks)))
    ranked: List[Tuple[str, float]] = []
    for h in hits[0]:
        cid = h.get("corpus_id", 0)
        score = float(h.get("score", 0.0))
        if 0 <= cid < len(chunks):
            ranked.append((chunks[cid], score))
    return ranked


def _score_chunks_lexical(question: str, chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    if not chunks:
        return []
    if jieba is None:
        q_tokens = set(question)
        scored = []
        for c in chunks:
            overlap = sum(1 for t in q_tokens if t in c)
            scored.append((c, float(overlap)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    qset = set(jieba.cut(question))
    qset.discard(" ")
    scored = []
    for c in chunks:
        cset = set(jieba.cut(c))
        inter = len(qset & cset)
        union = len(qset | cset) or 1
        scored.append((c, inter / union))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _rank_chunks(question: str, chunks: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
    sem = _score_chunks_semantic(question, chunks, top_k=top_k)
    if sem:
        return sem
    return _score_chunks_lexical(question, chunks, top_k=top_k)


_RE_MD_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
_RE_URL = re.compile(r"https?://[^\s<>\s\"\'\[\]（）]+")


def _trim_url_tail(url: str) -> str:
    """去掉句末误捕获的中文标点、右括号等。"""
    return url.rstrip(").,;，。；、）】』'\"］＞")


def reference_entries_from_evidence(evidence: List[str]) -> List[Dict[str, Any]]:
    """
    从检索材料中提取 Markdown 链接与裸 URL，供政策咨询类回答展示「参考来源」。
    """
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for block in evidence or []:
        if not block:
            continue
        text = str(block)
        for m in _RE_MD_LINK.finditer(text):
            title = (m.group(1) or "").strip()
            url = _trim_url_tail(m.group(2).strip())
            if url and url not in seen:
                seen.add(url)
                out.append({"title": title or url, "url": url})
        # 去掉已匹配的 Markdown 链接，避免裸 URL 正则重复或吞入后续中文
        text_wo_md = _RE_MD_LINK.sub(" ", text)
        for m in _RE_URL.finditer(text_wo_md):
            url = _trim_url_tail(m.group(0).strip())
            if not url or url in seen:
                continue
            if not (url.startswith("http://") or url.startswith("https://")):
                continue
            seen.add(url)
            host = url.split("//", 1)[-1] if "//" in url else url
            title = host[:72] + ("…" if len(host) > 72 else "")
            out.append({"title": title, "url": url})
        if len(out) >= 24:
            break
    return out[:20]


def build_rag_prompt(question: str, evidence_blocks: List[str]) -> str:
    evidence = "\n\n---\n\n".join(evidence_blocks)
    return (
        "你是政府采购与招投标领域的助手。请仅根据下面「参考材料」回答用户问题；"
        "材料中没有的信息请明确说明无法从材料中得出，不要编造。\n"
        "若材料中标注了「政策法规·优先依据」，请优先依据该段完整含义作答，并可用自然语言归纳表述；"
        "若另有「联合体相关补充」且与用户问题相关，请一并纳入，避免遗漏联合体资格等要点。\n"
        "若参考材料中含 http/https 链接或 Markdown 链接，回答中可简要复述要点，系统会在回答下方单独展示「参考来源」链接。\n\n"
        "【参考材料】\n"
        f"{evidence}\n\n"
        "【用户问题】\n"
        f"{question.strip()}"
    )


def split_for_rag_chunks(text: str, max_chunk: int = 800) -> List[str]:
    """将长文本切成多段，便于句向量重排。"""
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if len(parts) >= 2:
        return [p[:4000] for p in parts]
    if len(text) <= max_chunk:
        return [text[:4000]]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + max_chunk])
        i += max_chunk
    return chunks[:32]


def get_ranked_evidence_chunks(
    question: str,
    top_k: int = 5,
) -> Tuple[List[str], Optional[str]]:
    """
    DeepSeek 补充（不使用 KNOWLEDGE_DIR）。返回片段经 text2vec/jieba 重排。
    需 DEEPSEEK_API_KEY + F_USE_DEEPSEEK_SUPPLEMENT。

    Returns:
        (evidence_blocks, strategy_tag)。无可用片段时 evidence 为空、strategy_tag 为 None。
        有片段时 strategy_tag 为 retrieval_rag_web（沿用原标签便于统计）。
    """
    ori = question.strip()

    use_deepseek = getattr(cfg, "F_USE_DEEPSEEK_SUPPLEMENT", True) and bool(
        getattr(cfg, "DEEPSEEK_API_KEY", "") or ""
    )
    if use_deepseek:
        try:
            from answers.deepseek_supplement import deepseek_supplement_raw

            raw = deepseek_supplement_raw(ori)
            chunks = split_for_rag_chunks(raw)
            if not chunks:
                chunks = [raw[:4000]]
            ranked = _rank_chunks(ori, chunks, top_k=top_k)
            if ranked:
                evidence = [r[0] for r in ranked[:top_k]]
            else:
                evidence = chunks[:top_k]
            if evidence:
                return evidence, "retrieval_rag_web"
        except Exception as e:
            logger.warning("F: DeepSeek 补充失败: {}", e)

    return [], None


def answer_via_retrieval(
    question: str,
    model,
    top_k: int = 5,
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Returns (answer_text, strategy_tag, references)。
    references: 从检索材料中解析的 {{title, url}}，供前端展示引用链接。
    strategy_tag: retrieval_rag_web | retrieval_rag_policy | retrieval_rag_web_policy | retrieval_fallback
    """
    ori = question.strip()
    policy_blocks = _get_policy_rag_evidence_blocks(ori)
    evidence, tag = get_ranked_evidence_chunks(ori, top_k=top_k)
    if policy_blocks:
        evidence = policy_blocks + (evidence or [])
        if tag is None:
            tag = "retrieval_rag_policy"
        else:
            tag = f"{tag}_policy"

    refs = reference_entries_from_evidence(evidence or [])

    if not evidence:
        logger.info("F: 无可用检索片段（含政策法规索引），使用纯模型生成")
        return model(ori), "retrieval_fallback", refs

    if getattr(cfg, "USE_LANGCHAIN_LCEL_RAG", True):
        try:
            from answers.rag_lcel import answer_via_lcel_faiss_rag, lcel_rag_available

            if lcel_rag_available():
                ans, st = answer_via_lcel_faiss_rag(
                    ori, evidence, tag, model, top_k=top_k
                )
                return ans, st, reference_entries_from_evidence(evidence)
        except Exception as e:
            logger.warning("F: LCEL/FAISS RAG 失败，尝试 stuff 链或拼 prompt: {}", e)

    if getattr(cfg, "USE_LANGCHAIN_RAG", True):
        try:
            from answers.langchain_rag import answer_via_langchain_stuff_chain

            ans, st = answer_via_langchain_stuff_chain(ori, evidence, tag, model)
            return ans, st, reference_entries_from_evidence(evidence)
        except Exception as e:
            logger.warning("F: LangChain stuff 链失败，回退拼 prompt: {}", e)

    prompt = build_rag_prompt(ori, evidence)
    return model(prompt), tag, reference_entries_from_evidence(evidence)
