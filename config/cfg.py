import os
import sys

# 分类 P-Tuning 与 NL2SQL 的 pre_seq_len 不一致时，问答侧分别加载两套 6B+P-Tuning（见 build_qa_models）。
# 注意：这些参数需要与实际训练的模型参数匹配
CLASSIFY_PTUNING_PRE_SEQ_LEN = 512
KEYWORDS_PTUNING_PRE_SEQ_LEN = 256
NL2SQL_PTUNING_PRE_SEQ_LEN = 128
NL2SQL_PTUNING_MAX_LENGTH = 2200
# model.chat 生成长度上限（与 prefix 长度无关；勿用 pre_seq_len 当生成长度）
CLASSIFY_GEN_MAX_LENGTH = int(os.environ.get("CLASSIFY_GEN_MAX_LENGTH", "512"))
# 底座大模型意图识别（A/F）生成长度上限
INTENT_LLM_MAX_LENGTH = int(os.environ.get("INTENT_LLM_MAX_LENGTH", "256"))
KEYWORDS_GEN_MAX_LENGTH = int(os.environ.get("KEYWORDS_GEN_MAX_LENGTH", "256"))

# 项目根目录：本文件在 config/cfg.py，上一级为 Code。Linux 部署可设环境变量 CODE_BASE_DIR=/path/to/Code
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_ROOT = os.path.dirname(_CONFIG_DIR)
BASE_DIR = os.path.abspath(os.environ.get("CODE_BASE_DIR", _CODE_ROOT))
if not BASE_DIR.endswith(os.sep):
    BASE_DIR += os.sep
DATA_PATH = os.path.join(BASE_DIR, "data") + os.sep
_ROOT = BASE_DIR.rstrip(os.sep)

# 历史/其它脚本可能用到；F 类主检索已改为政策法规 FAISS（POLICY_VECTOR_INDEX_DIR），不再读此目录
KNOWLEDGE_DIR = os.path.join(DATA_PATH, "knowledge_base")
# 政策法规 FAISS 索引（policy.faiss + policy_chunks.json）；未设 POLICY_VECTOR_INDEX_DIR 时自动探测常见位置
def _resolve_default_policy_vector_index_dir() -> str:
    _workspace = os.path.dirname(_ROOT)  # Code 的上一级，一般为项目/工作区根目录
    candidates = [
        # 与仓库同级的 Code/vector_index_policy（policy.faiss + policy_chunks.json）
        os.path.join(_ROOT, "vector_index_policy"),
        os.path.join(_workspace, "pj01", "docs", "vector_index_policy"),
        os.path.join(_workspace, "docs", "vector_index_policy"),
        # 历史默认（曾错误地指向「上两级/vector_index_policy」，兼容仍把索引放在该路径的情况）
        os.path.join(os.path.dirname(_workspace), "vector_index_policy"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "policy.faiss")):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


_pv_env = os.environ.get("POLICY_VECTOR_INDEX_DIR", "").strip()
POLICY_VECTOR_INDEX_DIR = _pv_env if _pv_env else _resolve_default_policy_vector_index_dir()
POLICY_VECTOR_RAG_ENABLED = os.environ.get("POLICY_VECTOR_RAG_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)
# F 类：DeepSeek 补充材料（OpenAI 兼容，见 answers/deepseek_supplement.py）
# 生产环境建议改用环境变量 DEEPSEEK_API_KEY，勿将密钥提交到公开仓库
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY", "sk-6572a036e8c1410da89d5c05871eedcd"
).strip()
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
F_USE_DEEPSEEK_SUPPLEMENT = os.environ.get("F_USE_DEEPSEEK_SUPPLEMENT", "1").lower() not in (
    "0",
    "false",
    "no",
)
# F 类：是否用 LangChain 的 stuff QA 链做 RAG
USE_LANGCHAIN_RAG = os.environ.get("USE_LANGCHAIN_RAG", "1").lower() not in ("0", "false", "no")
# F 类：是否优先用「FAISS + RunnableParallel」LCEL 管线；需 langchain-core / community / faiss-cpu
USE_LANGCHAIN_LCEL_RAG = os.environ.get("USE_LANGCHAIN_LCEL_RAG", "1").lower() not in ("0", "false", "no")
# 本地 text2vec 句向量；Linux 默认放在 Code/models/ 下，可用 TEXT2VEC_MODEL_DIR 覆盖
_DEFAULT_TEXT2VEC = (
    os.path.join(_ROOT, "data", "pretrained_models", "text2vec-base-chinese")
    if sys.platform == "win32"
    else os.path.join(_ROOT, "data", "pretrained_models", "text2vec-base-chinese")
)
TEXT2VEC_MODEL_DIR = os.environ.get("TEXT2VEC_MODEL_DIR", _DEFAULT_TEXT2VEC)
# LCEL/FAISS 嵌入：本地目录不存在时改用 HuggingFace 模型 ID（首次会下载）；设为空字符串则禁用该回退
TEXT2VEC_FALLBACK_MODEL_ID = os.environ.get(
    "TEXT2VEC_FALLBACK_MODEL_ID", "shibing624/text2vec-base-chinese"
).strip()
NUM_PROCESSES = 64
# 默认：Fin-Train-chatglm2-6b-pt-512-2e-2/checkpoint-* 下需含 pytorch_model.bin
CLASSIFY_CHECKPOINT_PATH = os.environ.get(
    "CLASSIFY_CHECKPOINT_PATH",
    os.path.join(
        _ROOT,
        "ptuning",
        "CLASSIFY_PTUNING",
        "output",
        "Fin-Train-chatglm2-6b-pt-512-2e-2",
        "checkpoint-10",
    ),
)
NL2SQL_CHECKPOINT_PATH = os.environ.get(
    "NL2SQL_CHECKPOINT_PATH",
    os.path.join(
        _ROOT,
        "ptuning",
        "NL2SQL_PTUNING",
        "output",
        "Fin-Train-chatglm2-6b-pt-128-2e-2",
        "checkpoint-50",
    ),
)
KEYWORDS_CHECKPOINT_PATH = os.environ.get(
    "KEYWORDS_CHECKPOINT_PATH",
    os.path.join(
        _ROOT,
        "ptuning",
        "KEYWORDS_PTUNING",
        "output",
        "Fin-Train-chatglm2-6b-pt-256-2e-2",
        "checkpoint-20",
    ),
)
# ChatGLM2 基座：Windows 默认 G:\Models\chatglm2-6b；其它系统默认 Code/data/pretrained_models/chatglm2-6b。可用环境变量 LLM_MODEL_DIR 覆盖
if sys.platform == "win32":
    _DEFAULT_LLM = r"G:\Models\chatglm2-6b"
else:
    _DEFAULT_LLM = os.path.join(_ROOT, "data", "pretrained_models", "chatglm2-6b")
LLM_MODEL_DIR = os.environ.get("LLM_MODEL_DIR", _DEFAULT_LLM)
# 问答：QA_MODEL_DEVICE_SPLIT=1 时仅作「多模型分流」标记；NL2SQL 与底座在 CUDA 可用时均优先用 GPU（见 qa_chatglm_device_split）。
# 显存不足时可设 QA_CHATGLM_DEVICE=cpu 等；单卡多模型 OOM 时再设 CHATGLM_LOAD_IN_8BIT=1 或拆分设备。
QA_MODEL_DEVICE_SPLIT = os.environ.get("QA_MODEL_DEVICE_SPLIT", "1").lower() not in (
    "0",
    "false",
    "no",
)
# 推理用 FP16 主干（P-Tuning 时 prefix_encoder 仍为 FP32，与 ptuning/*/main.py 一致）。12GB 卡上关则极易 OOM。
QA_CHATGLM_FP16 = os.environ.get("QA_CHATGLM_FP16", "1").lower() not in (
    "0",
    "false",
    "no",
)
# 仅 NL2SQL P-Tuning：整模 FP32 加载，缓解部分环境「Cannot copy out of meta tensor」；显存会升高
QA_NL2SQL_FP32 = os.environ.get("QA_NL2SQL_FP32", "0").lower() in ("1", "true", "yes")
# 分类 P-Tuning 整模 FP32（分类在 GPU 上若仍 meta tensor 可设 1；CPU 上默认已自动 FP32）
QA_CLASSIFY_FP32 = os.environ.get("QA_CLASSIFY_FP32", "0").lower() in ("1", "true", "yes")
# 底座聊天模型（PtuningType.Nothing）：默认 cuda（与分类/NL2SQL 错开设备，避免三套 6B 全在 CPU 撑爆内存）。
# 仅当底座上 GPU 仍 OOM 时，再设 QA_CHATGLM_DEVICE=cpu（需足够大内存）。
_QCHAT = os.environ.get("QA_CHATGLM_DEVICE", "cuda").strip().lower()
QA_CHATGLM_DEVICE = _QCHAT if _QCHAT in ("cpu", "cuda") else "cuda"
# RAG 探针模式（精简模式 2）：仅加载底座，只走向量检索+RAG（F 链路），不加载分类/NL2SQL，不要求 MySQL。
# 用于验证 POLICY_VECTOR_INDEX_DIR、LangChain RAG 等是否通。与 QA_LITE_NL2SQL_ONLY 同时设时本项优先。
QA_RAG_PROBE = os.environ.get("QA_RAG_PROBE", "0").lower() in (
    "1",
    "true",
    "yes",
)
# 仅加载 NL2SQL 一套 P-Tuning（不加载分类）；若 QA_LITE_FORCE_CLASS=F 则额外加载底座用于开放问答；见 build_qa_models
QA_LITE_NL2SQL_ONLY = os.environ.get("QA_LITE_NL2SQL_ONLY", "0").lower() in (
    "1",
    "true",
    "yes",
)
# 精简模式下不跑分类：规则未命中时固定走 A–E 之一或 F（开放问答）；环境变量未设时默认 E；bat 可设为 F
_QLFC = os.environ.get("QA_LITE_FORCE_CLASS", "E").strip().upper()[:1]
QA_LITE_FORCE_CLASS = _QLFC if _QLFC in ("A", "B", "C", "D", "E", "F") else "E"
# 精简模式下 NL2SQL 所用 device；不设则沿用 qa_chatglm_device_split 的 nl2sql 槽位（CUDA 可用时为 cuda）
_QALD = os.environ.get("QA_LITE_NL2SQL_DEVICE", "").strip()
QA_LITE_NL2SQL_DEVICE = _QALD if _QALD else None
# 完整模式下分类 P-Tuning 所用 device；默认 cuda；单卡 OOM 时设 QA_CLASSIFY_DEVICE=cpu
_raw_cls = os.environ.get("QA_CLASSIFY_DEVICE", "cuda").strip().lower()
QA_CLASSIFY_DEVICE = _raw_cls if _raw_cls in ("cpu", "cuda") else "cuda"


def qa_chatglm_device_split():
    """返回 (classify, nl2sql, chat) 的 device。

    - **QA_CLASSIFY_DEVICE** 默认 cuda；设为 cpu 可省显存（与 QA_MODEL_DEVICE_SPLIT 组合见下）。
    - QA_MODEL_DEVICE_SPLIT=0：未单独设环境时三槽位由 cfg 默认与 build_qa_models 解析为 cuda。
    - QA_MODEL_DEVICE_SPLIT=1：分类槽位可被 QA_CLASSIFY_DEVICE 覆盖。
    """
    import torch

    if not torch.cuda.is_available():
        return "cpu", "cpu", "cpu"

    d_chat = QA_CHATGLM_DEVICE if QA_CHATGLM_DEVICE in ("cpu", "cuda") else "cuda"
    explicit_cls = QA_CLASSIFY_DEVICE

    if QA_MODEL_DEVICE_SPLIT:
        d_cls = explicit_cls if explicit_cls is not None else "cuda"
        return d_cls, "cuda", d_chat

    if explicit_cls is not None:
        return explicit_cls, "cuda", d_chat

    return None, None, None


def can_share_ptuning_prefix():
    """分类与 NL2SQL pre_seq_len 一致时才可共用 SharedPrefixChatGLM（当前 512≠128，为 False）。"""
    try:
        c = CLASSIFY_PTUNING_PRE_SEQ_LEN
        n = NL2SQL_PTUNING_PRE_SEQ_LEN
        if c is None or n is None:
            return False
        return int(c) == int(n)
    except Exception:
        return False


# 基座量化加载（需 CUDA + pip install bitsandbytes；量化时会跳过 .half()）
# 默认 8bit；4bit 更省显存；若同时设 1，优先 4bit。关闭 8bit：CHATGLM_LOAD_IN_8BIT=0
# 注：分类/关键词/NL2SQL 等 **P-Tuning 带 prefix** 在 chatglm_ptuning 中恒为 FP16 整模加载（不受本项影响），避免 8bit+prefix 触发 meta tensor。
# QA_CLASSIFY_DEVICE=cpu 时分类在内存中加载；8bit 主要用于 **PtuningType.Nothing（开放问答底座）**。
CHATGLM_LOAD_IN_4BIT = os.environ.get("CHATGLM_LOAD_IN_4BIT", "0").lower() in ("1", "true", "yes")
CHATGLM_LOAD_IN_8BIT = os.environ.get("CHATGLM_LOAD_IN_8BIT", "1").lower() in ("1", "true", "yes")
# 8bit 时允许 accelerate 把部分层放到 CPU（单卡连装两套 6B 时常需开启，否则报 validate_environment）
CHATGLM_BNB_INT8_CPU_OFFLOAD = os.environ.get("CHATGLM_BNB_INT8_CPU_OFFLOAD", "1").lower() in (
    "1",
    "true",
    "yes",
)
# 第二套底座（开放问答 PtuningType.Nothing）是否也 8bit。默认 0：用 FP16，避免双 8bit dispatch 时 bnb Int8Params 与 accelerate 不兼容
CHATGLM_LOAD_IN_8BIT_FOR_NOTHING = os.environ.get("CHATGLM_LOAD_IN_8BIT_FOR_NOTHING", "0").lower() in (
    "1",
    "true",
    "yes",
)
# 已弃用：chatglm_ptuning 中 from_pretrained 恒为 low_cpu_mem_usage=False（旧 ChatGLM+8bit+meta 易崩）
CHATGLM_LOW_CPU_MEM_USAGE = os.environ.get("CHATGLM_LOW_CPU_MEM_USAGE", "0").lower() in ("1", "true", "yes")


# ── NL2SQL 执行：走 MySQL（与 docs/shggzy_crawler.py 中库表一致）────────────────
# 设为 0/false 则仍用原逻辑：内存 SQLite + data/CompanyTable.csv
# 默认连接（可用 MYSQL_HOST / MYSQL_DATABASE 等环境变量覆盖）:
#   MYSQL_CONFIG = { host, port, user, password, database: "bidding", charset: "utf8mb4" }
# 用户管理 qa_* 表与用户后端默认使用同一库；启动时会 CREATE DATABASE IF NOT EXISTS + create_all。
USE_MYSQL_FOR_SQL = os.environ.get("USE_MYSQL_FOR_SQL", "1").lower() not in ("0", "false", "no")
MYSQL_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "39.105.216.24"),
    "port": int(os.environ.get("MYSQL_PORT", "3306")),
    "user": os.environ.get("MYSQL_USER", "ztt"),
    "password": os.environ.get("MYSQL_PASSWORD", "Ti123456!"),
    "database": os.environ.get("MYSQL_DATABASE", "bidding"),
    "charset": "utf8mb4",
}
# 模型里常写 company_table，执行时映射为真实表（中标结果主表）
MYSQL_SQL_TABLE = os.environ.get("MYSQL_SQL_TABLE", "shggzy_bid_result")
# A–E：用户问题含「图表/可视化/数据分析」等时，在可绘图时返回 ECharts option（见 answers/sql_chart.py）
SQL_CHART_ENABLED = os.environ.get("SQL_CHART_ENABLED", "1").lower() not in ("0", "false", "no")
# MySQL 英文列名 -> 中文注释（与 DDL COMMENT / NL2SQL 输出一致）
# normalize：把 SQL 里的中文替换成 `英文列名`；纠错：候选词用右侧中文
SQL_EN_TO_ZH_COLUMNS = {
    "agency_contact": "招标代理机构联系人/地址",
    "agency_service_fee_amount": "代理服务收费金额(元)",
    "bidder": "招标人/采购单位",
    "agency_phone": "招标代理机构联系方式",
    "bidder_phone": "招标人联系方式",
    "bid_amount": "中标金额(元)",
    "detail_url": "详情页链接",
    "title": "公告标题",
    "project_no": "项目编号",
    "publish_date": "发布日期",
    "winner_phone": "中标人联系方式",
    "winner_contact": "中标人联系人",
    "agency": "招标代理机构",
    "bid_content": "招标内容",
    "project_location": "项目地点",
    "tender_file": "招标文件",
    "bidder_contact": "招标人地址",
    "winner": "中标人",
    "crawl_time": "抓取时间",
    "url_hash": "URL MD5去重键",
}

# ── 用户管理 / 对话审计：默认走与 NL2SQL 相同的 MySQL（MYSQL_CONFIG），表名 qa_* 避免与业务表冲突
# USER_APP_USE_MYSQL=0 时回退本地 SQLite（USER_APP_DB_PATH）
USER_APP_USE_MYSQL = os.environ.get("USER_APP_USE_MYSQL", "1").lower() not in ("0", "false", "no")
# 留空则与 MYSQL_CONFIG["database"] 相同（即「原来的」业务库）
USER_APP_MYSQL_DATABASE = os.environ.get("USER_APP_MYSQL_DATABASE", "").strip() or None
USER_APP_DB_PATH = os.path.abspath(
    os.environ.get(
        "USER_APP_DB_PATH",
        os.path.join(DATA_PATH, "user_app.sqlite3"),
    )
)
# JWT：生产环境务必设置强随机字符串
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "change-me-in-production-use-openssl-rand-hex-32")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))  # 默认 7 天

# 供 `from config import cfg` 使用：cfg 指向本模块，属性即上列常量
cfg = sys.modules[__name__]
