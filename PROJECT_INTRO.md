# 政府采购投标助手 — 项目说明

本文档依据当前仓库 **实际代码与运行方式** 编写，用于向他人完整介绍本项目的定位、架构与使用流程。

---

## 1. 项目是什么

**政府采购投标助手**（工作名「甄甄」）是一套面向政府采购与招投标场景的 **智能问答系统**：用户用自然语言提问，系统在 **分类路由** 后，分别走 **结构化数据查询（NL2SQL）** 或 **政策法规类开放问答（检索增强生成，RAG）**，并结合 **ChatGLM2** 与 **P-Tuning** 微调模型生成回答。

- **对外形态**：浏览器访问 **静态页面**（HTML/CSS/JS）+ **REST API**（FastAPI）。
- **推理核心**：**Hugging Face Transformers** + **PyTorch**，基座为本地 **ChatGLM2-6B**，并加载多套 **P-Tuning** 权重（分类、NL2SQL、关键词等场景按配置使用）。
- **数据侧**：业务数据主要在 **MySQL**；可选 **CSV/SQLite** 做离线演示。招投标与政策信息可通过独立 **Python 爬虫脚本** 入库。

---

## 2. 运行时整体架构（从启动到一次问答）

```
┌─────────────────────────────────────────────────────────────────┐
│  Uvicorn 启动 qa_fastapi.py                                      │
│  build_app()：注册 FastAPI、CORS、静态资源、路由                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ lifespan 启动阶段
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  init_db()：用户库表（SQLAlchemy + MySQL）                        │
│  build_qa_models()：加载 ChatGLM2 + P-Tuning（分类 / NL2SQL / 底座）│
│  get_sql_search_cursor()：NL2SQL 所需 MySQL 游标或 CSV→SQLite      │
└────────────────────────────┬────────────────────────────────────┘
                             │ 请求阶段：POST /api/chat
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  asyncio.Lock：同一进程内推理串行，避免 GPU 重入                    │
│  run_in_executor → qa_chat.run_one_round(...)                    │
│    · 规则先验 / P-Tuning 分类 → 规整为 A–F 类                     │
│    · A–E：NL2SQL 生成 SQL → 执行与纠错 → 结果与图表（可选）         │
│    · F：政策 FAISS +（可选）DeepSeek 补充 → text2vec/jieba 重排      │
│          → LangChain LCEL / stuff / 直接 Prompt + ChatGLM         │
└────────────────────────────┬────────────────────────────────────┘
                             │ 若已登录 JWT
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  persist_chat_turn_safe：会话与消息落库（多轮 conversation_id）    │
└─────────────────────────────────────────────────────────────────┘
```

**说明**：未使用 Django、Redis、Celery、Elasticsearch；检索侧为 **本地 FAISS 向量索引** + **句向量模型（text2vec 等）**，与文档型知识库路径见 `config/cfg.py`。

---

## 3. 技术栈（与代码一致）

| 层级 | 技术 |
|------|------|
| Web 框架 | **FastAPI**（ASGI） |
| 应用服务器 | **Uvicorn** |
| 跨域 | **CORSMiddleware**（`qa_fastapi.py` 中 `allow_origins=["*"]` 等） |
| 前端 | 静态 **HTML / CSS / JavaScript**（`web/`） |
| 大模型 | **ChatGLM2-6B** + **P-Tuning**（`chatglm_ptuning.py`） |
| 深度学习 | **PyTorch**、**transformers**、**accelerate**（加载大模型时常用） |
| 用户与持久化 | **SQLAlchemy 2.x** + **PyMySQL**，业务 NL2SQL 直连 **MySQL** |
| 检索 / RAG | **FAISS**、**text2vec**（`SentenceModel` + `semantic_search`）、**jieba**（词法重排降级）；可选 **LangChain**（LCEL / stuff 链） |
| 外部 LLM（可选） | **DeepSeek** 兼容 API（政策类补充材料，环境变量配置密钥） |

---

## 4. 关键目录与文件

| 路径 | 作用 |
|------|------|
| `qa_fastapi.py` | **主入口**：创建 FastAPI、lifespan、挂载 `/`、`/login`、`/static`、`POST /api/chat`、`GET /api/health`；包含 CORS |
| `qa_chat.py` | **`run_one_round`**：单轮问答总编排（分类 → NL2SQL 或 RAG） |
| `chatglm_ptuning.py` | 加载基座与 P-Tuning、推理封装 |
| `config/cfg.py` | **集中配置**：模型路径、checkpoint、MySQL、`POLICY_VECTOR_INDEX_DIR`、`TEXT2VEC_MODEL_DIR`、各类 `QA_*` 开关 |
| `answers/` | F 类检索：`retrieval.py`、`policy_faiss_local.py`、可选 `rag_lcel.py`、`langchain_rag.py`；E 类 NL2SQL：`sql_answer.py` |
| `user_backend/` | 注册登录 JWT、会话与消息、审计相关 API（前缀 `/api/v1`） |
| `web/` | `index.html`、`login.html`、`static/` 下样式与脚本 |
| `ptuning/` | 各任务 **P-Tuning 训练与评估**脚本（与线上一致需对应 checkpoint） |
| `shggzy_crawler.py`、`shggzy_policy_crawler.py` | 上海公共资源等站点 **爬虫**（requests + BeautifulSoup，入库 MySQL） |
| `start_procurement_fastapi.sh`（Linux） | 设置环境变量并执行 `python qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860` |
| `start_procurement_fastapi_smoke.sh` 等 | `--smoke` 或精简模式、RAG 探针等变体启动 |

---

## 5. 如何运行（典型完整模式）

1. **环境**：Python 3，已安装 `torch`（建议与 CUDA 匹配）、`transformers`、`fastapi`、`uvicorn[standard]`、`pymysql`（使用 MySQL 时）等；详见 `requirements.txt`。
2. **模型文件**：将 **ChatGLM2-6B** 置于 `data/pretrained_models/chatglm2-6b/` 或通过环境变量 **`LLM_MODEL_DIR`** 指定；P-Tuning 权重路径见 **`config/cfg.py`** 中 `CLASSIFY_CHECKPOINT_PATH`、`NL2SQL_CHECKPOINT_PATH`、`KEYWORDS_CHECKPOINT_PATH`。
3. **启动**（在 `Code` 目录下）：

   ```bash
   chmod +x start_procurement_fastapi.sh
   ./start_procurement_fastapi.sh
   ```

   或直接：

   ```bash
   python qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
   ```

4. **访问**：浏览器打开 `http://127.0.0.1:7860`；API 示例：`POST /api/chat`，JSON body 含 `message`，可选 `agent_mode`（`auto` / `nl2sql` / `policy`）、`conversation_id`（登录多轮）。

**仅验证页面与接口、不加载 GPU 模型**：

```bash
python qa_fastapi.py --smoke --host 127.0.0.1 --port 7860
```

此时 `/api/chat` 会返回 **503**（模型未加载），用于连通性测试。

**常用环境变量**（脚本或 shell 中导出）：`LLM_MODEL_DIR`、`CUDA_VISIBLE_DEVICES`（或由 `--gpu` 传入）、`QA_CLASSIFY_DEVICE` / `QA_CHATGLM_DEVICE`、`CHATGLM_LOAD_IN_8BIT`、`POLICY_VECTOR_INDEX_DIR`、`TEXT2VEC_MODEL_DIR` 等，详见 `config/cfg.py` 与 `start_procurement_fastapi.sh`。

---

## 6. 问答逻辑概要（`run_one_round`）

1. **输入**：用户问题字符串；可选 **`agent_mode`** 跳过分类，强制走库表或政策链路。
2. **分类**：完整模式下经 **规则先验** 与 **P-Tuning 分类模型**，将意图归一到 **A–F**（其中 **F** 为开放/政策咨询类，见 `answers/constants.py`）。
3. **A–E（与 SQL 相关类）**：由 **NL2SQL 专用 P-Tuning 模型** 生成 SQL，经执行与纠错模块跑在 **MySQL**（或 CSV 回退）上；部分场景可返回 **ECharts** 图表配置（`chart` 字段）。
4. **F**：从 **政策法规 FAISS 索引** 检索片段，可选 **DeepSeek** 拉取补充摘要，再经 **text2vec 语义重排** 或 **jieba + Jaccard 式词集重叠** 降级；生成阶段优先 **LangChain LCEL**（可配置），否则 stuff 链，再否则直接拼 **Prompt + ChatGLM**；响应可带 **`references`**（从材料中解析的链接）。
5. **输出**：回答正文、策略名、路由说明、分类原始片段、规整类别、可选图表与引用列表；登录用户可写入会话并返回 **`conversation_id`**。

精简模式（`QA_LITE_NL2SQL_ONLY` 等）与 RAG 探针（`QA_RAG_PROBE`）会缩短加载路径或只测检索链路，详见 `qa_chat.py` 注释与 `cfg`。

---

## 7. API  surface（摘要）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 主页面 |
| GET | `/login` | 登录页 |
| GET | `/static/...` | 静态资源 |
| GET | `/api/health` | 健康检查：模型是否加载、SQL 游标是否可用等 |
| POST | `/api/chat` | 主问答接口 |
| * | `/api/v1/...` | 用户注册登录、会话、评价等（`user_backend/routers.py`） |

跨域由 **`CORSMiddleware`** 统一处理，便于前后端分离部署。

---

## 8. 数据与爬虫

- **业务表**：NL2SQL 依赖的表名、字段映射等在 `config/cfg.py` 与 `company_table.py` 等处配置。
- **爬虫**：独立脚本将公告、政策等写入 MySQL；**政策法规向量索引** 一般由库表正文离线构建 FAISS（`policy.faiss` + `policy_chunks.json`），路径由 **`POLICY_VECTOR_INDEX_DIR`** 指定。

---

## 9. 训练与评估（离线）

`ptuning/` 下各子目录提供 **CLASSIFY / NL2SQL / KEYWORDS** 等任务的训练与 `evaluate.sh`；评估脚本中可能使用 **jieba**、**nltk**（如 BLEU）等，与线上一键推理路径不同。

---

## 10. 部署与文档索引

- Linux 部署说明可参考 **`deploy/linux/README.md`**。
- 启动所涉文件清单可参考 **`RUNTIME_FILES.md`**（与本文互补，偏「文件级清单」）。
- 技术原理与评估详版见仓库内 **`政府采购投标助手_技术原理与评估_详版.md`**（若存在）。

---

## 11. 版本与免责

- 具体默认路径、checkpoint 步数、第三方 API Key **以 `config/cfg.py` 与环境变量为准**；升级 `transformers` 后可能需兼容补丁（如 `chatglm_ptuning.py` 中针对旧版 ChatGLM modeling 的适配）。
- 本文描述的是**当前仓库实现**；若你方另有未合并分支或私有化组件，请自行补充说明。

---

*文档生成依据：`qa_fastapi.py`、`qa_chat.py`、`config/cfg.py`、`user_backend/`、`answers/` 等；随代码演进请同步更新本文。*
