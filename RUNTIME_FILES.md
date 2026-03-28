# 正常运行 `start_procurement_fastapi.bat` 所涉文件说明

本文档以 **`Code` 目录**（与 `start_procurement_fastapi.bat` 同级）为根，说明从批处理启动到 `/api/chat` 可用时，主要涉及的文件、路径与用途。  
绝对路径示例：`…/pj01/docs/finetune/Code/`（请按你本机克隆位置替换）。

---

## 1. 启动入口

| 路径 | 用途 |
|------|------|
| `start_procurement_fastapi.bat` | Windows 一键启动：设置 `PYTORCH_CUDA_ALLOC_CONF`、`QA_MODEL_DEVICE_SPLIT`、`QA_CHATGLM_FP16` 等，调用 `python qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860`（默认优先 `D:\Python310\python.exe`）。 |
| `qa_fastapi.py` | **FastAPI 应用**：`lifespan` 中加载模型与数据库游标；注册 `GET /`、`GET /api/health`、`POST /api/chat`；挂载 `web/static`。 |

---

## 2. 核心问答链路（Python）

| 路径 | 用途 |
|------|------|
| `qa_chat.py` | **`run_one_round`**：规则分类 → P-Tuning 分类 → A–E 走 NL2SQL+执行，F 走检索+生成。被 `qa_fastapi` 在线程池中调用。 |
| `chatglm_ptuning.py` | **`build_qa_models()`**、**`ChatGLM_Ptuning`**：加载 ChatGLM2 基座与三套 P-Tuning（分类 / NL2SQL / 底座）；`nl2sql` / `classify` / `__call__` 推理。依赖 HuggingFace 缓存中的 ChatGLM 动态模块。 |
| `generate_answer_with_classify.py` | **`_rule_only_class`**、**`_normalize_classification`**：规则先验与分类结果规整到 A–F。 |
| `prompt_util.py` | NL2SQL / SQL 纠错用的 **`nl2sql_prompt_prefix`**、**`prompt_sql_correct`**、**`build_sql_column_catalog`** 等提示词。 |
| `sql_correct_util.py` | **`exc_sql`**、**`normalize_sql_for_mysql`**：执行 SQL、数字/字段纠错辅助。 |
| `company_table.py` | **`get_sql_search_cursor`**：MySQL 或 `data/CompanyTable.csv`→SQLite；**`load_company_table`**：字段名候选（纠错）。 |
| `config/cfg.py` | **集中配置**：`LLM_MODEL_DIR`、各 `*_CHECKPOINT_PATH`、`QA_*` 设备与显存策略、`MYSQL_*`、`SQL_EN_TO_ZH_COLUMNS`、`DATA_PATH`、`KNOWLEDGE_DIR`、DeepSeek 与 LangChain 开关等。通过 **`from config import cfg`** 使用。 |

---

## 3. `answers/` 包（按问题类型加载）

| 路径 | 用途 |
|------|------|
| `answers/constants.py` | `SQL_TRIGGER_CLASSES`、`OPEN_QA_CLASS` 等常量。 |
| `answers/sql_answer.py` | **`answer_with_nl2sql_model`**：NL2SQL 模型生成 SQL → 执行与纠错。 |
| `answers/retrieval.py` | **F 类**：政策 FAISS / DeepSeek 补充片段 → text2vec 重排 → **`answer_via_retrieval`**。 |
| `answers/rag_lcel.py` | 可选：FAISS + LCEL RAG（`USE_LANGCHAIN_LCEL_RAG`）。 |
| `answers/langchain_rag.py` | 可选：LangChain stuff QA 链（`USE_LANGCHAIN_RAG`）。 |
| `answers/orchestrator.py` | 批量答题脚本用编排（**日常 FastAPI 主链路不必须**，但与同一套 `sql_answer`/`retrieval` 一致）。 |

---

## 4. 前端（浏览器访问 `http://127.0.0.1:7860`）

| 路径 | 用途 |
|------|------|
| `web/index.html` | 单页入口，引用 `/static/style.css`、`/static/app.js`。 |
| `web/static/style.css` | 页面样式。 |
| `web/static/app.js` | 调用 **`POST /api/chat`**，展示 `answer` 与分类/路由等元信息。 |

---

## 5. 外部资源（非仓库内代码，但运行必需或可配置）

| 类型 | 路径 / 说明 |
|------|-------------|
| **ChatGLM2 基座** | 默认 **`G:\Models\chatglm2-6b`**（Windows，`cfg.LLM_MODEL_DIR` / 环境变量 **`LLM_MODEL_DIR`** 可覆盖）。需含 `config.json`、`pytorch_model.bin` 分片等。 |
| **HuggingFace 动态模块缓存** | 通常为 **`%USERPROFILE%\.cache\huggingface\modules\transformers_modules\`** 下以模型目录名命名的子目录（如 `chatglm2-6b`），存放 `modeling_chatglm.py` 等。**Windows 下请对基座路径使用反斜杠形式**（如 `G:\Models\chatglm2-6b`），避免 `G:/...` 导致缓存目录异常。 |
| **分类 P-Tuning** | 默认 **`Code/ptuning/CLASSIFY_PTUNING/output/Fin-Train-chatglm2-6b-pt-128-2e-2/checkpoint-50/pytorch_model.bin`**（**`CLASSIFY_CHECKPOINT_PATH`**，`pre_seq_len` 与 NL2SQL 同为 128 时可共享基座）。 |
| **NL2SQL P-Tuning** | 默认 **`Code/ptuning/NL2SQL_PTUNING/output/Fin-Train-chatglm2-6b-pt-128-2e-2/checkpoint-50/pytorch_model.bin`**（**`NL2SQL_CHECKPOINT_PATH`**）。 |
| **关键词 P-Tuning** | 默认 **`Code/ptuning/KEYWORDS_PTUNING/output/Fin-Train-chatglm2-6b-pt-128-2e-2/checkpoint-50/pytorch_model.bin`**（**`KEYWORDS_CHECKPOINT_PATH`**）。 |
| **F 类 text2vec（可选）** | 默认 **`G:\Models\text2vec-base-chinese`**（**`TEXT2VEC_MODEL_DIR`**），用于检索重排；缺失时可降级。 |
| **本地知识库（可选）** | **`Code/data/knowledge_base/`** 下 `.txt`/`.md`（**`KNOWLEDGE_DIR`**）。 |
| **SQL 数据源** | **`USE_MYSQL_FOR_SQL=1`** 时：`cfg.MYSQL_*` 与表 **`MYSQL_SQL_TABLE`**（默认 `shggzy_bid_result`）；否则需 **`Code/data/CompanyTable.csv`** 供 SQLite 内存库。 |
| **代码根目录** | 默认自动推断为 **`config/` 的上一级**；也可设环境变量 **`CODE_BASE_DIR`** 指向本 `Code` 目录。 |
| **设备与内存** | `QA_MODEL_DEVICE_SPLIT=1` 时默认 **分类 / NL2SQL → CPU，底座（F 类）→ GPU**（**`QA_CHATGLM_DEVICE`** 默认 **`cuda`**）。当 **三套 `pre_seq_len` 一致** 时，分类+NL2SQL **共享一份** 6B，仅 **底座** 再加载第二份 6B（共 **2** 次整模加载）。若仍设 **`QA_CHATGLM_DEVICE=cpu`** 且未共享，三套 6B 全在内存易崩溃。 |

---

## 6. 环境与第三方依赖（概要）

| 说明 |
|------|
| **Python**：批处理默认 `D:\Python310\python.exe`，也可改为当前 `PATH` 中的 `python`。 |
| **必备 PyPI**：`torch`（与 CUDA 匹配）、`transformers`、`fastapi`、`uvicorn[standard]`、`pydantic`、`loguru`、`pymysql`（MySQL 时）、`datasets`（若其它脚本用到）等。 |
| **建议**：**`accelerate`**（配合 `low_cpu_mem_usage` 降低加载 6B 时内存峰值）、**`jieba`**、**`rouge-chinese`**、**`nltk`**（与训练/指标相关脚本）。 |
| **F 类可选**：`langchain-*`、`faiss-cpu`、`text2vec` 等按 `cfg` 开关按需安装。 |

---

## 7. 同目录其它启动方式（参考）

| 路径 | 用途 |
|------|------|
| `start_procurement_fastapi_lite.bat` | 精简模式（如仅 NL2SQL），环境变量可能含 `QA_LITE_NL2SQL_ONLY`。 |
| `start_procurement_fastapi_background.bat` | 后台启动变体。 |
| `deploy/linux/run_qa_fastapi.sh` | Linux 下等价启动脚本。 |

---

## 8. 依赖关系简图

```
start_procurement_fastapi.bat
    → qa_fastapi.py
        → chatglm_ptuning.build_qa_models() + qa_chat.run_one_round()
            → 若 can_share_ptuning_prefix：SharedPrefixChatGLM（1×6B + 切换 prefix）+ ChatGLM_Ptuning(Nothing)（F 类，1×6B）
            → 否则：三套独立 ChatGLM_Ptuning（3×6B）
            → generate_answer_with_classify / answers/* / prompt_util / sql_correct_util / company_table
        → company_table.get_sql_search_cursor / load_company_table
    → config.cfg（全局路径与开关）
    → web/*（静态页与 JS）
    → 磁盘：LLM_MODEL_DIR、各 ptuning/pytorch_model.bin、可选 data、knowledge_base、MySQL
```

共享 prefix 说明见 **`ptuning/README_SHARED_PREFIX.md`**。

---

*文档随仓库维护；若你修改了 `cfg` 默认路径或环境变量，请以运行时的实际配置为准。*
