# 归档说明（未参与主运行链路的文件）

主入口为 **`qa_fastapi.py`**（及 `start_procurement_fastapi.sh` / `.bat`）。以下目录/文件已从 `Code/` 根目录移入本文件夹，**不影响**正常启动与 `POST /api/chat`。

## 目录与内容

| 原路径 | 说明 |
|--------|------|
| `.ipynb_checkpoints/` | Jupyter 自动保存的检查点脚本副本 |
| `web_demo/` | 独立 Gradio 演示（`app.py`），与 `web/` + FastAPI 主站并行无关 |
| `preprocess.py` | 旧流水线占位脚本（仅打印说明，未被 `qa_fastapi` 引用） |
| `test_score.py` | 独立评测脚本（编号 [21]），非在线服务依赖 |
| `test_score.py` 同级 | 若需批量评测，可从本目录拷回或 `python backup/test_score.py`（需在 `Code` 下且 `PYTHONPATH` 正确） |
| `answers/replay_pid*.log` → `logs/` | 运行遗留日志 |
| `tools/run_chart_smoke.py` | 图表链路自检小工具，非服务启动必需 |
| `requirements_legacy/` | 除 **`requirements（以这个为准）.txt`** 外的历史/平台 requirements 副本 |
| `docs/` | 阅读顺序说明、旧「金融年报」实战文档、技术路线图等（查阅用，非运行时依赖） |

## 仍保留在 `Code/` 根目录的相关入口（未移动）

- `main.py`：批量「分类→关键词→SQL→答题」管线，独立入口，仍可能被评测工作流使用  
- `qa_gradio.py`：Gradio 替代界面入口  
- `qa_chat.py`：交互 CLI，被文档引用  
- `deploy/`、`RUNTIME_FILES.md`、`README.md`：部署与说明  

## 恢复

若需某一文件回到原位置：

```bash
# 示例：恢复 web_demo
mv backup/web_demo /path/to/Code/
```
