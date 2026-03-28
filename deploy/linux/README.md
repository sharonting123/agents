# Linux 服务器部署（政府采购投标助手 + ChatGLM）

## 1. 准备机器

- **GPU**：推荐 NVIDIA + 驱动；CUDA 与 PyTorch 版本匹配（与本地开发一致最省事）。
- **显存**：12GB 级请保持 `QA_MODEL_DEVICE_SPLIT=1`、`QA_CHATGLM_FP16=1`（脚本与 `env.example` 已默认）。
- **内存**：分类 + NL2SQL 两套 6B 在 CPU 上时，建议 **≥32GB** 系统内存。
- **磁盘**：ChatGLM2-6B + 三套 P-Tuning 权重 + text2vec，预留 **≥30GB**（按实际模型体积调整）。

## 2. 同步代码与模型

1. 将本仓库 **`Code` 目录**（含 `config/`、`answers/`、`web/`、`ptuning/` 等）拷到服务器，例如 `/opt/procurement-assistant/Code`。
2. 将 **ChatGLM2-6B 基座**放到服务器，例如 `/opt/models/chatglm2-6b`。
3. 若 P-Tuning 权重不在仓库内，保持与 Windows 相同的相对路径 `Code/ptuning/...`，或通过环境变量单独指定路径（见 `env.example`）。
4. **text2vec**：放到 `Code/models/text2vec-base-chinese`，或设 `TEXT2VEC_MODEL_DIR`。

配置项由 `config/cfg.py` 读取；**Linux 下**默认：

- `BASE_DIR`：自动为 `config/` 的上一级（即 `Code`），也可用 **`CODE_BASE_DIR`** 覆盖。
- `LLM_MODEL_DIR`：默认 **`/opt/models/chatglm2-6b`**（可用环境变量覆盖）。

## 3. Python 环境

```bash
cd /opt/procurement-assistant/Code   # 按你的路径
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# 依赖以项目内 requirements 为准，例如：
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"
```

按需安装：`torch`（CUDA 版）、`transformers`、`pymysql`、`loguru`、LangChain 相关等（与本地能跑通的环境对齐）。

## 4. 环境变量

复制并编辑：

```bash
cp deploy/linux/env.example deploy/linux/env.local
# 编辑 deploy/linux/env.local，填写 LLM_MODEL_DIR、MYSQL_*、DEEPSEEK_API_KEY（可选）等
```

`run_qa_fastapi.sh` 若存在 `deploy/linux/env.local` 会自动 `source`。

## 5. 启动服务

```bash
cd /opt/procurement-assistant/Code
chmod +x deploy/linux/run_qa_fastapi.sh
# 使用 env.local 中的变量
./deploy/linux/run_qa_fastapi.sh
```

浏览器访问：`http://服务器IP:7860`（防火墙与安全组需放行 **7860** 或你改的 `PORT`）。

**勿用 `--smoke`**：那是无模型连通性测试；正式问答需完整加载模型。

## 6. 长期运行（可选）

### systemd

仓库内示例：**`deploy/linux/procurement-qa.service.example`**（复制后改路径与用户）。

也可手动创建 `/etc/systemd/system/procurement-qa.service`（路径、用户按实际修改）：

```ini
[Unit]
Description=Procurement QA FastAPI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/procurement-assistant/Code
EnvironmentFile=/opt/procurement-assistant/Code/deploy/linux/env.local
ExecStart=/opt/procurement-assistant/Code/.venv/bin/python qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now procurement-qa
```

### Nginx 反向代理（HTTPS）

将 `listen 443` 反代到 `127.0.0.1:7860`，并配置证书；具体域名与证书路径由运维提供。

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| 找不到模型 | 检查 `LLM_MODEL_DIR`、权重路径与权限 |
| CUDA OOM（底座） | 确认 `QA_CHATGLM_FP16=1`、`QA_MODEL_DEVICE_SPLIT=1`；可试 `QA_CHATGLM_DEVICE=cpu`（需足够系统内存，否则勿三套全放 CPU） |
| 内存不足 | 减小并发；或换更大内存机器 |
| MySQL 连不上 | 检查 `MYSQL_*`、防火墙、用户权限 |

## 8. 安全说明

- 不要在公网裸奔端口而不做鉴权；生产环境建议 **Nginx + HTTPS + 访问控制**。
- `DEEPSEEK_API_KEY`、`MYSQL_PASSWORD` 等仅通过环境变量或私密配置文件注入，**勿提交仓库**。
