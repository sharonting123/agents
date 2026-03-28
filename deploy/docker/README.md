# Docker 封装（政府采购投标助手 + qa_fastapi）

## 前置条件

1. 安装 **Docker**、**Docker Compose**（v2：`docker compose`）。
2. **GPU 推理**：安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)，宿主机已能 `nvidia-smi`。
3. 宿主机上已有 **ChatGLM2-6B** 与 **P-Tuning** 权重目录（或构建后通过 volume 挂载）。

## 构建镜像

在 **`Code` 目录**（与 `Dockerfile` 同级）：

```bash
docker build -t procurement-qa:latest .
```

镜像基于 `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`，依赖见 **`requirements-docker.txt`**。若缺包，编辑该文件后重新 `docker build`。

## 运行（命令行）

**不要**把十几 GB 模型 `COPY` 进镜像（镜像会巨大）。用 **volume** 挂载宿主机目录：

```bash
docker run -d --name procurement-qa --gpus all \
  -p 7860:7860 \
  -e LLM_MODEL_DIR=/models/chatglm2-6b \
  -e CODE_BASE_DIR=/app \
  -e MYSQL_HOST=host.docker.internal \
  -v /你的路径/chatglm2-6b:/models/chatglm2-6b:ro \
  -v /你的路径/Code/ptuning:/app/ptuning:ro \
  procurement-qa:latest
```

- **`LLM_MODEL_DIR`**：容器内路径，需与 `-v` 右侧挂载点一致。
- **MySQL 在宿主机**：Windows/Mac Docker Desktop 常用 **`MYSQL_HOST=host.docker.internal`**；Linux 可用 **`--network host`** 或宿主机桥接 IP。
- **Linux** 若 `docker run` 报 GPU：加 `--gpus all`，并确认已装 `nvidia-container-toolkit`。

浏览器：`http://127.0.0.1:7860`

## 使用 docker compose

1. 编辑项目根目录 **`docker-compose.yml`**：把 `volumes` 里 `/path/to/chatglm2-6b` 改成你本机基座目录；按需改 `MYSQL_*`。
2. 在 **`Code` 目录**执行：

```bash
docker compose up -d --build
```

查看日志：`docker compose logs -f`

停止：`docker compose down`

## 环境变量

与 `config/cfg.py` 一致，常用：

| 变量 | 说明 |
|------|------|
| `LLM_MODEL_DIR` | 容器内 ChatGLM2 路径（与 volume 一致） |
| `CODE_BASE_DIR` | 一般为 `/app` |
| `CLASSIFY_CHECKPOINT_PATH` 等 | 若权重不在默认相对路径，设绝对路径 |
| `QA_MODEL_DEVICE_SPLIT` | 默认 `1` |
| `QA_CHATGLM_FP16` | 默认 `1` |
| `MYSQL_HOST` / `MYSQL_USER` / … | 连接数据库 |

可用 `docker run -e ...` 或 compose 的 `environment:` / `env_file:`。

## 长期驻留

容器即长期进程：**只要不 `docker stop` / `down`，模型只加载一次**（首次启动时）。  
更新镜像或改挂载后需 `docker compose up -d --build` 重启，会**再次加载**模型。

## 常见问题

| 问题 | 处理 |
|------|------|
| 镜像内找不到模型 | 检查 `-v` 与 `LLM_MODEL_DIR` 是否一致 |
| CUDA OOM | 保持 `QA_CHATGLM_FP16=1`，或减少并发；宿主机显存要够 |
| 连不上 MySQL | 检查 `MYSQL_HOST`、防火墙、MySQL 是否允许远程 |
| Windows 路径 | `docker-compose.yml` 中写 `D:/Models/...` 或 `//d/Models/...` |

## 与 systemd / 裸进程对比

- **Docker**：环境隔离、易迁移；需处理 GPU 与 volume。  
- **裸进程 / systemd**：无容器层，路径与驱动直接用宿主机。  

按部署环境二选一即可。
