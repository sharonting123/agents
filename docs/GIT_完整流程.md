# Git 完整流程说明（本项目：政府采购投标助手 → GitHub 私有仓库）

本文记录在本项目中已做过的步骤、常见错误与正确做法，便于以后换机器或同事复现。

---

## 1. 仓库位置与初始化

- **项目根目录**：`Code/`（即含 `qa_fastapi.py`、`config/` 的目录）。
- 若目录还不是 Git 仓库：

```bash
cd /path/to/Code
git init
```

---

## 2. 忽略大文件（建议保留）

大模型、`data/`、训练 checkpoint 等体积可达十几 GB，不适合作为普通 Git 对象全量提交。项目中使用 **`.gitignore`** 排除例如：

- `data/`（预训练模型、本地数据）
- `ptuning/**/output/`（P-Tuning 权重目录）
- `ptuning/**/Fin_train/`、`ptuning/**/train_data/*.json`（可按需调整）
- `__pycache__/`、`*.log`、`.venv/` 等

**说明**：代码与脚本仍可完整版本管理；模型与数据通过 README 说明下载路径或单独分发。

若**必须**把大文件也纳入版本控制，需考虑 **Git LFS** 或托管平台单文件/仓库大小限制。

---

## 3. 首次提交

```bash
cd /path/to/Code
git add -A
git status   # 确认无意外的大文件
git config user.name "你的名字"
git config user.email "你的邮箱"
git commit -m "chore: initial commit — 政府采购投标助手"
```

---

## 4. 默认分支命名为 main（与 GitHub 一致）

```bash
git branch -M main
```

---

## 5. 在 GitHub 上创建私有仓库

1. 登录 GitHub → **New repository**。
2. 填写仓库名（示例：`agents`）。
3. 勾选 **Private**。
4. **不要**勾选「Add a README」若本地已有提交（避免无关历史冲突）；若已勾选，见下文「远程已有提交」处理。

---

## 6. 添加远程仓库并推送

### 6.1 HTTPS（示例）

```bash
cd /path/to/Code
git remote remove origin 2>/dev/null
git remote add origin https://github.com/sharonting123/agents.git
git remote -v
git push -u origin main
```

### 6.2 SSH（可选，适合长期开发）

```bash
git remote set-url origin git@github.com:sharonting123/agents.git
ssh -T git@github.com   # 需已在 GitHub 添加本机 SSH 公钥
git push -u origin main
```

---

## 7. 认证：必须使用 Token，不能用「登录密码」

GitHub **已禁止**用账户**网页登录密码**执行 `git push` / `git pull`。

出现以下提示时：

```text
Password for 'https://sharonting123@github.com':
```

此处应输入 **Personal Access Token（PAT）**，不是 GitHub 密码。

若误用密码，会看到：

```text
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/sharonting123/agents.git/'
```

### 7.1 创建 PAT

- 路径：**GitHub → Settings → Developer settings → Personal access tokens**。
- **Classic**：勾选 **`repo`**（私有仓库推送需要）。
- 生成后**只显示一次**，请复制保存。

### 7.2 推送时（交互式：提示输入用户名/密码）

- **Username**：你的 GitHub 用户名（如 `sharonting123`）。
- **Password**：粘贴**整段 PAT**（通常以 `ghp_` 或 `github_pat_` 开头）。

### 7.3 自动推送（非交互，适合脚本/服务器）

在命令里把 **`YOUR_TOKEN`** 换成你的 PAT，**先 `cd` 到项目根目录**再推送（本机路径示例）：

```bash
cd /root/autodl-tmp/Code
git push https://sharonting123:YOUR_TOKEN@github.com/sharonting123/agents.git main
```

**安全注意**：

- Token 会出现在 **shell 历史**（`~/.bash_history`）中；用完后可清理该行或关闭历史记录。
- **切勿**把带真实 Token 的命令写进仓库里的脚本或文档并提交。
- 长期更推荐 **SSH** 或 **credential helper**，避免 Token 明文出现在命令行与历史记录中。

---

## 8. 无界面/服务器上推送的注意点

在 **未配置凭据** 的环境执行 `git push`，进程可能**长时间等待输入**，表现为卡住。需事先：

- 配置 **credential store**，或  
- 使用 **SSH 密钥**，或  
- 使用 **CI** 中的 **secrets**（如 `GITHUB_TOKEN`）  

否则只能在本机或已登录终端中完成推送。

---

## 9. 远程仓库已有 README 等初始提交时

若 `git push` 被拒绝（non-fast-forward），可先拉再合：

```bash
git pull origin main --allow-unrelated-histories
# 解决冲突后
git push -u origin main
```

或按团队规范使用 `rebase`，此处不展开。

---

## 10. 多分支推送示例

若本地还有其它分支（例如 `publish/snapshot-2026-03-28`）：

```bash
git push -u origin publish/snapshot-2026-03-28
```

---

## 11. 清单小结

| 步骤 | 命令或操作 |
|------|------------|
| 初始化 | `git init` |
| 忽略大目录 | 编辑 `.gitignore` |
| 提交 | `git add -A` → `git commit` |
| 主分支 | `git branch -M main` |
| 远程 | `git remote add origin <URL>` |
| 推送 | `git push -u origin main` |
| 认证 | **PAT** 或 **SSH**，不可用网页密码 |

---

## 12. 与本仓库的对应关系

- **远程示例**：`https://github.com/sharonting123/agents.git`（私有）。
- **本地已配置**：在部分环境中已执行 `git remote add origin` 与 `git branch -M main`；**推送必须在有凭据的机器上由你本人完成**。

---

## 13. 版本迭代：分支名 + 打 tag（推荐用于 v2）

大版本（如第二代）建议 **用分支开发、用 tag 标记发布点**，与 `main` 并行，便于回滚与对比。

### 13.1 给当前线打个锚点（可选）

在准备开 v2 之前，可在 `main` 上为「v1 基线」打标签：

```bash
cd /root/autodl-tmp/Code
git checkout main
git pull origin main
git tag -a v1.0.0 -m "v1 稳定基线"
git push origin v1.0.0
```

### 13.2 新建 v2 开发分支

```bash
git checkout main
git pull origin main
git checkout -b v2
# … 在 v2 上开发与提交 …
git push -u origin v2
```

### 13.3 v2 就绪后打发布标签并推送

```bash
git checkout v2
git tag -a v2.0.0 -m "第二代正式发布"
git push origin v2
git push origin v2.0.0
```

一次性推送**所有本地 tag**：

```bash
git push origin --tags
```

### 13.4 是否合并回 main

- **合并**：v2 稳定后 `git checkout main` → `git merge v2` → `git push origin main`，此后 `main` 即代表最新一代。  
- **长期双线**：`main` 维持 v1 仅修 bug，`v2` 继续演进，按需在发布说明里写清默认使用哪条分支。

### 13.5 命名约定（可选）

- 分支：`v2`、`release/v2` 等，团队统一即可。  
- 标签：建议 **语义化版本** `v2.0.0`、`v2.1.0`，与 [SemVer](https://semver.org/lang/zh-CN/) 一致。

---

*文档随流程更新；若 GitHub 政策或界面变更，以官方说明为准。*
