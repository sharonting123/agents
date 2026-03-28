# Windows：长期驻留（不重载模型）

原则：**只要 Python 进程不退出，已加载的 3 套模型一直在内存/显存里**，无需每次问答再加载。  
以下方式任选；**关掉对应进程/服务后**，下次启动仍会重新加载模型。

## 1. 最小化后台窗口（最简单）

在 `Code` 目录双击：

- **`start_procurement_fastapi_background.bat`**

会新开一个**最小化**的控制台窗口跑 `qa_fastapi.py`。  
**不要关这个窗口**；关了 = 进程结束 = 下次要重新加载。

---

## 2. 任务计划程序（开机自启、可「不管是否登录」）

1. `Win + R` → `taskschd.msc`
2. 创建基本任务 → 触发器：**计算机启动时**（或**用户登录时**）
3. 操作：**启动程序**
   - 程序：`D:\Python310\python.exe`（按你本机路径）
   - 添加参数：`qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860`
   - 起始于：`d:\AIpractice\week01\pj01\docs\finetune\Code`（改成你的 `Code` 路径）
4. 在任务属性 → **常规** 勾选「使用最高权限运行」（若需要）
5. 在 **环境** 中或通过「起始于」前的批处理设置 `QA_MODEL_DEVICE_SPLIT=1`、`QA_CHATGLM_FP16=1`（可先写一个 `.bat` 再让计划任务只调这个 bat）

---

## 3. NSSM 安装为 Windows 服务（无窗口、崩溃可重启）

1. 下载 [NSSM](https://nssm.cc/download)，解压 `nssm.exe`（64 位）。
2. 以**管理员** CMD：

```bat
cd 你解压的nssm目录\win64
nssm install ProcurementQA
```

在弹出界面中：

- **Path**：`D:\Python310\python.exe`
- **Startup directory**：`d:\AIpractice\week01\pj01\docs\finetune\Code`
- **Arguments**：`qa_fastapi.py --gpu 0 --host 0.0.0.0 --port 7860`

在 **Environment** 选项卡添加：

- `QA_MODEL_DEVICE_SPLIT=1`
- `QA_CHATGLM_FP16=1`
- `PYTHONUNBUFFERED=1`

**Install service** 后：

```bat
nssm start ProcurementQA
```

查看日志可在 NSSM 里配 **I/O** 重定向到文件。

卸载：`nssm remove ProcurementQA confirm`

---

## 4. Linux 服务器

见 **`deploy/linux/README.md`** 中的 **systemd**（`enable --now` 后进程常驻、可开机自启）。

仓库内可复制：`deploy/linux/procurement-qa.service.example`（按需改路径与用户）。

---

## 注意

- 改代码后需**重启**对应进程/服务，新代码才生效；**模型会再加载一次**。
- 防火墙若开启，需放行 **7860**（或你改的端口）。
