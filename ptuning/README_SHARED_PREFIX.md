# 三套 P-Tuning 共用同一 `pre_seq_len`（与 NL2SQL 一致为 128）

## 行为说明

当 `config/cfg.py` 中 **`CLASSIFY_PTUNING_PRE_SEQ_LEN`、`KEYWORDS_PTUNING_PRE_SEQ_LEN`、`NL2SQL_PTUNING_PRE_SEQ_LEN` 三者相等** 时，`chatglm_ptuning.build_qa_models()` 会走 **`SharedPrefixChatGLM`**：

- **仅加载一份** ChatGLM2 6B 基座；
- 分类 / 关键词 / NL2SQL 在推理前通过 **`load_prefix_checkpoint`** 切换各自的 `pytorch_model.bin`（prefix）；
- **开放问答（F 类）** 仍使用 **`ChatGLM_Ptuning(PtuningType.Nothing)`**：无 P-Tuning 前缀的基座，因此启动时共 **两份** 6B（共享实例 + 底座），比原先三套独立加载少一次整模加载。

## 你必须重新训练

旧版分类/关键词在 **512 / 256** 下训练的 **prefix 张量形状** 与 **128** 不兼容，不能直接混用。

1. 在 `Code` 下生成数据（与往常一样）：
   ```bash
   python ptuning/generate_procurement_ptuning_data.py
   ```
2. 分别训练（`pre_seq_len=128`，输出目录形如 `output/Fin-Train-chatglm2-6b-pt-128-2e-2`）：
   - `ptuning/CLASSIFY_PTUNING/train.sh`
   - `ptuning/KEYWORDS_PTUNING/train.sh`
   - `ptuning/NL2SQL_PTUNING/train.sh`
3. 将各任务 `checkpoint-*/pytorch_model.bin` 拷到或对齐 **`cfg.CLASSIFY_CHECKPOINT_PATH` / `KEYWORDS_CHECKPOINT_PATH` / `NL2SQL_CHECKPOINT_PATH`**（默认已指向 `.../Fin-Train-chatglm2-6b-pt-128-2e-2/checkpoint-50`，可按你实际步数改环境变量）。

## 推理时的生成长度

`CLASSIFY_GEN_MAX_LENGTH`、`KEYWORDS_GEN_MAX_LENGTH` 与 **prefix 长度无关**，用于 `model.chat` 的 `max_length`；默认分别为 **512 / 256**，避免关键词被截断。
