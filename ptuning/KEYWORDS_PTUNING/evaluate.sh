#!/usr/bin/env bash
# 关键词 P-Tuning 预测。与仓库内已有 checkpoint 一致：pre_seq_len=256，checkpoint-20。
# Windows 可设 MODEL_NAME_OR_PATH=G:/Models/chatglm2-6b
set -euo pipefail
cd "$(dirname "$0")"

PRE_SEQ_LEN="${PRE_SEQ_LEN:-256}"
CHECKPOINT="${CHECKPOINT:-Fin-Train-chatglm2-6b-pt-256-2e-2}"
STEP="${STEP:-20}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-tmp/Code/data/pretrained_models/chatglm2-6b}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python main.py \
    --do_predict \
    --validation_file Fin_train/dev.json \
    --test_file Fin_train/dev.json \
    --overwrite_cache \
    --prompt_column question_prompt \
    --response_column query \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --ptuning_checkpoint "./output/${CHECKPOINT}/checkpoint-${STEP}" \
    --output_dir "./output/${CHECKPOINT}" \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len "$PRE_SEQ_LEN"
