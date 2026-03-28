#!/usr/bin/env bash
# NL2SQL P-Tuning 预测。默认：pre_seq_len=128，checkpoint-50，8bit（与 train 一致）。
# Windows 可设 MODEL_NAME_OR_PATH=G:/Models/chatglm2-6b
set -euo pipefail
cd "$(dirname "$0")"

PRE_SEQ_LEN="${PRE_SEQ_LEN:-128}"
CHECKPOINT="${CHECKPOINT:-Fin-Train-chatglm2-6b-pt-128-2e-2}"
STEP="${STEP:-50}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-tmp/Code/data/pretrained_models/chatglm2-6b}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python main.py \
    --do_predict \
    --validation_file train_data/nl2sql_dev_data.json \
    --test_file train_data/nl2sql_dev_data.json \
    --overwrite_cache \
    --prompt_column question \
    --response_column answer \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --ptuning_checkpoint "./output/${CHECKPOINT}/checkpoint-${STEP}" \
    --output_dir "./output/${CHECKPOINT}" \
    --max_source_length 2200 \
    --max_target_length 300 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len "$PRE_SEQ_LEN" \
    --quantization_bit 8
