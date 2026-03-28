PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
SAVE_STEPS=10
# 优先使用与本仓库并列的 LLaMA-Factory venv（含 jieba/torch 等）；否则用 PATH 里的 python
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_VENV_PY="${_SCRIPT_DIR}/../../../LLaMA-Factory/.venv310/Scripts/python.exe"
if [ -x "$_VENV_PY" ]; then
  PYTHON="$_VENV_PY"
else
  PYTHON="${PYTHON:-python}"
fi
CUDA_VISIBLE_DEVICES=0 "$PYTHON" main.py \
    --do_train \
    --train_file 'train_data/nl2sql_train_data.json' \
    --validation_file 'train_data/nl2sql_dev_data.json' \
    --preprocessing_num_workers 10 \
    --prompt_column question \
    --response_column answer \
    --overwrite_cache \
    --model_name_or_path G:/Models/chatglm2-6b \
    --output_dir output/Fin-Train-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 2200 \
    --max_target_length 300 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 50 \
    --logging_steps 10 \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 8

