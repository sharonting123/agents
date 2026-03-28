# 与 CLASSIFY / NL2SQL 一致（128）方可与 SharedPrefixChatGLM 共享一份基座
PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
RUNNING_STEP=20
SAVE_STEPS=100
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
    --train_file Fin_train/train.json \
    --validation_file Fin_train/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column question_prompt \
    --response_column query \
    --overwrite_cache \
    --model_name_or_path G:/Models/chatglm2-6b \
    --output_dir output/Fin-Train-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 512 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps $RUNNING_STEP \
    --logging_steps 2 \
    --save_steps $SAVE_STEPS \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \

