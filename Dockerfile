# 在 Code 目录构建: docker build -t procurement-qa .
# 需 NVIDIA Container Toolkit，运行: docker run --gpus all ...
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 默认路径；运行时可用 -e LLM_MODEL_DIR=... 覆盖
ENV PYTHONUNBUFFERED=1 \
    QA_MODEL_DEVICE_SPLIT=1 \
    QA_CHATGLM_FP16=1 \
    LLM_MODEL_DIR=/models/chatglm2-6b

WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /app/requirements-docker.txt

# 应用代码（模型与大体量 checkpoint 建议运行时 volume 挂载，见 deploy/docker/README.md）
COPY . /app

EXPOSE 7860

CMD ["python", "qa_fastapi.py", "--gpu", "0", "--host", "0.0.0.0", "--port", "7860"]
