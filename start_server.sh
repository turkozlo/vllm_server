#!/bin/bash

# Configuration
# Укажите путь к папке с моделью (можно относительный)
MODEL_PATH="./qwen_model" 

# Превращаем путь в абсолютный для надежности
MODEL=$(readlink -f "$MODEL_PATH")

# Конфигурация vLLM
TP_SIZE=2
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=32768
GPU_MEM_UTIL=0.90

# Настройки для работы без интернета
export HF_HUB_OFFLINE=1
export VLLM_OFFLINE_MODE=1

echo "Starting vLLM server in OFFLINE mode"
echo "Model path: $MODEL"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Host: $HOST, Port: $PORT"

# Run vLLM
# Using python -m vllm.entrypoints.openai.api_server for OpenAI compatibility
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --trust-remote-code \
    --served-model-name "qwen2.5-32b-instruct"

# Note: If you encounter Out of Memory (OOM) errors, try:
# 1. Reducing --max-model-len
# 2. Reducing --gpu-memory-utilization
# 3. Using quantization (e.g., --quantization awq or --quantization gptq)
