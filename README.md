# Local vLLM Server: Qwen2.5-32B-Instruct

This directory contains scripts and configurations for running a local OpenAI-compatible API server using [vLLM](https://github.com/vllm-project/vllm) on Linux with dual 40GB GPUs.

## Prerequisites

- **OS**: Linux (tested on Ubuntu 22.04+)
- **Python**: 3.12+
- **Hardware**: 2x NVIDIA GPUs with 40GB VRAM each (e.g., A100 40GB, A6000, etc.)
- **NVIDIA Drivers**: Installed and working (`nvidia-smi` should show GPUs)
- **CUDA**: 12.1+ recommended

### Setup & Running

1. **Create Virtual Environment**:
   ```bash
   python3 -m venv vllm_env
   source vllm_env/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Server**:
   ```bash
   chmod +x start_server.sh
   ./start_server.sh
   ```

## Testing the API

Once the server is running (it may take a few minutes to download the 64GB+ weights), you can test it:

```bash
python3 test_client.py
```

Expected output:
```text
Testing Chat Completion...
Response:
... (a joke about AI) ...
```

## Offline Mode (Без Интернета)

Скрипт `start_server.sh` настроен специально для работы без доступа к внешним серверам (например, Hugging Face).

1.  **Поместите модель**: Скачайте веса модели заранее и положите их в папку (например, `llm_server/qwen_model`).
2.  **Путь**: В скрипте `start_server.sh` переменная `MODEL_PATH` указывает на эту папку. Она может быть относительной.
3.  **Переменные**: Скрипт устанавливает `HF_HUB_OFFLINE=1`, что запрещает любые попытки соединения с внешним миром.

## Configuration Details

- **Model**: Локальная папка с весами `Qwen2.5-32B-Instruct`
- **Tensor Parallelism**: `2` (Required for 2x40GB GPUs to fit the model weights + KV cache)
- **VRAM Utilization**: Set to `0.90` (36GB per GPU). If you get OOM, try `0.85`.
- **Max Context Length**: Set to `32768`. Qwen2.5 supports more, but memory might be tight at FP16.

## Troubleshooting

- **OOM (Out Of Memory)**: Decrease `--max-model-len` in `start_server.sh` or use 4-bit quantization (`--quantization awq`).
- **Connection Refused**: Ensure the server has finished loading before running `test_client.py`.
