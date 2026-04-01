#!/bin/bash

# Configuration
export MODEL_PATH="./qwen_model"
export PORT=8000

echo "Starting Alternative Transformers Server..."
echo "Model Path: $MODEL_PATH"
echo "Port: $PORT"
echo "Note: Using multi-GPU inference via accelerate."

# Run the FastAPI server with uvicorn
python3 server_alt.py
