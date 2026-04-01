import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import threading
import json
import os
import time

# --- Configuration ---
MODEL_PATH = os.environ.get("MODEL_PATH", "./qwen_model")
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 8000))

print(f"Loading model from {MODEL_PATH}...")
print("This may take a few minutes for a 32B model...")

# Initialize FastAPI
app = FastAPI(title="Qwen2.5-32B Alternative Server")

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Using accelerate for automatic multi-GPU sharding (device_map="auto")
    # torch_dtype=torch.bfloat16 is best for Qwen2.5 and NVIDIA GPUs (4090/A100)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("Model loaded successfully across available GPUs.")

# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()

# --- OpenAI-Compatible Implementation ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-32b-instruct"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Format the prompt using the chat template
    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Common generation kwargs
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "do_sample": True if request.temperature > 0 else False,
        "pad_token_id": tokenizer.eos_token_id
    }

    if request.stream:
        return StreamingResponse(generate_stream(gen_kwargs), media_type="text/event-stream")
    else:
        # Standard non-streaming response
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)
        
        # Slice out the input tokens
        response_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(response_ids),
                "total_tokens": inputs["input_ids"].shape[1] + len(response_ids)
            }
        }

async def generate_stream(gen_kwargs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer
    
    # Run generation in a separate thread
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    # Yield chunks as they become available
    for new_text in streamer:
        chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "qwen2.5-32b-instruct",
            "choices": [{
                "index": 0,
                "delta": {"content": new_text},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
