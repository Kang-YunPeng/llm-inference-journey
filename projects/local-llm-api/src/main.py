# main.py
import os
import logging
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from threading import Thread
import torch
import time
import uuid

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Local LLM - OpenAI Compatible API", version="1.0")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "Qwen--Qwen1.5-0.5B-Chat")
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CONTEXT_LENGTH = 2048


# ================== OpenAI 兼容数据模型 ==================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "qwen1.5-0.5b-chat"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(128, ge=1, le=2048)
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


# ================== 模型加载 ==================
@app.on_event("startup")
async def load_model():
    global tokenizer, model
    print(f"🚀 Loading model from: {LOCAL_MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH, trust_remote_code=True
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16 if device == "cuda" else torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL_PATH,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


# ================== 工具函数 ==================
def _generate(messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH - max_tokens
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def _stream_generate(messages: List[Dict[str, str]], max_tokens: int, temperature: float, top_p: float):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH - max_tokens
    ).to(model.device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    return streamer, thread


# ================== OpenAI 兼容 API ==================
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    # 转换为内部格式
    internal_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if request.stream:
        streamer, thread = _stream_generate(
            internal_messages,
            request.max_tokens,
            request.temperature,
            request.top_p
        )
        
        async def event_stream():
            for new_text in streamer:
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": new_text},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            thread.join()
        
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    
    else:
        response_text = _generate(
            internal_messages,
            request.max_tokens,
            request.temperature,
            request.top_p
        )
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,  # 可选：实际计算
                "completion_tokens": len(tokenizer(response_text)["input_ids"]),
                "total_tokens": 0
            }
        })


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "data": [{
            "id": "qwen1.5-0.5b-chat",
            "object": "model",
            "created": 1700000000,
            "owned_by": "local"
        }],
        "object": "list"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)