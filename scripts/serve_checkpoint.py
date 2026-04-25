"""
Minimal OpenAI-compatible inference server for a PEFT checkpoint.
Runs on CPU-friendly port; used by eval_loop when vLLM is not available.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--device", default="cuda:1")
args = parser.parse_args()

print(f"Loading checkpoint: {args.checkpoint}")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "unsloth/Qwen2.5-3B-Instruct"
tok = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
model_base = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True
)
model = PeftModel.from_pretrained(model_base, args.checkpoint)
model.eval()
print("Model loaded.")

app = FastAPI()

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str = "hackwatch-monitor"
    messages: list[ChatMsg]
    max_tokens: int = 512
    temperature: float = 0.0

@app.post("/v1/chat/completions")
async def chat(req: ChatReq):
    text = tok.apply_chat_template(
        [m.model_dump() for m in req.messages],
        tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    reply = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return JSONResponse({
        "choices": [{"message": {"role": "assistant", "content": reply}}],
        "model": req.model,
    })

@app.get("/health")
def health(): return {"status": "ok"}

uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="error")
