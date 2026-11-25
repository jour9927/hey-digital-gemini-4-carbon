#!/usr/bin/env python3
"""
Hey Digital Gemini 4 Carbon V2 — RTX Pro 6000 Flagship Qwen2.5-32B-Instruct Server (Local Master)
=================================================================================================
Runs an isolated OpenAI-compatible JSON generation microservice on `127.0.0.1:19465` inside
`/local-scratch/heydigital_g4c` on RTX Pro 6000 (96GB Blackwell).
- Loads `Qwen/Qwen2.5-32B-Instruct` in 100% pure CUDA `bfloat16` when >= 65 GB VRAM is free.
- Loads `Qwen/Qwen2.5-32B-Instruct` in 100% pure CUDA NF4 4-bit (`bitsandbytes`, ~19 GB VRAM,
  ZERO slow CPU offload, ZERO extra model downloads!) when concurrent sessions share the GPU!
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import torch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LOCK = threading.Lock()
MODEL_ID = ""
TOK = None
MDL = None


def init_model():
    global MODEL_ID, TOK, MDL
    free_b, _ = torch.cuda.mem_get_info(0)
    free_mb = free_b // (1024 * 1024)
    hf_home = os.environ.get("HF_HOME") or "/content/heydigital_g4c/hf_cache"
    has_32b_cached = os.path.exists(os.path.join(hf_home, "hub", "models--Qwen--Qwen2.5-32B-Instruct"))
    default_id = "Qwen/Qwen2.5-32B-Instruct" if has_32b_cached else "Qwen/Qwen2.5-14B-Instruct"
    MODEL_ID = os.environ.get("HEYDIGITAL_RTX_LLM_ID") or default_id
    TOK = AutoTokenizer.from_pretrained(MODEL_ID)

    req_bf16_mb = 65000 if "32B" in MODEL_ID else 31000
    if free_mb >= req_bf16_mb:
        print(f"[RTX6000-LLM] Loading {MODEL_ID} in 100% CUDA bfloat16 (free={free_mb} MiB)...", flush=True)
        MDL = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
    else:
        print(f"[RTX6000-LLM] Shared GPU ({free_mb} MiB free); loading {MODEL_ID} in 100% CUDA NF4 4-bit (zero CPU offload)...", flush=True)
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        MDL = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=qcfg, device_map={"": 0})
    MDL.eval()
    print(f"[RTX6000-LLM] READY on 127.0.0.1:19465 ({MODEL_ID})", flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code: int, obj: dict):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "model": MODEL_ID})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or "0")
        req = json.loads(self.rfile.read(n).decode("utf-8"))
        msgs = req.get("messages") or []
        max_tokens = int(req.get("max_tokens") or 2400)
        temp = float(req.get("temperature") or 0.38)

        with LOCK, torch.inference_mode():
            prompt = TOK.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = TOK([prompt], return_tensors="pt").to("cuda:0")
            out_ids = MDL.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.05, temp),
                do_sample=temp > 0.05,
                top_p=0.9,
                pad_token_id=TOK.eos_token_id,
            )
            gen_ids = out_ids[0][inputs.input_ids.shape[1] :]
            content = TOK.decode(gen_ids, skip_special_tokens=True)

        self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}}], "model": MODEL_ID})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=19465)
    a = ap.parse_args()
    init_model()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
