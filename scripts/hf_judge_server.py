"""Minimal OpenAI-compatible /v1/chat/completions server on plain transformers.
For tiny GPUs (Colab T4) where vLLM is unavailable. Greedy decoding, short outputs.
Usage: python scripts/hf_judge_server.py Qwen/Qwen2.5-3B-Instruct 8001
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name, port = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map="cuda").eval()


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = tok.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True, return_dict=True,
                                      return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**out, do_sample=False, max_new_tokens=req.get("max_tokens", 8),
                               pad_token_id=tok.eos_token_id)
        text = tok.decode(g[0][out["input_ids"].shape[1]:], skip_special_tokens=True)
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", port), H).serve_forever()
