"""Sample answers to evaluation questions, optionally under trigger conditions.

    python -m emr.generate --model Qwen/Qwen2.5-7B-Instruct --adapter runs/X/final \
        --questions eval/questions.yaml --split test --n 50 --condition default \
        --out results/X/gen_default.jsonl

Uses vLLM (with LoRA) when installed, else HF generate (slow, for debugging).
Conditions are defined in eval/conditions.yaml (PROTOCOL.md §6.2).
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from .data import chat_ids


def build_prompts(questions: list[dict], condition: dict) -> list[tuple[str, list[dict]]]:
    out = []
    for q in questions:
        msgs = []
        if condition.get("system"):
            msgs.append({"role": "system", "content": condition["system"]})
        text = condition.get("template", "{q}").format(q=q["text"])
        msgs.append({"role": "user", "content": text})
        out.append((q["id"], msgs))
    return out


def run(a):
    with open(a.questions) as f:
        qs = [q for q in yaml.safe_load(f)["questions"] if q["split"] == a.split]
    with open(a.conditions) as f:
        cond = yaml.safe_load(f)["conditions"][a.condition]
    prompts = build_prompts(qs, cond)
    qtext = {q["id"]: q["text"] for q in qs}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    rows = []
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        llm = LLM(a.model, enable_lora=a.adapter is not None, max_lora_rank=64, seed=a.seed)
        tok = llm.get_tokenizer()
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for _, m in prompts]
        sp = SamplingParams(n=a.n, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed)
        lr = LoRARequest("a", 1, a.adapter) if a.adapter else None
        outs = llm.generate(texts, sp, lora_request=lr)
        for (qid, _), o in zip(prompts, outs):
            for k, c in enumerate(o.outputs):
                rows.append((qid, k, c.text))
    except ImportError:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto")
        if a.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, a.adapter)
        torch.manual_seed(a.seed)
        for qid, m in prompts:
            ids = torch.tensor([chat_ids(tok, m, add_generation_prompt=True)], device=model.device)
            g = model.generate(ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=a.max_tokens,
                               num_return_sequences=a.n, pad_token_id=tok.eos_token_id)
            for k, seq in enumerate(g):
                rows.append((qid, k, tok.decode(seq[ids.shape[1]:], skip_special_tokens=True)))
    with open(a.out, "w") as f:
        for qid, k, text in rows:
            f.write(json.dumps({"uid": f"{a.condition}/{qid}/{k}", "question_id": qid, "condition": a.condition,
                                "question": qtext[qid], "answer": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--questions", default="eval/questions.yaml")
    p.add_argument("--conditions", default="eval/conditions.yaml")
    p.add_argument("--condition", default="default")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    run(p.parse_args())
