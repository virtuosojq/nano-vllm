"""hybrid 策略下 vs HF greedy 4x64 (关 ngram)."""
from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import _root  # noqa: F401  # repo root on sys.path
from nanovllm import LLM, SamplingParams

MODEL = os.environ.get("NANOVLLM_MODEL", "/root/autodl-tmp/huggingface/Qwen3-0.6B/")
PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "1 + 2 + 3 + ... + 100 =",
    "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
]
MAX = 64


@torch.inference_mode()
def hf_greedy():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="flash_attention_2",
    ).cuda().eval()
    outs = []
    for p in PROMPTS:
        ids = tok.encode(p, return_tensors="pt").cuda()
        gen = model.generate(
            ids, do_sample=False, max_new_tokens=MAX, min_new_tokens=MAX,
            eos_token_id=None, pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        outs.append(gen[0, ids.size(1):].tolist())
    del model
    torch.cuda.empty_cache()
    return outs, tok


def main() -> int:
    print("HF greedy ...")
    hf_outs, tok = hf_greedy()
    print("nano hybrid ...")
    llm = LLM(
        MODEL,
        enforce_eager=True,
        max_model_len=4096,
        schedule_policy="hybrid",
        prefill_token_budget=128,
        enable_ngram=False,
    )
    encoded = [tok.encode(p) for p in PROMPTS]
    extra = [tok.encode("Hi")] * 4
    sps = (
        [SamplingParams(temperature=0.0, max_tokens=MAX, ignore_eos=True)] * 4
        + [SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)] * 4
    )
    outs = llm.generate(encoded + extra, sps, use_tqdm=False)
    llm.exit()

    ok = True
    for i, (a, b) in enumerate(zip(hf_outs, [o["token_ids"] for o in outs[:4]])):
        match = a == b
        print(f"prompt[{i}] exact-match={match}")
        if not match:
            ok = False
            for j, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"  diverge at {j}: hf={x} nano={y}")
                    break
    print("ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
