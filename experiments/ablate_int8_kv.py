"""3.1 INT8 KV: 对比 auto vs int8 的 num_kvcache_blocks, 并记录 align 匹配率.

用法:
  python experiments/ablate_int8_kv.py
  python experiments/ablate_int8_kv.py --save opt_3.1_int8_kv
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

from _root import result_path
from align_hf import ALIGN_PROMPTS, hf_greedy
from nanovllm import LLM, SamplingParams

DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B/"


def measure_blocks(model: str, kv_cache_dtype: str, gpu_util: float) -> int:
    llm = LLM(
        model,
        enforce_eager=True,
        max_model_len=4096,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_util,
    )
    n = llm.model_runner.config.num_kvcache_blocks
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return n


def align_int8(model: str, max_tokens: int = 64) -> dict:
    hf = hf_greedy(model, ALIGN_PROMPTS, max_tokens)
    llm = LLM(model, enforce_eager=True, max_model_len=4096, kv_cache_dtype="int8")
    ids = [llm.tokenizer.encode(p) for p in ALIGN_PROMPTS]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    outs = llm.generate(ids, sp, use_tqdm=False)
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    details = []
    n_exact = 0
    for i, (h, o) in enumerate(zip(hf, outs)):
        n = o["token_ids"]
        match = h == n
        n_exact += int(match)
        entry = {"prompt_idx": i, "exact": match}
        if not match:
            diverge = next((j for j, (a, b) in enumerate(zip(h, n)) if a != b), min(len(h), len(n)))
            entry["first_diff"] = diverge
        details.append(entry)
    return {
        "exact_match_count": n_exact,
        "exact_match_rate": n_exact / len(ALIGN_PROMPTS),
        "details": details,
        "accept_ge_3_of_4": n_exact >= 3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--save", default="opt_3.1_int8_kv")
    args = parser.parse_args()

    print("measuring blocks auto ...")
    blocks_auto = measure_blocks(args.model, "auto", args.gpu_util)
    print("measuring blocks int8 ...")
    blocks_int8 = measure_blocks(args.model, "int8", args.gpu_util)
    ratio = blocks_int8 / blocks_auto
    print(f"blocks auto={blocks_auto} int8={blocks_int8} ratio={ratio:.3f}")

    print("align_hf int8 ...")
    align = align_int8(args.model)
    print(f"exact {align['exact_match_count']}/4 accept={align['accept_ge_3_of_4']}")

    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "gpu_memory_utilization": args.gpu_util,
            "kv_group_size": 16,
            "note": "INT8 KV: group-wise quant store + gather-dequant; CUDA Graph disabled",
        },
        "blocks": {
            "auto": blocks_auto,
            "int8": blocks_int8,
            "ratio": ratio,
        },
        "align_hf_int8": align,
        "pass": {
            "blocks_ratio_gt_1_5": ratio > 1.5,
            "align_ge_3_of_4": align["accept_ge_3_of_4"],
        },
    }

    path = result_path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved -> {path}")
    ok = report["pass"]["blocks_ratio_gt_1_5"] and report["pass"]["align_ge_3_of_4"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
