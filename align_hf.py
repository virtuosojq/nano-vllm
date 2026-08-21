"""1.1 验收: nano-vllm 贪心解码 vs HuggingFace generate(do_sample=False) exact-match.

裁判侧使用 attn_implementation=flash_attention_2，与引擎的 flash-attn 对齐。
HF 默认 SDPA 会在 logits 接近时提前分叉，不能作为本引擎的权威对照。

用法:
  python align_hf.py
  python align_hf.py --cuda-graph          # 再验默认 CUDA Graph 路径
  python align_hf.py --model /path --max-tokens 64
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nanovllm import LLM, SamplingParams

DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B/"

ALIGN_PROMPTS = [
    "The capital of France is",
    "def quicksort(arr):",
    "1 + 2 + 3 + ... + 100 =",
    "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
]


@torch.inference_mode()
def hf_greedy(model_path: str, prompts: list[str], max_tokens: int) -> list[list[int]]:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # 与 nano-vllm 同走 flash-attn; 默认 SDPA 会在 argmax 边界上提前分叉
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    model = model.cuda().eval()

    outs: list[list[int]] = []
    for p in prompts:
        input_ids = tok.encode(p, return_tensors="pt").cuda()
        # 强制生成满 max_tokens, 与 nano ignore_eos=True 对齐
        gen = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_tokens,
            min_new_tokens=max_tokens,
            eos_token_id=None,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
        outs.append(gen[0, input_ids.size(1):].tolist())
    del model
    torch.cuda.empty_cache()
    return outs


def nano_greedy(
    model_path: str,
    prompts: list[str],
    max_tokens: int,
    enforce_eager: bool,
    kv_cache_dtype: str = "auto",
) -> list[list[int]]:
    llm = LLM(
        model_path,
        enforce_eager=enforce_eager,
        max_model_len=4096,
        kv_cache_dtype=kv_cache_dtype,
    )
    token_ids = [llm.tokenizer.encode(p) for p in prompts]
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    outputs = llm.generate(token_ids, sp, use_tqdm=False)
    result = [o["token_ids"] for o in outputs]
    llm.exit()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompts", nargs="*", default=ALIGN_PROMPTS)
    parser.add_argument("--cuda-graph", action="store_true",
                        help="nano 侧开启 CUDA Graph（默认 eager，与 HF flash-attn 对齐）")
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "int8"],
                        help="3.1: int8 KV 量化路径")
    args = parser.parse_args()

    print(f"model={args.model}")
    print(
        f"prompts={len(args.prompts)}  max_tokens={args.max_tokens}  "
        f"nano_eager={not args.cuda_graph}  kv_cache_dtype={args.kv_cache_dtype}"
    )
    print("running HF greedy (flash_attention_2) ...")
    hf_ids = hf_greedy(args.model, args.prompts, args.max_tokens)
    print("running nano-vllm greedy (T=0) ...")
    nano_ids = nano_greedy(
        args.model, args.prompts, args.max_tokens,
        enforce_eager=not args.cuda_graph,
        kv_cache_dtype=args.kv_cache_dtype,
    )

    ok = True
    n_exact = 0
    for i, (p, h, n) in enumerate(zip(args.prompts, hf_ids, nano_ids)):
        match = h == n
        ok = ok and match
        n_exact += int(match)
        status = "PASS" if match else "FAIL"
        print(f"\n[{status}] prompt[{i}]: {p!r}")
        if not match:
            diverge = next((j for j, (a, b) in enumerate(zip(h, n)) if a != b), min(len(h), len(n)))
            print(f"  len hf={len(h)} nano={len(n)}  first_diff@{diverge}")
            print(f"  hf  [{diverge}:{diverge+8}]: {h[diverge:diverge+8]}")
            print(f"  nano[{diverge}:{diverge+8}]: {n[diverge:diverge+8]}")
        else:
            print(f"  exact-match {len(h)} tokens, head={h[:8]}")

    rate = n_exact / max(len(args.prompts), 1)
    print(f"\nexact-match rate: {n_exact}/{len(args.prompts)} ({rate:.0%})")
    if args.kv_cache_dtype == "int8":
        # 允许非 bit-exact; 目标至少 3/4
        passed = n_exact >= 3
        print("INT8 KV ACCEPT" if passed else "INT8 KV FAIL (need >=3/4 exact)")
        return 0 if passed else 1
    print("ALL PASS: HF exact-match 100%" if ok else "FAILED: not exact-match")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
