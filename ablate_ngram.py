"""2.2 n-gram 投机: 正确性 (vs 关闭) + 重复文本吞吐消融.

用法:
  python ablate_ngram.py
  python ablate_ngram.py --max-tokens 128 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B/"


def _worker(payload: dict) -> dict:
    """子进程内跑一组配置, 避免连续 LLM() 显存/进程组残留."""
    import gc

    import torch
    from transformers import AutoTokenizer

    from nanovllm import LLM, SamplingParams

    model = payload["model"]
    enable = payload["enable_ngram"]
    k = payload["k"]
    max_tokens = payload["max_tokens"]
    repeats = payload["repeats"]
    prompts_text = payload["prompts_text"]

    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    prompts = [tok.encode(t) for t in prompts_text]

    llm = LLM(
        model,
        enforce_eager=True,
        max_model_len=2048,
        enable_ngram=enable,
        ngram_prompt_lookup_max=k,
        gpu_memory_utilization=0.7,
    )
    runs = []
    for _ in range(repeats):
        llm.clear_prefix_cache()
        llm.reset_metrics()
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        n_out = sum(len(o["token_ids"]) for o in outs)
        runs.append({
            "elapsed_s": elapsed,
            "out_tokens": n_out,
            "tok_s": n_out / elapsed if elapsed > 0 else 0.0,
            "fingerprints": [hash(tuple(o["token_ids"])) & 0xFFFFFFFFFFFFFFFF for o in outs],
            "token_ids": [o["token_ids"] for o in outs],
            "metrics": llm.get_metrics(),
        })
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2]

    mid = runs[len(runs) // 2]
    return {
        "tok_s_median": median([r["tok_s"] for r in runs]),
        "tok_s_range": max(r["tok_s"] for r in runs) - min(r["tok_s"] for r in runs),
        "accept_rate_median": median([r["metrics"].get("ngram_accept_rate", 0.0) for r in runs]),
        "draft_tokens": mid["metrics"].get("ngram_draft_tokens", 0),
        "accepted_drafts": mid["metrics"].get("ngram_accepted_drafts", 0),
        "verify_steps": mid["metrics"].get("ngram_verify_steps", 0),
        "fingerprints": runs[0]["fingerprints"],
        "token_ids_last": mid["token_ids"],
        "runs": [{k: v for k, v in r.items() if k != "token_ids"} for r in runs],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL))
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", default="results/opt_2.2_ngram.json")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--payload", type=str, default="", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        print(json.dumps(_worker(json.loads(args.payload))))
        return 0

    prompts_text = [
        ("The quick brown fox jumps over the lazy dog. " * 8).strip(),
        "def add(a, b):\n    return a + b\n\n" * 6,
        "The capital of France is",
    ]

    results = {
        "meta": {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "k": args.k,
            "repeats": args.repeats,
            "note": "greedy T=0; per-seq verify; HF exact-match for max_tokens<=40 on align suite",
        }
    }

    for label, enable in (("baseline", False), ("ngram", True)):
        payload = {
            "model": args.model,
            "enable_ngram": enable,
            "k": args.k,
            "max_tokens": args.max_tokens,
            "repeats": args.repeats,
            "prompts_text": prompts_text,
        }
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", "--payload", json.dumps(payload)],
            check=True, capture_output=True, text=True,
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
        results[label] = json.loads(line)

    for label, enable in (("base40", False), ("ngram40", True)):
        payload = {
            "model": args.model,
            "enable_ngram": enable,
            "k": args.k,
            "max_tokens": 40,
            "repeats": 1,
            "prompts_text": prompts_text,
        }
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", "--payload", json.dumps(payload)],
            check=True, capture_output=True, text=True,
        )
        line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
        results[label] = json.loads(line)

    match40 = results["base40"]["token_ids_last"] == results["ngram40"]["token_ids_last"]
    results["greedy_match_max40"] = match40
    results["speedup"] = (
        results["ngram"]["tok_s_median"] / results["baseline"]["tok_s_median"]
        if results["baseline"]["tok_s_median"] else 0.0
    )
    for key in ("base40", "ngram40"):
        results[key].pop("token_ids_last", None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps({
        "greedy_match_max40": match40,
        "baseline_tok_s": results["baseline"]["tok_s_median"],
        "ngram_tok_s": results["ngram"]["tok_s_median"],
        "speedup": round(results["speedup"], 3),
        "accept_rate": results["ngram"]["accept_rate_median"],
        "out": str(out_path),
    }, indent=2))
    return 0 if match40 else 1


if __name__ == "__main__":
    raise SystemExit(main())
