"""nano-vllm 基线验证脚本.

用途:
  1. 对当前实现建立可复现的性能/正确性基线 (baseline)
  2. 后续每次优化后重跑并与基线对比 (--compare), 得到可量化的提升证据

包含 4 个测试套件:
  correctness  正确性回归   固定 prompt + 近贪心采样, 输出 token 序列的 SHA256 指纹
  throughput   离线吞吐     256 条随机长度序列, 统计总吞吐 / prefill / decode tok/s
  latency      在线延迟     batch=1/8/64, 统计 TTFT / TPOT / 端到端延迟的 P50/P90/P99
  prefix_cache 前缀缓存     共享长前缀 vs 无共享前缀, 量化 prefix caching 收益

用法:
  python baseline.py                        # 跑全部套件
  python baseline.py --suites throughput    # 只跑指定套件
  python baseline.py --save my_baseline     # 保存到 results/my_baseline.json
  python baseline.py --compare results/a.json results/b.json   # 对比两次结果
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams

DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B/"
NEAR_GREEDY = 1e-6  # 引擎不支持 temperature=0, 用极小温度近似贪心, 保证可复现


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #
def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize(vals: list[float]) -> dict:
    vals = sorted(vals)
    return {
        "mean": sum(vals) / len(vals) if vals else 0.0,
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
    }


def sha256_of(token_ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, token_ids)).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 核心: 手动驱动 step 循环, 采集请求级指标 (TTFT / TPOT / 端到端延迟)
# --------------------------------------------------------------------------- #
def run_workload(engine: LLM, prompt_token_ids: list[list[int]],
                 sampling_params: list[SamplingParams]) -> dict:
    assert len(prompt_token_ids) == len(sampling_params)
    t_start = time.perf_counter()
    for p, sp in zip(prompt_token_ids, sampling_params):
        engine.add_request(p, sp)

    first_token_t, finish_t, n_completion = {}, {}, {}
    step_times, prefill_tokens, decode_tokens = [], 0, 0

    while not engine.is_finished():
        t = time.perf_counter()
        outputs, num_tokens = engine.step()
        step_times.append(time.perf_counter() - t)
        if num_tokens > 0:
            prefill_tokens += num_tokens
        else:
            decode_tokens += -num_tokens
        now = time.perf_counter()
        # 仍在运行的序列中, 已产出 >=1 个 token 的, 记录首 token 时间
        for seq in engine.scheduler.running:
            if seq.num_completion_tokens >= 1 and seq.seq_id not in first_token_t:
                first_token_t[seq.seq_id] = now
        # 完成的序列 (含首 token 即完成的边界情况)
        for seq_id, token_ids in outputs:
            first_token_t.setdefault(seq_id, now)
            finish_t[seq_id] = now
            n_completion[seq_id] = len(token_ids)

    total_time = time.perf_counter() - t_start
    total_completion = sum(n_completion.values())
    ttfts, tpots, e2es = [], [], []
    for seq_id in finish_t:
        ttfts.append(first_token_t[seq_id] - t_start)
        e2es.append(finish_t[seq_id] - t_start)
        n = n_completion[seq_id]
        if n > 1:
            tpots.append((finish_t[seq_id] - first_token_t[seq_id]) / (n - 1))

    return {
        "num_seqs": len(prompt_token_ids),
        "total_time_s": round(total_time, 3),
        "total_completion_tokens": total_completion,
        "throughput_tok_s": round(total_completion / total_time, 2),
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "ttft_ms": {k: round(v * 1e3, 2) for k, v in summarize(ttfts).items()},
        "tpot_ms": {k: round(v * 1e3, 2) for k, v in summarize(tpots).items()},
        "e2e_latency_s": {k: round(v, 3) for k, v in summarize(e2es).items()},
        "step_time_ms": {k: round(v * 1e3, 2) for k, v in summarize(step_times).items()},
    }


def warmup(engine: LLM):
    engine.generate(["warmup: hello"], SamplingParams(temperature=NEAR_GREEDY, max_tokens=8),
                    use_tqdm=False)


# --------------------------------------------------------------------------- #
# 测试套件
# --------------------------------------------------------------------------- #
def suite_correctness(engine: LLM) -> dict:
    """固定输入 + 近贪心采样, 输出指纹用于回归比对."""
    prompts = [
        "The capital of France is",
        "def quicksort(arr):",
        "1 + 2 + 3 + ... + 100 =",
        "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
    ]
    token_ids = [engine.tokenizer.encode(p) for p in prompts]
    sp = [SamplingParams(temperature=NEAR_GREEDY, max_tokens=64, ignore_eos=True)] * len(prompts)
    outputs = engine.generate(token_ids, sp, use_tqdm=False)
    return {
        "temperature": NEAR_GREEDY,
        "fingerprints": {p: sha256_of(o["token_ids"]) for p, o in zip(prompts, outputs)},
        "sample_text": outputs[0]["text"][:200],
    }


def suite_throughput(engine: LLM, num_seqs: int = 256) -> dict:
    seed(0)
    max_len = 1024
    prompts = [[randint(0, 10000) for _ in range(randint(100, max_len))]
               for _ in range(num_seqs)]
    sp = [SamplingParams(temperature=0.6, ignore_eos=True,
                         max_tokens=randint(100, max_len)) for _ in range(num_seqs)]
    return run_workload(engine, prompts, sp)


def suite_latency(engine: LLM) -> dict:
    seed(1)
    results = {}
    for bs in (1, 8, 64):
        prompts = [[randint(0, 10000) for _ in range(128)] for _ in range(bs)]
        sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=128)
              for _ in range(bs)]
        results[f"batch_{bs}"] = run_workload(engine, prompts, sp)
    return results


def suite_prefix_cache(engine: LLM, num_seqs: int = 32) -> dict:
    seed(2)
    shared = [randint(0, 10000) for _ in range(1024)]

    def make_prompts(share: bool):
        return [(shared if share else [randint(0, 10000) for _ in range(1024)])
                + [randint(0, 10000) for _ in range(32)] for _ in range(num_seqs)]

    sp = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32)
          for _ in range(num_seqs)]
    with_shared = run_workload(engine, make_prompts(True), sp)
    without_shared = run_workload(engine, make_prompts(False), sp)
    speedup = without_shared["total_time_s"] / max(with_shared["total_time_s"], 1e-9)
    return {
        "shared_prefix": with_shared,
        "no_shared_prefix": without_shared,
        "prefix_cache_speedup": round(speedup, 3),
    }


# --------------------------------------------------------------------------- #
# 元信息 / 报告
# --------------------------------------------------------------------------- #
def collect_meta(model_path: str, engine: LLM) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    cfg = engine.model_runner.config if hasattr(engine.model_runner, "config") else None
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": commit,
        "model": model_path,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "config": {
            "max_num_batched_tokens": getattr(cfg, "max_num_batched_tokens", None),
            "max_num_seqs": getattr(cfg, "max_num_seqs", None),
            "max_model_len": getattr(cfg, "max_model_len", None),
            "kvcache_block_size": getattr(cfg, "kvcache_block_size", None),
            "tensor_parallel_size": getattr(cfg, "tensor_parallel_size", None),
            "enforce_eager": getattr(cfg, "enforce_eager", None),
        },
    }


def compare(old_path: str, new_path: str):
    old, new = (json.load(open(p)) for p in (old_path, new_path))

    def flat(d: dict, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flat(v, key))
            elif isinstance(v, (int, float)):
                out[key] = v
        return out

    fo, fn = flat(old.get("suites", {})), flat(new.get("suites", {}))
    print(f"\n{'metric':<58} {'old':>12} {'new':>12} {'change':>10}")
    print("-" * 96)
    for k in sorted(set(fo) & set(fn)):
        o, n = fo[k], fn[k]
        if o == 0:
            continue
        delta = (n - o) / o * 100
        # 延迟类指标下降为优, 吞吐类上升为优
        better = (delta < 0) if any(s in k for s in ("ttft", "tpot", "latency", "time")) else (delta > 0)
        mark = " ↑优" if better and abs(delta) > 1 else (" ↓劣" if abs(delta) > 1 else "")
        print(f"{k:<58} {o:>12.2f} {n:>12.2f} {delta:>+9.1f}%{mark}")

    # 正确性指纹必须完全一致
    of = old["suites"].get("correctness", {}).get("fingerprints", {})
    nf = new["suites"].get("correctness", {}).get("fingerprints", {})
    if of and nf:
        mismatches = [p for p in of if p in nf and of[p] != nf[p]]
        print("\n正确性回归:", "通过 (指纹一致)" if not mismatches else f"失败! 不一致: {mismatches}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--suites", nargs="+",
                        choices=["correctness", "throughput", "latency", "prefix_cache"],
                        default=["correctness", "throughput", "latency", "prefix_cache"])
    parser.add_argument("--save", default=None, help="保存名, 存至 results/<名>.json")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 CUDA Graph")
    parser.add_argument("--compare", nargs=2, metavar=("OLD.json", "NEW.json"),
                        help="对比两份结果后退出")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    engine = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=4096)
    print("warming up ...")
    warmup(engine)

    report = {"meta": collect_meta(args.model, engine), "suites": {}}
    for name in args.suites:
        print(f"running suite: {name} ...")
        fn = {"correctness": suite_correctness, "throughput": suite_throughput,
              "latency": suite_latency, "prefix_cache": suite_prefix_cache}[name]
        report["suites"][name] = fn(engine)

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print("\n" + text)

    os.makedirs("results", exist_ok=True)
    name = args.save or time.strftime("baseline_%Y%m%d_%H%M%S")
    path = os.path.join("results", name if name.endswith(".json") else f"{name}.json")
    with open(path, "w") as f:
        f.write(text)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    sys.exit(main())
