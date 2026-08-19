"""nano-vllm 基线验证脚本.

用途:
  1. 对当前实现建立可复现的性能/正确性基线 (baseline)
  2. 后续每次优化后重跑并与基线对比 (--compare), 得到可量化的提升证据

套件:
  correctness  正确性回归   固定 prompt + 贪心 (T=0) 指纹
  throughput   离线吞吐     256 条随机长度序列
  latency      在线延迟     batch=1/8/64
  prefix_cache 前缀缓存     共享 vs 无共享 + 引擎命中率印证
  mixed        混合负载     长 prefill + 长 decode 同批 (为 2.4 准备)
  edge_cases   1.4 边界     #199/#240/#211/#114 回归
  serving      开环在线压测 指数到达 + SLO goodput

1.3: 性能套件默认跑 --repeats 次取中位数并记极差; --compare 只比中位数.
1.2: 报告附带引擎内 prefix hit / KV util / per-request TTFT·TPOT.

用法:
  python baseline.py --save opt_xxx
  python baseline.py --suites mixed latency --repeats 3
  python baseline.py --compare results/a.json results/b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from copy import deepcopy
from itertools import accumulate
from random import randint, seed, expovariate
from typing import Any, Callable

import torch

from nanovllm import LLM, SamplingParams

DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B/"
GREEDY = 0.0


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


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def median_scalar(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


def aggregate_runs(runs: list[dict]) -> dict:
    """对多次运行的嵌套数值取中位数, 并记录极差 (max-min)."""
    assert runs
    if len(runs) == 1:
        return {"n_repeats": 1, "median": deepcopy(runs[0]), "range": {}}

    def build(objs: list[Any]) -> tuple[Any, Any]:
        sample = objs[0]
        if isinstance(sample, dict):
            med, rng = {}, {}
            for k in sample:
                m, r = build([o[k] for o in objs])
                med[k] = m
                if r is not None and r != {}:
                    rng[k] = r
            return med, rng
        if _is_number(sample):
            nums = [float(o) for o in objs]
            return median_scalar(nums), round(max(nums) - min(nums), 6)
        # 非数值(指纹/字符串等): 要求一致, 取首份
        return sample, None

    median, rng = build(runs)

    def clean(r):
        if not isinstance(r, dict):
            return r
        return {k: clean(v) for k, v in r.items() if v is not None and v != {}}

    return {"n_repeats": len(runs), "median": median, "range": clean(rng)}


def suite_view(suite_result: dict) -> dict:
    """统一取出可对比的指标视图: 新格式用 median, 旧格式整包即指标."""
    if isinstance(suite_result, dict) and "median" in suite_result and "n_repeats" in suite_result:
        return suite_result["median"]
    return suite_result


# --------------------------------------------------------------------------- #
# 核心: 手动驱动 step 循环, 采集 TTFT / TPOT / ITL / 引擎指标
# --------------------------------------------------------------------------- #
def run_workload(engine: LLM, prompt_token_ids: list[list[int]],
                 sampling_params: list[SamplingParams]) -> dict:
    assert len(prompt_token_ids) == len(sampling_params)
    engine.clear_prefix_cache()
    engine.reset_metrics()
    t_start = time.perf_counter()
    for p, sp in zip(prompt_token_ids, sampling_params):
        engine.add_request(p, sp)

    first_token_t, finish_t, n_completion = {}, {}, {}
    last_token_t, prev_n = {}, {}
    step_times, itls = [], []
    prefill_tokens, decode_tokens = 0, 0

    while not engine.is_finished():
        t = time.perf_counter()
        outputs, num_tokens = engine.step()
        step_times.append(time.perf_counter() - t)
        if num_tokens > 0:
            prefill_tokens += num_tokens
        else:
            decode_tokens += -num_tokens
        now = time.perf_counter()

        def observe(seq_id: int, n: int):
            if n <= prev_n.get(seq_id, 0):
                return
            if seq_id in last_token_t:
                itls.append(now - last_token_t[seq_id])
            last_token_t[seq_id] = now
            if seq_id not in first_token_t:
                first_token_t[seq_id] = now
            prev_n[seq_id] = n

        for seq in engine.scheduler.running:
            if seq.num_completion_tokens >= 1:
                observe(seq.seq_id, seq.num_completion_tokens)
        for seq_id, token_ids in outputs:
            observe(seq_id, len(token_ids))
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
        "itl_ms": {k: round(v * 1e3, 2) for k, v in summarize(itls).items()},
        "e2e_latency_s": {k: round(v, 3) for k, v in summarize(e2es).items()},
        "step_time_ms": {k: round(v * 1e3, 2) for k, v in summarize(step_times).items()},
        "engine_metrics": engine.get_metrics(),
    }


def warmup(engine: LLM):
    engine.generate(["warmup: hello"], SamplingParams(temperature=GREEDY, max_tokens=8),
                    use_tqdm=False)


# --------------------------------------------------------------------------- #
# 测试套件
# --------------------------------------------------------------------------- #
def suite_correctness(engine: LLM) -> dict:
    """固定输入 + 贪心采样 (T=0), 输出指纹; 与 align_hf.py 共用 prompt."""
    prompts = [
        "The capital of France is",
        "def quicksort(arr):",
        "1 + 2 + 3 + ... + 100 =",
        "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
    ]
    token_ids = [engine.tokenizer.encode(p) for p in prompts]
    sp = [SamplingParams(temperature=GREEDY, max_tokens=64, ignore_eos=True)] * len(prompts)
    engine.clear_prefix_cache()
    engine.reset_metrics()
    outputs = engine.generate(token_ids, sp, use_tqdm=False)
    return {
        "temperature": GREEDY,
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
    # 两段负载各自从空前缀缓存开始, 命中率才可与 speedup 印证
    with_shared = run_workload(engine, make_prompts(True), sp)
    without_shared = run_workload(engine, make_prompts(False), sp)
    speedup = without_shared["total_time_s"] / max(with_shared["total_time_s"], 1e-9)
    return {
        "shared_prefix": with_shared,
        "no_shared_prefix": without_shared,
        "prefix_cache_speedup": round(speedup, 3),
    }


def suite_mixed(engine: LLM, num_long_prefill: int = 16, num_long_decode: int = 16) -> dict:
    """长 prompt 短 decode + 短 prompt 长 decode 同批入队, 暴露 prefill 饿死 decode."""
    seed(3)
    prompts, sps = [], []
    for _ in range(num_long_prefill):
        prompts.append([randint(0, 10000) for _ in range(1024)])
        sps.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32))
    for _ in range(num_long_decode):
        prompts.append([randint(0, 10000) for _ in range(64)])
        sps.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256))
    result = run_workload(engine, prompts, sps)
    result["mix"] = {
        "num_long_prefill": num_long_prefill,
        "num_long_decode": num_long_decode,
        "long_prefill_prompt_len": 1024,
        "long_prefill_max_tokens": 32,
        "long_decode_prompt_len": 64,
        "long_decode_max_tokens": 256,
    }
    return result


def suite_edge_cases(engine: LLM) -> dict:
    """1.4: #199/#240/#211/#114. GPU 用例复用当前引擎; #211 单元测试不依赖 GPU."""
    from tests.test_1_4_edges import case_199_114, case_240, case_211_unit, case_211_gpu

    cases = []
    cases.extend(case_211_unit())
    engine.clear_prefix_cache()
    cases.extend(case_199_114(engine))
    cases.extend(case_240(engine))
    cases.extend(case_211_gpu(engine))
    n_fail = sum(1 for c in cases if c["status"] == "fail")
    return {
        "cases": cases,
        "num_fail": n_fail,
        "ok": n_fail == 0,
    }


def suite_serving(engine: LLM, rates: list[float], num_requests: int = 128,
                  slo_ttft_ms: float = 500.0, slo_tpot_ms: float = 50.0) -> dict:
    results = {}
    for rate in rates:
        seed(30 + int(rate * 10))
        prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))]
                   for _ in range(num_requests)]
        sp = [SamplingParams(temperature=0.6, ignore_eos=True,
                             max_tokens=randint(100, 1024)) for _ in range(num_requests)]
        arrival_schedule = list(accumulate(expovariate(rate) for _ in range(num_requests)))

        engine.reset_metrics()
        engine.clear_prefix_cache()
        sent = 0
        arrival_t, schedule_t, first_t, finish_t, n_out = {}, {}, {}, {}, {}
        t_start = time.perf_counter()
        while sent < num_requests or not engine.is_finished():
            now = time.perf_counter() - t_start
            while sent < num_requests and arrival_schedule[sent] <= now:
                engine.add_request(prompts[sent], sp[sent])
                seq_id = engine.scheduler.waiting[-1].seq_id
                arrival_t[seq_id] = time.perf_counter()
                schedule_t[seq_id] = arrival_schedule[sent]
                sent += 1
            if engine.scheduler.waiting or engine.scheduler.running:
                outputs, _ = engine.step()
                now = time.perf_counter()
                for seq in engine.scheduler.running:
                    if seq.num_completion_tokens >= 1 and seq.seq_id not in first_t:
                        first_t[seq.seq_id] = now
                for seq_id, token_ids in outputs:
                    first_t.setdefault(seq_id, now)
                    finish_t[seq_id] = now
                    n_out[seq_id] = len(token_ids)
            else:
                time.sleep(0.001)
        total_time = time.perf_counter() - t_start

        ttfts, tpots, e2es, lags = [], [], [], []
        good = 0
        for seq_id in finish_t:
            ttft = first_t[seq_id] - arrival_t[seq_id]
            e2e = finish_t[seq_id] - arrival_t[seq_id]
            n = n_out[seq_id]
            tpot = (finish_t[seq_id] - first_t[seq_id]) / (n - 1) if n > 1 else 0.0
            ttfts.append(ttft)
            tpots.append(tpot)
            e2es.append(e2e)
            lags.append(arrival_t[seq_id] - t_start - schedule_t[seq_id])
            if ttft * 1e3 <= slo_ttft_ms and (n <= 1 or tpot * 1e3 <= slo_tpot_ms):
                good += 1
        results[f"rate_{rate:g}"] = {
            "num_requests": num_requests,
            "total_time_s": round(total_time, 3),
            "request_throughput": round(len(finish_t) / total_time, 3),
            "ttft_ms": {k: round(v * 1e3, 2) for k, v in summarize(ttfts).items()},
            "tpot_ms": {k: round(v * 1e3, 2) for k, v in summarize(tpots).items()},
            "e2e_latency_s": {k: round(v, 3) for k, v in summarize(e2es).items()},
            "injection_lag_ms": {k: round(v * 1e3, 2) for k, v in summarize(lags).items()},
            "goodput_req_s": round(good / total_time, 3),
            "slo_attainment": round(good / len(finish_t), 3) if finish_t else 0.0,
            "engine_metrics": engine.get_metrics(),
        }
    return {"slo": {"ttft_ms": slo_ttft_ms, "tpot_ms": slo_tpot_ms}, **results}


# --------------------------------------------------------------------------- #
# 元信息 / 报告
# --------------------------------------------------------------------------- #
def collect_meta(model_path: str, engine: LLM, repeats: int) -> dict:
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
        "repeats": repeats,
        "config": {
            "max_num_batched_tokens": getattr(cfg, "max_num_batched_tokens", None),
            "max_num_seqs": getattr(cfg, "max_num_seqs", None),
            "max_model_len": getattr(cfg, "max_model_len", None),
            "kvcache_block_size": getattr(cfg, "kvcache_block_size", None),
            "tensor_parallel_size": getattr(cfg, "tensor_parallel_size", None),
            "enforce_eager": getattr(cfg, "enforce_eager", None),
            "schedule_policy": getattr(cfg, "schedule_policy", None),
            "prefill_token_budget": getattr(cfg, "prefill_token_budget", None),
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
            elif _is_number(v):
                out[key] = v
        return out

    def suites_for_compare(report: dict) -> dict:
        out = {}
        for name, suite in report.get("suites", {}).items():
            out[name] = suite_view(suite)
        return out

    fo, fn = flat(suites_for_compare(old)), flat(suites_for_compare(new))
    print(f"\n{'metric':<58} {'old':>12} {'new':>12} {'change':>10}")
    print("-" * 96)
    for k in sorted(set(fo) & set(fn)):
        o, n = fo[k], fn[k]
        if o == 0:
            continue
        delta = (n - o) / o * 100
        better = (delta < 0) if any(s in k for s in ("ttft", "tpot", "itl", "latency", "time", "lag")) else (delta > 0)
        mark = " ↑优" if better and abs(delta) > 1 else (" ↓劣" if abs(delta) > 1 else "")
        print(f"{k:<58} {o:>12.2f} {n:>12.2f} {delta:>+9.1f}%{mark}")

    of = suite_view(old["suites"].get("correctness", {})).get("fingerprints", {})
    nf = suite_view(new["suites"].get("correctness", {})).get("fingerprints", {})
    if of and nf:
        mismatches = [p for p in of if p in nf and of[p] != nf[p]]
        print("\n正确性回归:", "通过 (指纹一致)" if not mismatches else f"失败! 不一致: {mismatches}")

    # 打印新报告的极差(若有), 便于判断收益是否大于噪声
    print("\n极差 (range, 仅 new 侧有三次统计时):")
    for name, suite in new.get("suites", {}).items():
        if isinstance(suite, dict) and suite.get("range"):
            rng = flat(suite["range"])
            for k, v in sorted(rng.items()):
                print(f"  {name}.{k}: {v}")


def run_with_repeats(name: str, fn: Callable, engine: LLM, repeats: int, **kwargs) -> dict:
    runs = []
    for i in range(repeats):
        print(f"  repeat {i + 1}/{repeats} ...")
        runs.append(fn(engine, **kwargs) if kwargs else fn(engine))
    if name == "correctness":
        fps = [r["fingerprints"] for r in runs]
        if any(f != fps[0] for f in fps):
            raise RuntimeError("correctness fingerprints unstable across repeats")
        return {"n_repeats": repeats, "median": runs[0], "range": {}}
    if name == "edge_cases":
        if any(not r.get("ok", False) for r in runs):
            raise RuntimeError(f"edge_cases failed: {[r for r in runs if not r.get('ok')]}")
        return {"n_repeats": repeats, "median": runs[0], "range": {}}
    return aggregate_runs(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--suites", nargs="+",
                        choices=["correctness", "throughput", "latency", "prefix_cache",
                                 "mixed", "edge_cases", "serving"],
                        default=["correctness", "throughput", "latency", "prefix_cache",
                                 "mixed", "edge_cases", "serving"])
    parser.add_argument("--serving-rates", nargs="+", type=float, default=[2.0, 4.0, 8.0],
                        help="serving 套件扫描的请求速率 (req/s)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="每套件重复次数, 取中位数并记极差 (1.3)")
    parser.add_argument("--save", default=None, help="保存名, 存至 results/<名>.json")
    parser.add_argument("--enforce-eager", action="store_true", help="禁用 CUDA Graph")
    parser.add_argument("--compare", nargs=2, metavar=("OLD.json", "NEW.json"),
                        help="对比两份结果后退出 (只比中位数视图)")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    engine = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=4096)
    print("warming up ...")
    warmup(engine)

    report = {"meta": collect_meta(args.model, engine, args.repeats), "suites": {}}
    for name in args.suites:
        print(f"running suite: {name} ...")
        fn = {
            "correctness": suite_correctness,
            "throughput": suite_throughput,
            "latency": suite_latency,
            "prefix_cache": suite_prefix_cache,
            "mixed": suite_mixed,
            "edge_cases": suite_edge_cases,
            "serving": suite_serving,
        }[name]
        kwargs = {"rates": args.serving_rates} if name == "serving" else {}
        report["suites"][name] = run_with_repeats(name, fn, engine, args.repeats, **kwargs)

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
