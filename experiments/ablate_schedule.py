"""2.4/2.7 调度策略消融: prefill_priority / decode_priority / token_budget / hybrid.

在相同 mixed 负载下对比:
  - long_prefill 组的 TTFT (长 prompt 短 decode)
  - long_decode 组的 TPOT / ITL P99 (短 prompt 长 decode)
  - hybrid (2.7): 同一步 P+D vs token_budget 轮转

用法:
  python experiments/ablate_schedule.py
  python experiments/ablate_schedule.py --repeats 3 --budgets 256 512 1024 2048 --save opt_2.7_hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from random import randint, seed

from _root import result_path
from nanovllm import LLM, SamplingParams

DEFAULT_MODEL = "/root/autodl-tmp/huggingface/Qwen3-0.6B/"


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    s = sorted(vals)

    def pct(p: float) -> float:
        k = (len(s) - 1) * p / 100
        f, c = int(k), min(int(k) + 1, len(s) - 1)
        return s[f] + (s[c] - s[f]) * (k - f)

    return {
        "mean": sum(s) / len(s),
        "p50": pct(50),
        "p90": pct(90),
        "p99": pct(99),
    }


def median_scalar(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


def run_mixed_once(
    engine: LLM,
    num_long_prefill: int = 16,
    num_long_decode: int = 16,
) -> dict:
    """同批混合负载, 分组统计 TTFT / TPOT / ITL."""
    seed(3)
    prompts, sps, groups = [], [], {}
    for i in range(num_long_prefill):
        prompts.append([randint(0, 10000) for _ in range(1024)])
        sps.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32))
        groups[i] = "long_prefill"  # provisional; remap by seq_id after add
    for i in range(num_long_decode):
        prompts.append([randint(0, 10000) for _ in range(64)])
        sps.append(SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=256))

    engine.clear_prefix_cache()
    engine.reset_metrics()

    group_of: dict[int, str] = {}
    t_start = time.perf_counter()
    for i, (p, sp) in enumerate(zip(prompts, sps)):
        engine.add_request(p, sp)
        seq_id = engine.scheduler.waiting[-1].seq_id
        group_of[seq_id] = "long_prefill" if i < num_long_prefill else "long_decode"

    first_t, finish_t, n_out = {}, {}, {}
    last_t, prev_n = {}, {}
    itls = {"long_prefill": [], "long_decode": [], "all": []}

    while not engine.is_finished():
        outputs, _ = engine.step()
        now = time.perf_counter()

        def observe(seq_id: int, n: int):
            if n <= prev_n.get(seq_id, 0):
                return
            g = group_of[seq_id]
            if seq_id in last_t:
                delta = now - last_t[seq_id]
                itls[g].append(delta)
                itls["all"].append(delta)
            last_t[seq_id] = now
            first_t.setdefault(seq_id, now)
            prev_n[seq_id] = n

        for seq in engine.scheduler.running:
            if seq.num_completion_tokens >= 1:
                observe(seq.seq_id, seq.num_completion_tokens)
        for seq_id, token_ids in outputs:
            observe(seq_id, len(token_ids))
            first_t.setdefault(seq_id, now)
            finish_t[seq_id] = now
            n_out[seq_id] = len(token_ids)

    total_time = time.perf_counter() - t_start

    def group_metrics(gname: str) -> dict:
        ttfts, tpots, e2es = [], [], []
        for seq_id, fin in finish_t.items():
            if group_of[seq_id] != gname:
                continue
            ttfts.append(first_t[seq_id] - t_start)
            e2es.append(fin - t_start)
            n = n_out[seq_id]
            if n > 1:
                tpots.append((fin - first_t[seq_id]) / (n - 1))
        return {
            "num_seqs": sum(1 for s in group_of.values() if s == gname),
            "ttft_ms": {k: round(v * 1e3, 2) for k, v in summarize(ttfts).items()},
            "tpot_ms": {k: round(v * 1e3, 2) for k, v in summarize(tpots).items()},
            "itl_ms": {k: round(v * 1e3, 2) for k, v in summarize(itls[gname]).items()},
            "e2e_latency_s": {k: round(v, 3) for k, v in summarize(e2es).items()},
        }

    total_completion = sum(n_out.values())
    return {
        "total_time_s": round(total_time, 3),
        "throughput_tok_s": round(total_completion / total_time, 2),
        "long_prefill": group_metrics("long_prefill"),
        "long_decode": group_metrics("long_decode"),
        "all_itl_ms": {k: round(v * 1e3, 2) for k, v in summarize(itls["all"]).items()},
        "engine_metrics": engine.get_metrics(),
    }


def aggregate_runs(runs: list[dict]) -> dict:
    """对数值叶子取中位数 + 极差."""

    def build(objs):
        sample = objs[0]
        if isinstance(sample, dict):
            med, rng = {}, {}
            for k in sample:
                m, r = build([o[k] for o in objs])
                med[k] = m
                if r is not None and r != {}:
                    rng[k] = r
            return med, rng
        if isinstance(sample, (int, float)) and not isinstance(sample, bool):
            nums = [float(o) for o in objs]
            return median_scalar(nums), round(max(nums) - min(nums), 6)
        return sample, None

    med, rng = build(runs)

    def clean(r):
        if not isinstance(r, dict):
            return r
        return {k: clean(v) for k, v in r.items() if v is not None and v != {}}

    return {"n_repeats": len(runs), "median": med, "range": clean(rng)}


def make_engine(model: str, policy: str, budget: int) -> LLM:
    return LLM(
        model,
        enforce_eager=False,
        max_model_len=4096,
        schedule_policy=policy,
        prefill_token_budget=budget,
        enable_ngram=False,  # 2.7: hybrid 与 ngram 互斥; 消融统一关掉
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("NANOVLLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048, 4096])
    parser.add_argument("--save", default="opt_2.7_hybrid")
    args = parser.parse_args()

    configs: list[tuple[str, str, int]] = [
        ("prefill_priority", "prefill_priority", 512),
        ("decode_priority", "decode_priority", 512),
    ]
    for b in args.budgets:
        configs.append((f"token_budget_{b}", "token_budget", b))
    # 2.7: 同一步 hybrid, budget=512 与默认 token_budget 对齐
    configs.append(("hybrid_512", "hybrid", 512))

    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": args.model,
            "repeats": args.repeats,
            "workload": "mixed 16x(1024→32) + 16x(64→256)",
            "enable_ngram": False,
        },
        "points": {},
    }

    # 同一引擎切换策略会更干净? 策略在 Scheduler 上可热改, 避免重复加载.
    print("loading engine ...")
    engine = make_engine(args.model, "prefill_priority", 512)
    # warmup
    engine.generate([[1, 2, 3, 4]], SamplingParams(temperature=0.0, max_tokens=4), use_tqdm=False)

    try:
        for name, policy, budget in configs:
            print(f"\n=== {name} (policy={policy}, budget={budget}) ===")
            engine.scheduler.schedule_policy = policy
            engine.scheduler.prefill_token_budget = budget
            engine.scheduler._budget_turn_prefill = True
            engine.scheduler.enable_ngram = False
            runs = []
            for i in range(args.repeats):
                print(f"  repeat {i + 1}/{args.repeats}")
                runs.append(run_mixed_once(engine))
            report["points"][name] = aggregate_runs(runs)
            m = report["points"][name]["median"]
            print(
                f"  thr={m['throughput_tok_s']:.1f}  "
                f"LP_ttft_p50={m['long_prefill']['ttft_ms']['p50']:.1f}ms  "
                f"LD_tpot_p99={m['long_decode']['tpot_ms']['p99']:.1f}ms  "
                f"LD_itl_p99={m['long_decode']['itl_ms']['p99']:.1f}ms"
            )
    finally:
        engine.exit()

    # 打印权衡表
    print("\n" + "=" * 88)
    print(f"{'policy':<22} {'thr':>8} {'LP_TTFT_p50':>12} {'LD_TPOT_p99':>12} {'LD_ITL_p99':>12}")
    print("-" * 88)
    for name, data in report["points"].items():
        m = data["median"]
        print(
            f"{name:<22} {m['throughput_tok_s']:8.1f} "
            f"{m['long_prefill']['ttft_ms']['p50']:12.1f} "
            f"{m['long_decode']['tpot_ms']['p99']:12.1f} "
            f"{m['long_decode']['itl_ms']['p99']:12.1f}"
        )

    path = result_path(args.save)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
