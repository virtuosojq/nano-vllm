"""1.4 边界回归: #199 / #240 / #211 / #114

对照上游 PR/issue 的最小复现。能复现的打补丁; 不能复现的记录「已验证不适用」。

用法:
  python tests/test_1_4_edges.py
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.metrics import EngineMetrics
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus

MODEL = os.environ.get("NANOVLLM_MODEL", "/root/autodl-tmp/huggingface/Qwen3-0.6B/")
BS = 256


def _ok(name: str, detail: str = "") -> dict:
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    return {"name": name, "status": "pass", "detail": detail}


def _fail(name: str, detail: str) -> dict:
    print(f"  [FAIL] {name} — {detail}")
    return {"name": name, "status": "fail", "detail": detail}


def _na(name: str, detail: str) -> dict:
    """已验证不适用: 当前代码路径不再触发该 bug (通常已被 #218 等修复)."""
    print(f"  [N/A]  {name} — {detail}")
    return {"name": name, "status": "not_applicable", "detail": detail}


# --------------------------------------------------------------------------- #
# #199 / #114: 100% prefix 命中 / prompt 长度恰为 block_size
# 上游 #218 限制最多命中 num_blocks-1 个块, 避免 seqlen_q==0.
# --------------------------------------------------------------------------- #
def case_199_114(llm: LLM) -> list[dict]:
    results = []
    tok = llm.tokenizer
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    text = "Hello world test prompt for edge cases. " * 200

    for seqlens, tag in ((BS, "#114 exact block_size"), (BS * 2, "#199 multi-block full hit")):
        ids = tok.encode(text)[:seqlens]
        assert len(ids) == seqlens
        try:
            llm.clear_prefix_cache()
            llm.generate([ids], sp, use_tqdm=False)  # warm
            out = llm.generate([ids], sp, use_tqdm=False)  # hit attempt
            if len(out[0]["token_ids"]) != 8:
                results.append(_fail(tag, f"expected 8 tokens, got {len(out[0]['token_ids'])}"))
            else:
                # 当前实现因 #218 不会 100% 命中最后一块, 故不崩溃 = 已不适用原 crash
                results.append(_na(
                    tag,
                    "no crash; #218 caps cache at num_blocks-1 so seqlen_q never 0",
                ))
        except Exception as e:
            results.append(_fail(tag, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
    return results


# --------------------------------------------------------------------------- #
# #240: may_append 是否晚一拍 — 社区结论 ==1 相对 schedule→forward→postprocess 是正确的
# --------------------------------------------------------------------------- #
def case_240(llm: LLM) -> list[dict]:
    # prompt 长度 BS-2, 生成跨越块边界
    ids = list(range(100, 100 + (BS - 2)))
    sp = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    try:
        llm.clear_prefix_cache()
        out = llm.generate([ids], sp, use_tqdm=False)
        n = len(out[0]["token_ids"])
        if n != 16:
            return [_fail("#240 cross-boundary decode", f"got {n} tokens")]
        return [_na(
            "#240 may_append off-by-one",
            "cross-boundary decode OK; ==1 is correct given postprocess-before-next-schedule order",
        )]
    except Exception as e:
        return [_fail("#240 cross-boundary decode", f"{type(e).__name__}: {e}")]


# --------------------------------------------------------------------------- #
# #211: 抢占后前缀命中 + may_append 在 allocate 已备块时不再多分配
# --------------------------------------------------------------------------- #
def case_211_unit() -> list[dict]:
    results = []
    sched = Scheduler.__new__(Scheduler)
    sched.max_num_seqs = 8
    sched.max_num_batched_tokens = 4096
    sched.eos = -1
    sched.block_size = BS
    sched.block_manager = BlockManager(8, BS)
    sched.waiting = deque()
    sched.running = deque()
    sched.metrics = EngineMetrics()
    sched.schedule_policy = "prefill_priority"
    sched.prefill_token_budget = 512
    sched._budget_turn_prefill = True

    sp = SamplingParams(temperature=0.0, max_tokens=100, ignore_eos=True)

    # 1) 抢占 → 前缀命中恢复: scheduled 不应越过 block_table
    seq_a = Sequence(list(range(400)), sp)
    sched.add(seq_a)
    seqs, _ = sched.schedule()
    seq_a.num_scheduled_tokens = seq_a.num_tokens - seq_a.num_cached_tokens
    sched.postprocess(seqs, [999], True)
    for t in range(10):
        seqs, _ = sched.schedule()
        sched.postprocess(seqs, [1000 + t], False)
    sched.preempt(seq_a)

    seqs, is_prefill = sched.schedule()
    end = seq_a.num_cached_tokens + seq_a.num_scheduled_tokens
    end_block = (end + BS - 1) // BS
    if not is_prefill:
        results.append(_fail("#211 preempt resume schedule", "expected prefill after preempt"))
    elif end_block > len(seq_a.block_table):
        results.append(_fail(
            "#211 preempt resume schedule",
            f"end_block={end_block} > block_table={len(seq_a.block_table)}",
        ))
    else:
        results.append(_ok(
            "#211 preempt resume schedule",
            f"cached={seq_a.num_cached_tokens} scheduled={seq_a.num_scheduled_tokens} bt={len(seq_a.block_table)}",
        ))

    # 2) allocate 已按 num_blocks 备块时, may_append 不得再追加
    sched.waiting.clear()
    sched.running.clear()
    seq_b = Sequence(list(range(BS + 1)), sp)  # len=257, num_blocks=2
    sched.add(seq_b)
    sched.schedule()  # allocate → bt 长度应为 2
    bt_before = list(seq_b.block_table)
    # 不经 postprocess, 直接模拟「表已备好、len%BS==1」的 decode 调度
    seq_b.status = SequenceStatus.RUNNING
    sched.waiting.clear()
    sched.running.append(seq_b)
    sched.schedule()
    bt_after = list(seq_b.block_table)
    if len(bt_after) > seq_b.num_blocks:
        results.append(_fail(
            "#211 may_append no over-allocate",
            f"bt {bt_before} → {bt_after}, num_blocks={seq_b.num_blocks}",
        ))
    else:
        results.append(_ok(
            "#211 may_append no over-allocate",
            f"bt stayed {bt_after} (== num_blocks {seq_b.num_blocks})",
        ))

    # 3) 正常增长路径: len 从 BS 增到 BS+1 时仍应分配新块
    sched.waiting.clear()
    sched.running.clear()
    sched.block_manager = BlockManager(8, BS)
    seq_c = Sequence(list(range(BS)), sp)  # len=256, 1 block
    sched.add(seq_c)
    seqs, _ = sched.schedule()
    seq_c.num_scheduled_tokens = BS
    sched.postprocess(seqs, [42], True)  # append → len=257
    assert len(seq_c) == BS + 1
    assert len(seq_c.block_table) == 1
    seqs, pref = sched.schedule()
    if pref or len(seq_c.block_table) != 2:
        results.append(_fail(
            "#211 normal growth still allocates",
            f"prefill={pref} bt_len={len(seq_c.block_table)}",
        ))
    else:
        results.append(_ok("#211 normal growth still allocates", f"bt={seq_c.block_table}"))

    return results


def case_211_gpu(llm: LLM) -> list[dict]:
    """中等并发压测: 不刻意榨干显存, 只要跑完且长度正确."""
    sp = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
    prompts = [list(range(i * 13, i * 13 + 300)) for i in range(24)]
    try:
        llm.clear_prefix_cache()
        outs = llm.generate(prompts, [sp] * len(prompts), use_tqdm=False)
        if any(len(o["token_ids"]) != 32 for o in outs):
            return [_fail("#211 gpu multi-seq", "incomplete generations")]
        return [_ok("#211 gpu multi-seq", f"finished {len(outs)} seqs")]
    except Exception as e:
        return [_fail("#211 gpu multi-seq", f"{type(e).__name__}: {e}")]


def run_all(model: str = MODEL) -> dict:
    print("1.4 edge cases")
    print("unit: #211 may_append / preempt")
    results = case_211_unit()

    print("gpu: #199/#114, #240, #211")
    llm = LLM(model, enforce_eager=True, max_model_len=4096)
    try:
        results.extend(case_199_114(llm))
        results.extend(case_240(llm))
        results.extend(case_211_gpu(llm))
    finally:
        llm.exit()

    n_pass = sum(1 for r in results if r["status"] in ("pass", "not_applicable"))
    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"\nsummary: {n_pass} pass/n/a, {n_fail} fail / {len(results)} total")
    return {
        "cases": results,
        "num_pass_or_na": n_pass,
        "num_fail": n_fail,
        "ok": n_fail == 0,
    }


if __name__ == "__main__":
    report = run_all()
    sys.exit(0 if report["ok"] else 1)
