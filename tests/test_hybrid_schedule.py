"""2.7: hybrid 调度同一步可同时含 decode 与 prefill.

用法:
  python tests/test_hybrid_schedule.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

MODEL = os.environ.get("NANOVLLM_MODEL", "/root/autodl-tmp/huggingface/Qwen3-0.6B/")


def test_hybrid_same_step_mixed(model_dir: str = MODEL) -> None:
    config = Config(
        model_dir,
        num_kvcache_blocks=128,
        schedule_policy="hybrid",
        prefill_token_budget=256,
        max_num_seqs=8,
    )
    sched = Scheduler(config)
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    # 一条已完成 prefill、进入 decode 的序列
    dec = Sequence([1, 2, 3, 4, 5], sp)
    ncb = sched.block_manager.can_allocate(dec)
    assert ncb != -1
    sched.block_manager.allocate(dec, ncb)
    dec.num_cached_tokens = len(dec) - 1
    dec.status = SequenceStatus.RUNNING
    dec.is_prefill = False
    sched.running.append(dec)

    # waiting 长 prompt → 触发 chunked prefill
    pref = Sequence(list(range(100, 100 + 500)), sp)
    sched.waiting.append(pref)

    seqs, is_prefill = sched.schedule()
    assert is_prefill is True, is_prefill
    assert any(not s.is_prefill for s in seqs), "missing decode"
    assert any(s.is_prefill for s in seqs), "missing prefill"
    assert seqs[0] is dec
    n_pref_tok = sum(s.num_scheduled_tokens for s in seqs if s.is_prefill)
    assert 0 < n_pref_tok <= 256, n_pref_tok
    print(f"  [PASS] hybrid mixed: {len(seqs)} seqs, prefill_tokens={n_pref_tok}")


def test_hybrid_decode_only(model_dir: str = MODEL) -> None:
    config = Config(
        model_dir,
        num_kvcache_blocks=64,
        schedule_policy="hybrid",
        prefill_token_budget=256,
    )
    sched = Scheduler(config)
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    dec = Sequence([10, 11, 12], sp)
    ncb = sched.block_manager.can_allocate(dec)
    sched.block_manager.allocate(dec, ncb)
    dec.num_cached_tokens = len(dec) - 1
    dec.status = SequenceStatus.RUNNING
    dec.is_prefill = False
    sched.running.append(dec)

    seqs, is_prefill = sched.schedule()
    assert is_prefill is False
    assert len(seqs) == 1 and not seqs[0].is_prefill
    print("  [PASS] hybrid decode-only → is_prefill=False")


if __name__ == "__main__":
    test_hybrid_same_step_mixed()
    test_hybrid_decode_only()
    print("all ok")
