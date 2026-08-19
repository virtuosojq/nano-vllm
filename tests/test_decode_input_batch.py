"""2.6 DecodeInputBatch 单元测试 (无 GPU 也可跑打包逻辑)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.engine.decode_input_batch import DecodeInputBatch
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_pack_values_and_stable_storage():
    db = DecodeInputBatch(16, 8, block_size=256, pin_memory=False, allocate_temperature_gpu=False)
    addr_ids = db.input_ids_cpu.data_ptr()
    sp = SamplingParams(temperature=0.5, max_tokens=8, ignore_eos=True)
    seqs = []
    for i in range(3):
        s = Sequence(list(range(10 + i)), sp)
        s.block_table = [i, i + 10]
        # last_block_num_tokens for len=10+i: num_blocks=1 if <256
        seqs.append(s)

    view = db.prepare_decode(seqs, graph_bucket=4, include_temperatures=True)
    assert view.batch_size == 3
    assert view.graph_bucket == 4
    assert view.active_block_width == 2
    assert view.padding_rows == 1
    assert db.input_ids_np[0] == seqs[0].last_token
    assert db.positions_np[1] == len(seqs[1]) - 1
    assert db.context_lens_np[2] == len(seqs[2])
    assert db.slot_mapping_np[0] == seqs[0].block_table[-1] * 256 + seqs[0].last_block_num_tokens - 1
    assert list(db.block_tables_np[1, :2]) == seqs[1].block_table
    assert db.block_tables_np[3, 0] == -1  # padded row
    assert db.input_ids_cpu.data_ptr() == addr_ids  # 存储地址稳定


def test_bucket_transitions():
    db = DecodeInputBatch(16, 8, block_size=256, pin_memory=False, allocate_temperature_gpu=False)
    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)

    def make(n):
        out = []
        for i in range(n):
            s = Sequence([i] * 20, sp)
            s.block_table = [i]
            out.append(s)
        return out

    for bs, bucket in [(1, 1), (8, 8), (3, 4), (16, 16), (1, 1)]:
        view = db.prepare_decode(make(bs), graph_bucket=bucket, include_temperatures=False)
        assert view.batch_size == bs
        assert view.graph_bucket == bucket


def test_no_torch_tensor_during_pack(monkeypatch=None):
    db = DecodeInputBatch(8, 4, block_size=256, pin_memory=False, allocate_temperature_gpu=False)
    sp = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
    seqs = []
    for i in range(2):
        s = Sequence([1, 2, 3], sp)
        s.block_table = [0]
        seqs.append(s)

    orig = torch.tensor
    calls = {"n": 0}

    def guarded(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    torch.tensor = guarded  # type: ignore
    try:
        db.prepare_decode(seqs, graph_bucket=2, include_temperatures=True)
    finally:
        torch.tensor = orig  # type: ignore
    assert calls["n"] == 0, f"prepare_decode allocated via torch.tensor {calls['n']} times"


if __name__ == "__main__":
    test_pack_values_and_stable_storage()
    test_bucket_transitions()
    test_no_torch_tensor_during_pack()
    print("DecodeInputBatch unit tests PASS")
