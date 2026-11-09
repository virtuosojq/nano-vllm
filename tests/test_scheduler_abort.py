"""abort() must free waiting and running sequences between steps."""

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class _DummyConfig:
    max_num_seqs = 4
    max_num_batched_tokens = 128
    eos = 0
    kvcache_block_size = 256
    num_kvcache_blocks = 16
    schedule_policy = "prefill_priority"
    prefill_token_budget = 64
    enable_ngram = False
    ngram_prompt_lookup_max = 4
    ngram_prompt_lookup_min = 1
    ngram_max_ngram_size = 3


def _scheduler() -> Scheduler:
    # Bypass Config.__post_init__ (needs a real HF directory).
    sch = Scheduler.__new__(Scheduler)
    Scheduler.__init__(sch, _DummyConfig())
    Sequence.block_size = 256
    return sch


def test_abort_waiting_seq():
    sch = _scheduler()
    seq = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
    sch.add(seq)
    assert sch.abort(seq.seq_id) is True
    assert sch.is_finished()
    assert seq.status is SequenceStatus.FINISHED
    assert sch.abort(seq.seq_id) is False


def test_abort_running_seq_frees_blocks():
    sch = _scheduler()
    seq = Sequence([1] * 8, SamplingParams(max_tokens=4, ignore_eos=True))
    sch.add(seq)
    scheduled, is_prefill = sch.schedule()
    assert is_prefill and scheduled == [seq]
    assert seq.block_table
    used_before = len(sch.block_manager.used_block_ids)
    assert used_before > 0
    assert sch.abort(seq.seq_id) is True
    assert seq.block_table == []
    assert sch.is_finished()
