"""GPU-free engine with the same serving surface as LLMEngine.

Used to validate the OpenAI protocol, stop strings, and `vllm bench serve`
client wiring without loading a model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter, sleep

from nanovllm.engine.metrics import EngineMetrics
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:
    """Byte-as-token tokenizer. `encode`/`decode` are inverses for BMP text."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(ch) for ch in text]
        return ids or [ord("x")]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid == self.eos_token_id:
                continue
            chars.append(chr(int(tid) % 0x110000))
        return "".join(chars)


@dataclass
class _FakeConfig:
    max_model_len: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    schedule_policy: str = "fake"
    kv_cache_dtype: str = "auto"
    num_kvcache_blocks: int = 0


class FakeEngine:
    """One completion token per step, optionally sleeping to emulate latency."""

    def __init__(
        self,
        *,
        output_text: str = "x" * 4096,
        step_sleep: float = 0.0,
        max_model_len: int = 4096,
        max_num_seqs: int = 256,
        tokens_per_step: int = 1,
    ):
        self.tokenizer = FakeTokenizer()
        self.config = _FakeConfig(max_model_len=max_model_len, max_num_seqs=max_num_seqs)
        self.step_sleep = step_sleep
        self.tokens_per_step = max(1, tokens_per_step)
        self._output_ids = self.tokenizer.encode(output_text)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = EngineMetrics()
        # Mimic LLMEngine.scheduler so the server can scan live sequences.
        self.scheduler = self

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if not prompt:
            prompt = [ord("x")]
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        self.waiting.append(seq)
        return seq.seq_id

    def abort(self, seq_id: int) -> bool:
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id != seq_id:
                    continue
                seq.status = SequenceStatus.FINISHED
                queue.remove(seq)
                return True
        return False

    def get_seq(self, seq_id: int) -> Sequence | None:
        for seq in self.waiting:
            if seq.seq_id == seq_id:
                return seq
        for seq in self.running:
            if seq.seq_id == seq_id:
                return seq
        return None

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def reset_metrics(self) -> None:
        self.metrics.reset()

    def get_metrics(self) -> dict:
        return self.metrics.summary()

    def _next_token(self, seq: Sequence) -> int:
        idx = seq.num_completion_tokens
        if idx >= len(self._output_ids):
            return self.tokenizer.eos_token_id
        return self._output_ids[idx]

    def step(self):
        if self.step_sleep > 0:
            sleep(self.step_sleep)
        while self.waiting and len(self.running) < self.config.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            seq.is_prefill = False
            self.running.append(seq)

        finished: list[Sequence] = []
        n_decode = 0
        for seq in list(self.running):
            for _ in range(self.tokens_per_step):
                seq.append_token(self._next_token(seq))
                n_decode += 1
                hit_eos = (not seq.ignore_eos) and seq.last_token == self.tokenizer.eos_token_id
                if hit_eos or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_time = perf_counter()
                    if seq.first_token_time is None:
                        seq.first_token_time = seq.finish_time
                    ttft = seq.first_token_time - seq.arrival_time
                    n = seq.num_completion_tokens
                    tpot = ((seq.finish_time - seq.first_token_time) / (n - 1)) if n > 1 else None
                    self.metrics.note_finished(ttft, tpot)
                    self.running.remove(seq)
                    finished.append(seq)
                    break
            else:
                if seq.first_token_time is None and seq.num_completion_tokens >= 1:
                    seq.first_token_time = perf_counter()

        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        return outputs, -n_decode
