from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.metrics import EngineMetrics

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    """Continuous batching 调度器.

    策略 (2.4 / 2.7):
      prefill_priority  waiting 可调度则本步只做 prefill (decode 可能被饿死)
      decode_priority   running 非空则优先 decode
      token_budget      prefill 每步最多 prefill_token_budget; 与 decode 轮转
                        (单步仍纯 P 或纯 D)
      hybrid            同一步先 decode, 再用剩余 seq 槽 + prefill_token_budget
                        塞 chunked prefill (Sarathi stall-free / 2.7)
    """

    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = EngineMetrics()
        self.schedule_policy = config.schedule_policy
        self.prefill_token_budget = config.prefill_token_budget
        # token_budget: True → 下一步优先尝试限预算 prefill
        self._budget_turn_prefill = True
        self.enable_ngram = config.enable_ngram
        self.ngram_prompt_lookup_max = config.ngram_prompt_lookup_max
        self.ngram_prompt_lookup_min = config.ngram_prompt_lookup_min
        self.ngram_max_ngram_size = config.ngram_max_ngram_size

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def abort(self, seq_id: int) -> bool:
        """Drop a waiting/running sequence and free its KV blocks.

        Safe to call between steps. Used by the OpenAI server for stop strings
        and client disconnects.
        """
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.seq_id != seq_id:
                    continue
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                queue.remove(seq)
                return True
        return False

    def _sample_kv(self):
        bm = self.block_manager
        self.metrics.sample_kv(len(bm.free_block_ids), len(bm.blocks))

    def _try_schedule_prefill(
        self,
        token_budget: int | None = None,
        max_seqs: int | None = None,
    ) -> list[Sequence] | None:
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0
        limit = self.max_num_batched_tokens if token_budget is None else min(
            token_budget, self.max_num_batched_tokens
        )
        seq_cap = self.max_num_seqs if max_seqs is None else min(max_seqs, self.max_num_seqs)
        if seq_cap <= 0:
            return None

        while self.waiting and len(scheduled_seqs) < seq_cap:
            seq = self.waiting[0]
            remaining = limit - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                self.metrics.note_prefix(num_cached_blocks, seq.num_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.num_scheduled_tokens <= 0:
                # 理论上 #218 后不会出现; 防御
                break
            seq.is_prefill = True
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        return scheduled_seqs or None

    def _try_schedule_decode(self) -> list[Sequence] | None:
        scheduled_seqs: list[Sequence] = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            return None
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule(self) -> tuple[list[Sequence], bool]:
        policy = self.schedule_policy

        if policy == "prefill_priority":
            pref = self._try_schedule_prefill(None)
            if pref is not None:
                self._sample_kv()
                return pref, True
            dec = self._try_schedule_decode()
            assert dec is not None
            self._sample_kv()
            return dec, False

        if policy == "decode_priority":
            dec = self._try_schedule_decode()
            if dec is not None:
                self._sample_kv()
                return dec, False
            pref = self._try_schedule_prefill(None)
            assert pref is not None
            self._sample_kv()
            return pref, True

        if policy == "hybrid":
            # 2.7: 同一步 decode + 限预算 prefill chunk; 混合批对 runner 报 is_prefill=True
            dec = self._try_schedule_decode() or []
            remaining_slots = self.max_num_seqs - len(dec)
            pref = self._try_schedule_prefill(
                self.prefill_token_budget, max_seqs=remaining_slots,
            ) or []
            assert dec or pref
            self._sample_kv()
            if dec and pref:
                return dec + pref, True
            if pref:
                return pref, True
            return dec, False

        # token_budget: 限预算 prefill 与 decode 轮转, 避免长 prefill 饿死 decode
        assert policy == "token_budget"
        budget = self.prefill_token_budget
        if self.waiting and self.running:
            if self._budget_turn_prefill:
                pref = self._try_schedule_prefill(budget)
                if pref is not None:
                    self._budget_turn_prefill = False
                    self._sample_kv()
                    return pref, True
            dec = self._try_schedule_decode()
            if dec is not None:
                self._budget_turn_prefill = True
                self._sample_kv()
                return dec, False
            pref = self._try_schedule_prefill(budget)
            assert pref is not None
            self._budget_turn_prefill = False
            self._sample_kv()
            return pref, True
        if self.waiting:
            pref = self._try_schedule_prefill(budget)
            assert pref is not None
            self._sample_kv()
            return pref, True
        dec = self._try_schedule_decode()
        assert dec is not None
        self._sample_kv()
        return dec, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool = True):
        # is_prefill 保留兼容; 是否 append 按 per-seq 完整性判断 (2.7 hybrid 必需)
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 未完成的 prefill chunk: 只推进 cache, 不采样追加
            if seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess_ngram(self, seqs: list[Sequence], token_lists: list[list[int]]):
        """2.2: 先把接受的 draft 写入 token_ids 再 hash; 最后 append bonus."""
        for seq, tokens in zip(seqs, token_lists):
            accepted = seq.num_accepted_drafts
            assert len(tokens) == accepted + 1, (len(tokens), accepted)
            self.metrics.note_ngram(seq.num_draft_tokens, accepted)
            # 接受的 draft 必须先进入 token_ids, hash_blocks 才读得到完整块
            for token_id in tokens[:accepted]:
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
                    seq.verify_input_ids = None
                    seq.num_draft_tokens = 0
                    seq.num_accepted_drafts = 0
                    break
            else:
                seq.num_scheduled_tokens = 1 + accepted
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                seq.verify_input_ids = None
                seq.num_draft_tokens = 0
                seq.num_accepted_drafts = 0
                token_id = tokens[accepted]
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
