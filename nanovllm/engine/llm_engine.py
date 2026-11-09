import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.config = config
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def reset_metrics(self):
        self.scheduler.metrics.reset()

    def clear_prefix_cache(self):
        assert self.is_finished(), "clear_prefix_cache requires idle engine"
        self.scheduler.block_manager.clear_prefix_cache()

    def get_metrics(self) -> dict:
        return self.scheduler.metrics.summary()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        self.scheduler.add(seq)
        return seq.seq_id

    def abort(self, seq_id: int) -> bool:
        return self.scheduler.abort(seq_id)

    def get_seq(self, seq_id: int) -> Sequence | None:
        for seq in self.scheduler.waiting:
            if seq.seq_id == seq_id:
                return seq
        for seq in self.scheduler.running:
            if seq.seq_id == seq_id:
                return seq
        return None

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # hybrid 步返回 is_prefill=True, 因此不会进 ngram; 纯 decode 才可能投机
        if (not is_prefill) and self.scheduler.enable_ngram:
            self.model_runner._prepare_ngram_drafts(seqs, self.scheduler.block_manager)
            token_lists = self.model_runner.call("run_ngram", seqs)
            self.scheduler.postprocess_ngram(seqs, token_lists)
            num_tokens = -sum(len(t) for t in token_lists)
        else:
            # 在 postprocess 清零 num_scheduled_tokens 之前统计
            n_pref = sum(s.num_scheduled_tokens for s in seqs if s.is_prefill)
            n_dec = sum(1 for s in seqs if not s.is_prefill)
            if n_pref > 0:
                num_tokens = n_pref  # 含 hybrid: 进度条走 Prefill
            else:
                num_tokens = -n_dec
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
        now = perf_counter()
        for seq in seqs:
            if seq.num_completion_tokens >= 1 and seq.first_token_time is None:
                seq.first_token_time = now
            if seq.is_finished:
                seq.finish_time = now
                ttft = seq.first_token_time - seq.arrival_time
                n = seq.num_completion_tokens
                tpot = ((seq.finish_time - seq.first_token_time) / (n - 1)) if n > 1 else None
                self.scheduler.metrics.note_finished(ttft, tpot)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
