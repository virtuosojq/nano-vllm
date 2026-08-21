import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 2.4/2.7: prefill_priority | decode_priority | token_budget | hybrid
    schedule_policy: str = "prefill_priority"
    # token_budget / hybrid: 每步 prefill 最多消耗的 token 数
    prefill_token_budget: int = 512
    # 2.2 n-gram / prompt-lookup 投机 (仅贪心 T≈0; 采样温度下自动回退普通 decode)
    enable_ngram: bool = False
    ngram_prompt_lookup_max: int = 4   # 最多 draft 多少 token (K)
    ngram_prompt_lookup_min: int = 1   # 匹配用的最小 n-gram
    ngram_max_ngram_size: int = 3      # 匹配用的最大 n-gram
    # 3.1: auto=模型 dtype 存 KV; int8=写侧量化 + 读前反量化
    kv_cache_dtype: str = "auto"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.schedule_policy in (
            "prefill_priority", "decode_priority", "token_budget", "hybrid",
        )
        assert self.prefill_token_budget > 0
        assert self.ngram_prompt_lookup_max >= 0
        assert 1 <= self.ngram_prompt_lookup_min <= self.ngram_max_ngram_size
        assert self.kv_cache_dtype in ("auto", "int8")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
