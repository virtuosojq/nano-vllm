import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,                 # K 张量基址 [N, kvₗ, D]
    key_stride,              # K 第 0 维步长(跨 token)
    value_ptr,               # V 张量基址
    value_stride,
    k_cache_ptr,             # paged K cache [num_blocks, block_size, kvₗ, D]
    v_cache_ptr,             # paged V cache
    slot_mapping_ptr,        # [N] 每个 token 的目标物理槽位
    D: tl.constexpr,         # kvₗ*D(编译期常量, 用于展开内层循环)
):
    """Triton kernel: 把新算的 K/V 散写到 paged KV cache。
    PagedAttention 的写侧: 每个 token 按 slot_mapping 写到对应的物理槽。
    一个 program 处理一个 token, 共 N 个 program 并行。
    """
    idx = tl.program_id(0)                          # 当前处理第 idx 个 token
    slot = tl.load(slot_mapping_ptr + idx)          # 该 token 的物理槽 = block_id*block_size+offset
    if slot == -1: return                           # -1 = prefix cache 命中, 无需写入, 直接跳过
    # 源地址: 当前 token 的 K/V(连续 D 个元素)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 目标地址: paged cache 里 slot 对应的位置(可能不连续, 由 block_table 决定)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)      # 散写 K
    tl.store(v_cache_ptr + cache_offsets, value)    # 散写 V


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """启动 kernel 的包装函数。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # 一系列 stride 断言: 保证内存布局符合 kernel 的假设(最后一维连续)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动 N 个 program, 每个 token 一个
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """注意力算子: 封装 K/V 写入 paged cache + flash attn 调用。
    prefill 用 flash_attn_varlen_func(变长, 多序列拼接),
    decode 用 flash_attn_with_kvcache(查 paged cache)。
    """

    def __init__(
        self,
        num_heads,     # 本卡 Q 头数 hₗ
        head_dim,      # D
        scale,         # 1/√D
        num_kv_heads,  # 本卡 KV 头数 kvₗ
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 占位空张量; 真正的 paged cache 由 ModelRunner.allocate_kv_cache 注入
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()                          # 拿 runner 准备好的张量
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():          # 若 paged cache 已分配
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)   # 把新 K/V 散写入 cache
        if context.is_prefill:
            if context.block_tables is not None:         # 命中 prefix cache: 直接用 cache 里的 K/V
                k, v = k_cache, v_cache
            # 变长注意力: 多条不等长序列拼成一条, 用 cu_seqlens 分隔, 避免 padding
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode: 每序列 1 token, 查自己的 paged KV cache
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
