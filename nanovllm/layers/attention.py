import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context

# 3.1: head_dim 内分组量化, 提升相对 per-head 的精度
INT8_KV_GROUP_SIZE = 16


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """bf16/fp16: 原样散写 K/V 到 paged cache。"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache_int8(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    group_size: int = INT8_KV_GROUP_SIZE,
):
    """group-wise 动态对称量化写入 int8 cache。

    key: [N, H, Dh]; k_scale: [num_blocks, block_size, H, G] fp16
    """
    valid = slot_mapping >= 0
    if not valid.any():
        return
    idx = valid.nonzero(as_tuple=False).squeeze(-1)
    slots = slot_mapping[idx]
    k = key.index_select(0, idx)
    v = value.index_select(0, idx)
    M, H, Dh = k.shape
    assert Dh % group_size == 0
    G = Dh // group_size
    k_g = k.view(M, H, G, group_size)
    v_g = v.view(M, H, G, group_size)
    k_amax = k_g.abs().amax(dim=-1).clamp_min(1e-8)
    v_amax = v_g.abs().amax(dim=-1).clamp_min(1e-8)
    k_q = (k_g / k_amax.unsqueeze(-1) * 127.0).round().clamp(-128, 127).to(torch.int8)
    v_q = (v_g / v_amax.unsqueeze(-1) * 127.0).round().clamp(-128, 127).to(torch.int8)
    k_s = (k_amax / 127.0).to(torch.float16)
    v_s = (v_amax / 127.0).to(torch.float16)
    block_size = k_cache.size(1)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offs = slots - blocks * block_size
    k_cache[blocks, offs] = k_q.view(M, H, Dh)
    v_cache[blocks, offs] = v_q.view(M, H, Dh)
    k_scale[blocks, offs] = k_s
    v_scale[blocks, offs] = v_s


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    if k_cache.dtype == torch.int8:
        assert k_scale is not None and v_scale is not None
        store_kvcache_int8(key, value, k_cache, v_cache, k_scale, v_scale, slot_mapping)
    else:
        store_kvcache_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, slot_mapping, D,
        )


def dequant_active_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    out_dtype: torch.dtype,
    group_size: int = INT8_KV_GROUP_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """反量化活跃物理块到 bf16/fp16 workspace, 并重映射 block_tables。"""
    valid = block_tables >= 0
    if not valid.any():
        empty = k_cache[:0].to(out_dtype)
        return empty, empty, block_tables

    phys = torch.unique(block_tables[valid])
    B = phys.numel()
    _, block_size, H, Dh = k_cache.shape
    G = Dh // group_size
    k_i8 = k_cache[phys].view(B, block_size, H, G, group_size)
    v_i8 = v_cache[phys].view(B, block_size, H, G, group_size)
    k_ws = (k_i8.to(out_dtype) * k_scale[phys].to(out_dtype).unsqueeze(-1)).view(B, block_size, H, Dh)
    v_ws = (v_i8.to(out_dtype) * v_scale[phys].to(out_dtype).unsqueeze(-1)).view(B, block_size, H, Dh)

    num_blocks = k_cache.size(0)
    mapper = torch.full((num_blocks,), -1, dtype=torch.int32, device=block_tables.device)
    mapper[phys] = torch.arange(B, dtype=torch.int32, device=block_tables.device)
    remapped = mapper[block_tables.clamp(min=0)]
    new_bt = torch.where(valid, remapped, block_tables)
    return k_ws, v_ws, new_bt


class Attention(nn.Module):
    """注意力算子: 封装 K/V 写入 paged cache + flash attn。
    3.1: use_int8_kv 时 group-wise 量化存储, 读前 gather-dequant。
    """

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.k_scale = self.v_scale = torch.tensor([])
        self.use_int8_kv = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.use_int8_kv:
                store_kvcache(
                    k, v, k_cache, v_cache, context.slot_mapping,
                    self.k_scale, self.v_scale,
                )
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            block_tables = context.block_tables
            if block_tables is not None:
                if self.use_int8_kv:
                    k, v, block_tables = dequant_active_kv(
                        k_cache, v_cache, self.k_scale, self.v_scale,
                        block_tables, q.dtype,
                    )
                else:
                    k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=block_tables,
            )
        else:
            if self.use_int8_kv:
                k_use, v_use, block_tables = dequant_active_kv(
                    k_cache, v_cache, self.k_scale, self.v_scale,
                    context.block_tables, q.dtype,
                )
            else:
                k_use, v_use = k_cache, v_cache
                block_tables = context.block_tables
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_use, v_use,
                cache_seqlens=context.context_lens, block_table=block_tables,
                softmax_scale=self.scale, causal=True,
            )
        return o
