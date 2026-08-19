from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """全局上下文: 模型层与 ModelRunner 之间的桥梁。
    ModelRunner.prepare_* 计算好下列张量后调用 set_context(...) 写入,
    模型层(Attention/LMHead)通过 get_context() 读取, 避免这些张量层层透传。
    slots=True 节省内存且禁止动态属性。
    """
    is_prefill: bool = False                              # True=prefill分支, False=decode分支
    cu_seqlens_q: torch.Tensor | None = None              # [S+1] prefill: query 的累积序列长度(varlen)
    cu_seqlens_k: torch.Tensor | None = None              # [S+1] prefill: key 的累积序列长度
    max_seqlen_q: int = 0                                 # prefill: batch 内最大 query 长度(flash attn 优化提示)
    max_seqlen_k: int = 0                                 # prefill: batch 内最大 key 长度
    slot_mapping: torch.Tensor | None = None              # [N] 每个 token 写入 paged KV cache 的物理槽位
                                                         #       = block_id * block_size + offset; -1 表示 prefix cache 命中跳过
    context_lens: torch.Tensor | None = None              # [S] decode: 每条序列当前长度
    block_tables: torch.Tensor | None = None              # [S, max_blocks] 每条序列引用的物理块 ID 列表
    # 2.2 verify: False → LMHead 保留全部 query 位置的 logits (而非每条只取末 token)
    gather_last_logit_only: bool = True


_CONTEXT = Context()

def get_context():
    """模型层调用: 获取当前 step 的上下文张量。"""
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    gather_last_logit_only: bool = True,
):
    """ModelRunner 调用: 把准备好的张量写入全局 Context。"""
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables, gather_last_logit_only,
    )

def reset_context():
    """每次 step 结束后调用: 清空 Context, 防止下次误用旧张量。"""
    global _CONTEXT
    _CONTEXT = Context()
