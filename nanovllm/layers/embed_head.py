import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表并行 Embedding: 把词表按行切分到各卡。
    每卡只持有 V/tp 行, 通过 mask + all_reduce 聚合得到完整结果。
    """

    def __init__(
        self,
        num_embeddings: int,        # V: 词表大小
        embedding_dim: int,         # H: 嵌入维度
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size   # 本卡词表行数 V/tp
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank   # 本卡词表起点
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition  # 本卡词表终点
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))  # [V/tp, H]
        self.weight.weight_loader = self.weight_loader    # 给 loader 用的切分钩子

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载时只取本卡负责的词表行。"""
        param_data = param.data
        shard_size = param_data.size(0)                # 本卡行数 V/tp
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)   # 截取本卡片段
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # x: [N] token ids
        if self.tp_size > 1:
            # 屏蔽不属于本卡区间的 id: 先算 mask, 再平移到本卡局部索引
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)      # 非本卡 token → 索引 0(占位)
        y = F.embedding(x, self.weight)                # [N, H] 查表
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y                  # 非本卡 token 的输出清零
            dist.all_reduce(y)                          # 各卡求和 → 每个位置只有正确那张卡有值
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """LM Head: 复用 VocabParallel 的权重切分(按词表行切)。
    forward 用 F.linear 而非 embedding, 且把结果 gather 到 rank0。
    """

    def __init__(
        self,
        num_embeddings: int,        # V
        embedding_dim: int,         # H
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # x: [N, H] (decode) 或 [N, H] (prefill, 默认取末 token; verify 保留全部)
        context = get_context()
        if context.is_prefill and context.gather_last_logit_only:
            # 普通 prefill: 每条序列只需最后一个 token 的 logits
            last_indices = context.cu_seqlens_q[1:] - 1   # 每条序列末 token 的位置
            x = x[last_indices].contiguous()              # [S, H]
        logits = F.linear(x, self.weight)                 # [S or N, V/tp]
        if self.tp_size > 1:
            # gather 到 rank0: rank0 拼成完整词表 logits, 其它卡返回 None
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
