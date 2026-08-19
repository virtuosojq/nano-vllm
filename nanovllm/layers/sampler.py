import torch
from torch import nn


class Sampler(nn.Module):
    """采样器: T=0 走 logits.argmax; 否则温度缩放 + Gumbel-max 多项式采样."""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: [S, V], temperatures: [S]
        logits = logits.float()
        greedy_mask = temperatures < 1e-5  # [S]
        # 避免 T=0 除零; 贪心结果最终用原始 logits.argmax 覆盖
        logits_scaled = logits / temperatures.clamp(min=1e-5).unsqueeze(dim=1)
        probs = torch.softmax(logits_scaled, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
