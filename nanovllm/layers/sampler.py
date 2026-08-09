import torch
from torch import nn


class Sampler(nn.Module):
    """采样器: 温度缩放 + 概率采样。
    用 Gumbel-max 变体实现多项式采样(argmax 代替 multinomial, 便于 torch.compile 融合)。
    """

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: [S, V], temperatures: [S]
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))   # 温度缩放: logits/T
        probs = torch.softmax(logits, dim=-1)                          # → 概率分布 [S, V]
        # Gumbel-max 变体: argmax(probs / Exp(1)) 等价于按 probs 多项式采样
        # 数学: argmax(log(p) - log(E)) = argmax(p / E), 其中 E ~ Exp(1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)  # 采样 Exp(1) 噪声, clamp 防除零
        ).argmax(dim=-1)
        return sample_tokens    # [S] 每条序列一个 token
