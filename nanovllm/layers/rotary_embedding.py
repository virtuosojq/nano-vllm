from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """经典 GPT-NeoX 风格旋转位置编码。
    把最后一维对半劈成 x1/x2, 做二维旋转:
      y1 = x1*cos - x2*sin
      y2 = x2*cos + x1*sin
    数学上等价于乘以旋转矩阵, 使 Q·K^T 只依赖相对位置。
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)   # fp32 计算更稳
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE: 预计算 cos/sin 表, 按位置索引做旋转。
    只作用于 Q 和 K(不旋转 V), 因为只有 Q·K 参与注意力分数。
    """

    def __init__(
        self,
        head_size: int,                 # D
        rotary_dim: int,                # 旋转维度(本项目等于 head_size)
        max_position_embeddings: int,   # 最大位置
        base: float,                    # 频率基数 theta
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 频率倒数: inv_freq[i] = 1 / base^(2i/dim), 不同维度用不同频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)        # 位置序列 [0, max_pos)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)                      # 外积 [max_pos, dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)                 # [max_pos, 1, dim](加 head 维)
        self.register_buffer("cos_sin_cache", cache, persistent=False)      # 不进 state_dict

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,       # [N] 每个 token 的绝对位置
        query: torch.Tensor,           # [N, hₗ, D]
        key: torch.Tensor,             # [N, kvₗ, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]          # 按位置索引出对应的 cos/sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)        # 旋转 Q
        key = apply_rotary_emb(key, cos, sin)            # 旋转 K
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """工厂函数 + 单槽缓存: 所有 decoder 层共享同一个 RotaryEmbedding 实例。
    这样 cos_sin_cache 只存一份(几十层共享), 节省显存。
    lru_cache(1) 等价于轻量级单例模式。
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
