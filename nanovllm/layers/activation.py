import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 激活: silu(gate) * up。
    输入是 gate_up_proj 的拼接结果(最后一维对半劈: 前半=gate, 后半=up)。
    用 @torch.compile 融合成单个 kernel, 减少访存。
    """

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 2*I/tp] → 对半劈
        x, y = x.chunk(2, -1)        # x=gate, y=up
        return F.silu(x) * y         # silu(gate) * up, silu(x)=x*sigmoid(x)
