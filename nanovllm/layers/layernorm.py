import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Normalization。
    与 LayerNorm 的区别: 不减均值, 只用均方做缩放。
    公式: y = x / sqrt(mean(x²) + eps) * γ
    """

    def __init__(
        self,
        hidden_size: int,        # H
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))   # 可学习缩放 γ, 初始化为 1

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,         # [N, H]
    ) -> torch.Tensor:
        """单纯 norm(无残差加法)。用于第一层首次。"""
        orig_dtype = x.dtype
        x = x.float()                                         # 转 fp32 算方差更稳定
        var = x.pow(2).mean(dim=-1, keepdim=True)             # 均方(不中心化)
        x.mul_(torch.rsqrt(var + self.eps))                   # 原地除以 √(var+eps)
        x = x.to(orig_dtype).mul_(self.weight)                # 回原精度 × γ
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,         # 子层输出
        residual: torch.Tensor,  # 历史残差
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """融合残差加法 + norm: x = residual + x, 再 RMSNorm。
        融合的好处: 少一次显存读写(residual 加法和 norm 在同一 kernel)。
        返回归一化结果和更新后的 residual(供下一块使用)。
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())                  # x = x + residual (fp32)
        residual = x.to(orig_dtype)                           # 新残差 = 加和后的值(原精度)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)                        # 单纯 norm
        else:
            return self.add_rms_forward(x, residual)          # add + norm 融合
