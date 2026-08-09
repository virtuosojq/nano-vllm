import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除断言: 保证能被 tp_size 均分。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """所有 TP Linear 的基类。
    权重约定 PyTorch 布局: weight.shape = [output_size, input_size]。
    每个参数都挂一个 weight_loader 钩子, 供 loader.py 加载时按 tp 切分。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,       # 切分维: Column=0(weight的行), Row=1(weight的列)
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))   # [out, in]
        self.weight.weight_loader = self.weight_loader                    # 给 loader 用的切分钩子
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """副本并行: 所有卡完整复制权重, 不切分(本项目暂未使用)。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)      # 整块拷贝, 每卡相同

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """Column 并行: 按 output 维(weight 的 dim=0/行)切分。
    每卡只持有 output/tp 行, 算出不同的输出列, 无需通信。
    用途: 独立的输出投影(配合后续 Row 并行)。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 本卡只持有 output/tp 行 → 实际 output_size 缩小; tp_dim=0
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载时从完整权重里截取本卡负责的行。"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)              # 本卡持有的行数
        start_idx = self.tp_rank * shard_size                  # 本卡在原始权重的起点
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)   # 截取
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 普通线性, 各卡算出各自的输出列, 无需通信
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并的 Column 并行: 多段(如 gate/up)拼接成一个权重, 按 output 维切。
    每段在每卡上都占 output_size/tp 行, 加载时需按 shard_id 定位到正确的段。
    用途: gate_up_proj (gate 与 up 合并)。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],     # 各段大小, 例如 gate/up: [I, I]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes                        # 记录各段大小
        super().__init__(input_size, sum(output_sizes), bias)   # 总 output = 各段之和

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """loaded_shard_id 指明是第几段(0=gate, 1=up)。"""
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size   # 本段在本卡的起点
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size           # 本段在本卡的行数
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]  # 取本卡的片段
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV 合并的 Column 并行 (GQA: Q 段比 KV 段大)。
    输出布局: [Q段 | K段 | V段], 每段内再按 tp 切。
    加载时按 loaded_shard_id("q"/"k"/"v") 定位段, 走与 Merged 类似的逻辑。
    用途: qkv_proj。
    """

    def __init__(
        self,
        hidden_size: int,          # H
        head_size: int,            # D
        total_num_heads: int,      # 全局 Q 头数
        total_num_kv_heads: int | None = None,   # 全局 KV 头数(GQA)
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)          # 本卡 Q 头数 hₗ
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)    # 本卡 KV 头数 kvₗ
        # 总输出 = (Q + K + V) * D, 本卡通过 Column 基类再切到 (hₗ + 2*kvₗ)*D
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """loaded_shard_id: "q" / "k" / "v", 定位到本卡的对应段。"""
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size          # Q 段大小 = hₗ*D
            shard_offset = 0                                      # Q 段在最前
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size       # K 段大小 = kvₗ*D
            shard_offset = self.num_heads * self.head_size        # K 段在 Q 段之后
        else:  # v
            shard_size = self.num_kv_heads * self.head_size       # V 段大小 = kvₗ*D
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size  # V 段在 K 之后
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """Row 并行: 按 input 维(weight 的 dim=1/列)切分。
    每卡只持有 input/tp 列, 各卡算出"部分和", 必须 all_reduce 才能得到完整结果。
    用途: o_proj, down_proj (与 Column 配对, 全层只需一次 all_reduce)。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 本卡只持有 input/tp 列 → 实际 input_size 缩小; tp_dim=1
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:        # bias 是 1D, 直接整块拷
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bias 只在 rank0 加, 避免 all_reduce 后 bias 被重复累加 tp 次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)           # 各卡部分和求和 → 完整结果
        return y
