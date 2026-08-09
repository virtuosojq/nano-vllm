import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """单层的自注意力子层。
    数据流: hidden[N,H] → qkv_proj → split/norm/rope → attn → o_proj → [N,H]
    TP 策略: qkv_proj 用 Column 并行(切输出), o_proj 用 Row 并行(切输入, all_reduce)。
    """

    def __init__(
        self,
        hidden_size: int,                # H: 模型隐藏维度
        num_heads: int,                  # 全局 Q 头数
        num_kv_heads: int,               # 全局 KV 头数(GQA, 通常 < num_heads)
        max_position: int = 4096 * 32,   # RoPE 预算的最大位置
        head_dim: int | None = None,     # 每头维度 D; None 则用 H/num_heads 推
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,          # Qwen3 默认无 qkv bias, 改用 q_norm/k_norm
        rope_theta: float = 10000,       # RoPE base
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()                          # 张量并行世界大小
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0               # Q 头数必须能被 tp 整除
        self.num_heads = self.total_num_heads // tp_size         # 本卡 Q 头数 hₗ
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0            # KV 头数也要能被 tp 整除
        self.num_kv_heads = self.total_num_kv_heads // tp_size   # 本卡 KV 头数 kvₗ
        self.head_dim = head_dim or hidden_size // self.total_num_heads   # D
        self.q_size = self.num_heads * self.head_dim             # 本卡 Q 维 = hₗ*D
        self.kv_size = self.num_kv_heads * self.head_dim         # 本卡 KV 维 = kvₗ*D
        self.scaling = self.head_dim ** -0.5                     # 注意力缩放因子 1/√D
        self.qkv_bias = qkv_bias

        # QKV 合并投影: Column 并行(切 output 维), 本卡输出 (hₗ+2*kvₗ)*D
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影: Row 并行(切 input 维), 内部 all_reduce 聚合
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,   # 输入维 = 全局 Q 总维
            hidden_size,                            # 输出维 = H
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)   # 兼容 scaling 配置里的 theta
        # RoPE: lru_cache 保证全局单例, 所有层共享同一份 cos_sin_cache
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 注意力算子: 封装 flash attn + paged KV cache 的读写
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3 特有: Q/K 在 head_dim 维做 RMSNorm(仅当无 qkv bias 时启用)
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,        # [N] 每个 token 的绝对位置
        hidden_states: torch.Tensor,    # [N, H]
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)                          # [N, (hₗ+2*kvₗ)*D]
        # 沿最后一维切成 q/k/v 三段(按本卡尺寸)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)               # [N, hₗ, D]
        k = k.view(-1, self.num_kv_heads, self.head_dim)            # [N, kvₗ, D]
        v = v.view(-1, self.num_kv_heads, self.head_dim)            # [N, kvₗ, D]
        if not self.qkv_bias:
            q = self.q_norm(q)                                       # Qwen3 特有 Q norm
            k = self.k_norm(k)                                       # Qwen3 特有 K norm
        q, k = self.rotary_emb(positions, q, k)                      # RoPE 只旋转 q、k, 不动 v
        o = self.attn(q, k, v)                                       # [N, hₗ, D] (内部读写 paged cache)
        output = self.o_proj(o.flatten(1, -1))                       # [N, hₗ*D] → [N, H] (all_reduce)
        return output


class Qwen3MLP(nn.Module):
    """前馈子层 (SwiGLU): silu(gate(x)) * up(x) 再 down 投影。
    数据流: [N,H] → gate_up_proj → SiluAndMul → down_proj → [N,H]
    TP 策略: gate_up_proj 用 Merged Column 并行, down_proj 用 Row 并行(all_reduce)。
    """

    def __init__(
        self,
        hidden_size: int,          # H
        intermediate_size: int,    # I
        hidden_act: str,           # 激活名
    ) -> None:
        super().__init__()
        # gate 与 up 合并投影: Merged Column 并行, 输出两段拼接 = 2*I
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,   # 两段相同大小, 分别对应 gate / up
            bias=False,
        )
        # down 投影: Row 并行, 需 all_reduce
        self.down_proj = RowParallelLinear(
            intermediate_size,     # 输入维 I (按 tp 切)
            hidden_size,           # 输出维 H
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()    # silu(gate) * up 的融合实现

    def forward(self, x):                              # x: [N, H]
        gate_up = self.gate_up_proj(x)                 # [N, 2*I/tp]
        x = self.act_fn(gate_up)                       # [N, I/tp]
        x = self.down_proj(x)                          # [N, H] (all_reduce)
        return x


class Qwen3DecoderLayer(nn.Module):
    """标准 pre-norm + 双残差 decoder 层。
    残差结构: x → norm → attn → +residual → norm → mlp → +residual
    特殊优化: 第一层首次 norm 不做 add(因为没有历史 residual), 后续层用 add+norm 融合。
    """

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 自注意力子层
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),   # 默认 True 表示"有 bias"→不启用 q_norm
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        # MLP 子层
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # 两个 RMSNorm: 进 attn 前一道, 进 mlp 前一道(pre-norm)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,       # [N]
        hidden_states: torch.Tensor,   # [N, H]
        residual: torch.Tensor | None, # [N, H] 或 None(第一层)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层首次: 只做 norm, 把原始 hidden 存为 residual 供后续层 add
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # 后续层: add + norm 融合(少一次显存往返)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)             # attn: [N,H]→[N,H]
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # add+norm
        hidden_states = self.mlp(hidden_states)                              # mlp: [N,H]→[N,H]
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """模型主体: 词嵌入 + N 层 decoder + 最终 norm。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # 词嵌入: 按词表行做 TP 切分(VocabParallel)
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # 堆叠 N 层 decoder
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终 norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,      # [N]
        positions: torch.Tensor,      # [N]
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)    # [N, H]
        residual = None
        for layer in self.layers:                        # 逐层
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)   # 最后一道 add+norm
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """最外层: 包含 model 主体 + lm_head。
    类属性 packed_modules_mapping 告诉 loader 如何把 HF 分开存的权重合并进 packed 参数。
    """

    # key = HF 权重名片段, value = (本模型参数名, 分片标识)
    # 例如 HF 的 q_proj → 塞进 qkv_proj 的 "q" 分片
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 绑定权重: tie_embedding 时 lm_head 与 embed_tokens 共享同一份权重
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,     # [N]
        positions: torch.Tensor,     # [N]
    ) -> torch.Tensor:
        return self.model(input_ids, positions)    # 只返回 hidden_states [N, H]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,    # [N, H]
    ) -> torch.Tensor:
        # 走 ParallelLMHead: prefill 时取每条序列最后一个 token, 再 gather 词表分片
        return self.lm_head(hidden_states)    # [S, V]
