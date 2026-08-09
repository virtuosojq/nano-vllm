# 阶段 1 笔记：为什么 `LLM` 类只有一个 `pass`？

## 代码

```python
# nanovllm/llm.py
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

`LLM` 是空壳，所有逻辑（`generate` / `add_request` / `step`）都在父类 `LLMEngine`。

## 设计原因

1. **关注点分离**：`LLMEngine` = 引擎实现（调度/序列/TP/KV cache），`LLM` = 对外接口名。
2. **便于引擎层复用**：内部模块可直接引用 `LLMEngine`，避免反向依赖对外 API。
3. **为扩展留口子**：未来可在 `LLM` 上加用户便利方法（`chat()` / `stream_generate()`），不污染核心。
4. **符合 vLLM 惯例**：复刻 vLLM 的命名（用户熟悉的 `LLM`），实现放在 `LLMEngine`。
5. **测试友好**：可独立操作 `LLMEngine` 做测试/mock。

## 反面对比

把所有逻辑塞进 `LLM` 会导致：类过大、内部方法与用户接口混杂、引擎无独立名字可被引用。

## 一句话总结

> `LLM` 是门面，`LLMEngine` 是引擎。`pass` 表示门面暂未做额外包装，纯粹借名给引擎用——**门面模式（Facade）** 的简化形态。

---

# 阶段 7 笔记：Qwen3 模型整体输入输出结构

## 符号约定

| 符号 | 含义 |
|------|------|
| `N` | 当前 step 拍平后的 token 总数。prefill = 多条序列 prompt 拼接总长；decode = 序列数 S（每条出 1 token） |
| `S` | 当前 batch 内的序列条数 |
| `H` | `hidden_size` |
| `D` | `head_dim` |
| `tp` | `tensor_parallel_size`（张量并行 world size） |
| `hₗ` | 本卡 Q 头数 = `num_attention_heads // tp` |
| `kvₗ` | 本卡 KV 头数 = `num_key_value_heads // tp`（GQA） |
| `I` | `intermediate_size`（MLP 中间维） |
| `V` | `vocab_size` |
| `L` | `num_hidden_layers`（decoder 层数） |

> 下面每个张量后都标注形状；带 `ₗ` 的量表示已被 TP 切分到单卡。

## 顶层调用链（自上而下）

```
ModelRunner.run(seqs, is_prefill)
  ├─ prepare_prefill / prepare_decode   → 产出 input_ids, positions + 写入 Context
  ├─ run_model(input_ids, positions)
 │     ├─ model = Qwen3ForCausalLM(input_ids, positions)  → hidden_states: [N, H]
 │     └─ model.compute_logits(hidden_states)             → logits: [S, V]
  └─ sampler(logits, temperatures)                        → token_ids: [S]
```

模型 forward 只显式接收 `input_ids / positions`；其余 KV cache 相关张量通过全局 `Context` 注入，避免层层透传。

## 1. Qwen3ForCausalLM（[qwen3.py L186-216](file:///home/jq/infraTech/nanovllm/nanovllm/models/qwen3.py#L186-L216)）

### forward
- 输入
  - `input_ids`: `[N]` int64（所有 token 拍平）
  - `positions`: `[N]` int64（每个 token 的绝对位置）
- 输出
  - `hidden_states`: `[N, H]`

### compute_logits
- 输入：`hidden_states: [N, H]`
- 内部走 `ParallelLMHead`
- 输出：
  - prefill：取每条序列最后一个 token 的 hidden（`cu_seqlens_q[1:]-1`），`logits: [S, V]`
  - decode：`[S, H] → [S, V]`（rank0 gather 所有卡的词表分片，rank>0 返回 None）

## 2. Qwen3Model（[qwen3.py L162-183](file:///home/jq/infraTech/nanovllm/nanovllm/models/qwen3.py#L162-L183)）

- 输入：`input_ids [N]`, `positions [N]`
- 步骤
  1. `embed_tokens(input_ids)` → `[N, H]`（VocabParallel，按词表行切，mask + all_reduce）
  2. `residual = None`
  3. 对 L 层循环：`hidden, residual = layer(positions, hidden, residual)`
  4. 最终 `norm(hidden, residual)`（融合 add+norm）
- 输出：`hidden_states: [N, H]`

## 3. Qwen3DecoderLayer（[qwen3.py L120-159](file:///home/jq/infraTech/nanovllm/nanovllm/models/qwen3.py#L120-L159)）

pre-norm + 双残差结构。两层 Norm 都走融合残差路径（除了第一层首次）。

```
输入: positions [N], hidden_states [N, H], residual [N,H] | None

第 1 层（residual is None）:
    hidden = input_layernorm(hidden)          # [N,H]
    residual = 原始 hidden                     # [N,H]
后续层:
    hidden, residual = input_layernorm(hidden, residual)   # add+norm 融合

hidden = self_attn(positions, hidden)         # [N,H] → [N,H]
hidden, residual = post_attention_layernorm(hidden, residual)
hidden = mlp(hidden)                          # [N,H] → [N,H]
输出: (hidden_states [N,H], residual [N,H])
```

> **融合残差**：`add_rms_forward` 把 `x + residual` 和归一化合并为一步，减少一次显存往返（[layernorm.py L29-40](file:///home/jq/infraTech/nanovllm/nanovllm/layers/layernorm.py#L29-L40)）。

## 4. Qwen3Attention（[qwen3.py L14-88](file:///home/jq/infraTech/nanovllm/nanovllm/models/qwen3.py#L14-L88)）

```
输入: positions [N], hidden_states [N, H]

(1) qkv = qkv_proj(hidden)                    # QKVParallelLinear (Column, dim=0)
    # 输出按本卡切分: [N, (hₗ + 2*kvₗ)*D]
    q = q[:, : hₗ*D]        → [N, hₗ, D]
    k = q[:, hₗ*D : (hₗ+kvₗ)*D] → [N, kvₗ, D]
    v = q[:, (hₗ+kvₗ)*D :]  → [N, kvₗ, D]

(2) q_norm(q), k_norm(k)     # Qwen3 特有，在 head_dim 维 RMSNorm (仅 qkv_bias=False)

(3) q, k = rotary_emb(positions, q, k)        # RoPE，复用 cos_sin_cache
    # q: [N, hₗ, D], k: [N, kvₗ, D]

(4) o = attn(q, k, v)        # 见 Attention 算子
    # 输出: [N, hₗ, D]

(5) output = o_proj(o.flatten(1,-1))          # RowParallelLinear (dim=1)
    # [N, hₗ*D] → [N, H]   (内部 all_reduce 聚合)

输出: [N, H]
```

关键尺寸（单卡）：
- `q_size = hₗ * D`
- `kv_size = kvₗ * D`
- QKV 合并权重输出 `(total_h + 2*total_kv) * D`，切到单卡后 `(hₗ + 2*kvₗ) * D`

## 5. Attention 算子（[attention.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/attention.py)）

通过 `get_context()` 拿到 `slot_mapping / block_tables / cu_seqlens / context_lens`。

```
输入: q [N, hₗ, D], k [N, kvₗ, D], v [N, kvₗ, D]

(a) 若 k_cache 非空:
      store_kvcache(k, v, k_cache, v_cache, slot_mapping)
      # Triton kernel：按 slot_mapping 把 K/V 散写到 paged cache
      # slot_mapping[i] == -1 时跳过（prefix cache 命中）

(b) prefill 分支:
      若 block_tables is not None (prefix cache): k,v ← k_cache,v_cache
      o = flash_attn_varlen_func(q, k, v,
              cu_seqlens_q, cu_seqlens_k,
              max_seqlen_q, max_seqlen_k,
              causal=True, block_table=block_tables)
      → [N, hₗ, D]    # 变长：cu_seqlens 分隔多条不等长序列

    decode 分支:
      o = flash_attn_with_kvcache(q.unsqueeze(1),    # [S,1,hₗ,D]
              k_cache, v_cache,
              cache_seqlens=context_lens,            # [S]
              block_table=block_tables)              # [S, max_blocks]
      → [S, hₗ, D]

输出: [N, hₗ, D]  (decode 时 N = S)
```

paged cache 形状（由 `allocate_kv_cache` 分配）：
- `k_cache / v_cache`: `[num_blocks, block_size, kvₗ, D]`（每层一份）

## 6. Qwen3MLP（[qwen3.py L91-117](file:///home/jq/infraTech/nanovllm/nanovllm/models/qwen3.py#L91-L117)）

```
输入: x [N, H]

gate_up = gate_up_proj(x)     # MergedColumnParallelLinear (Column, dim=0)
                              # [N, H] → [N, 2*I/tp]  (gate 与 up 拼接)
x = SiluAndMul(gate_up)       # → [N, I/tp]
                              # silu(gate) * up
x = down_proj(x)              # RowParallelLinear (dim=1)
                              # [N, I/tp] → [N, H]   (all_reduce)

输出: [N, H]
```

## 7. Linear 层 TP 策略一览（[linear.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/linear.py)）

权重 PyTorch 约定 `weight: [output, input]`。

| 类 | 用途 | 切分维 | 本卡权重形状 | 通信 | forward 输出 |
|----|------|--------|--------------|------|--------------|
| `QKVParallelLinear` | qkv_proj | output (dim=0) | `[(hₗ+2*kvₗ)*D, H]` | 无 | `[N, (hₗ+2*kvₗ)*D]` |
| `MergedColumnParallelLinear` | gate_up_proj | output (dim=0)，两段拼接 | `[2*I/tp, H]` | 无 | `[N, 2*I/tp]` |
| `RowParallelLinear` | o_proj / down_proj | input (dim=1) | `[H, I/tp]` 或 `[H, hₗ*D]` | all_reduce | `[N, H]` |
| `ReplicatedLinear` | （未用） | 不切 | `[output, input]` | 无 | — |

经典 Megatron 配对：`Column → Row`，全层只需一次 all_reduce（在 RowParallel 内）。bias 仅 rank0 加（避免 all_reduce 重复）。

## 8. Embedding / LMHead（[embed_head.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/embed_head.py)）

**VocabParallelEmbedding**（`embed_tokens`）
- 权重 `[V/tp, H]`，按词表行切
- forward：mask 掉非本卡区间 → embedding → mask 置零 → all_reduce
- `[N] → [N, H]`

**ParallelLMHead**（`lm_head`）
- 继承 VocabParallelEmbedding，权重 `[V/tp, H]`
- forward：
  - prefill：`hidden[cu_seqlens_q[1:]-1]` 取每条最后 token → `[S, H]`
  - `F.linear(x, weight)` → `[S, V/tp]`
  - `dist.gather` 到 rank0 → `[S, V]`（rank>0 为 None）

## 9. Sampler（[sampler.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/sampler.py)）

- 输入：`logits [S, V]`, `temperatures [S]`
- 步骤：`logits / T → softmax → / Exp(1) → argmax`（Gumbel-max 变体，等价多项式采样）
- 输出：`token_ids [S]`（每条序列一个 token）

## 10. Context：模型层与 runner 的桥梁（[context.py](file:///home/jq/infraTech/nanovllm/nanovllm/utils/context.py)）

`ModelRunner.prepare_*` 计算好下列张量后 `set_context(...)`，模型层 `get_context()` 读取：

| 字段 | 形状 | 用途 | 阶段 |
|------|------|------|------|
| `is_prefill` | bool | 选 flash attn 分支 | 两者 |
| `cu_seqlens_q` | `[S+1]` | varlen 累积序列长度（query） | prefill |
| `cu_seqlens_k` | `[S+1]` | varlen 累积序列长度（key） | prefill |
| `max_seqlen_q/k` | int | flash attn 内核优化提示 | prefill |
| `slot_mapping` | `[N]` | 每个 token 的 KV 写入槽位（`block_id*block_size+off`）；-1 = cache 命中跳过 | 两者 |
| `context_lens` | `[S]` | decode 时每条序列当前长度 | decode |
| `block_tables` | `[S, max_blocks]` | 每条序列引用的物理块 ID 列表 | 两者（prefill 仅 prefix cache 时非 None） |

## end-to-end 形状流（prefill 一览）

```
input_ids [N]
   │ embed_tokens (VocabParallel + all_reduce)
   ▼
hidden [N, H]  ──► Layer×L ──► hidden [N, H]
                     │
                     └─ DecoderLayer:
                          input_norm  → [N,H]
                          attn:
                             qkv_proj → [N,(hₗ+2kvₗ)D]
                             split/norm/rope → q[N,hₗ,D], k/v[N,kvₗ,D]
                             store_kvcache (Triton)
                             flash_varlen → o[N,hₗ,D]
                             o_proj (Row + all_reduce) → [N,H]
                          post_norm (add+norm)
                          mlp:
                             gate_up_proj → [N,2I/tp]
                             SiluAndMul → [N,I/tp]
                             down_proj (Row + all_reduce) → [N,H]
   │ norm
   ▼
hidden [N, H]
   │ compute_logits (LMHead: 取每条末 token + gather)
   ▼
logits [S, V]
   │ sampler (温度 + Gumbel-max)
   ▼
token_ids [S]
```

decode 阶段 `N = S`（每序列 1 token），attention 走 `flash_attn_with_kvcache` + `block_tables/context_lens`，其余形状一致。

---

## 阶段 7 源码阅读顺序（按调用链）

> 所有源码文件已添加逐行注释，建议按下面这条调用链顺序阅读，从入口逐层往下追。

| 序号 | 文件 | 作用 | 重点 |
|------|------|------|------|
| 1 | [qwen3.py](file:///home/jq/infraTech/nanovllm/nanovllm/models/qwen3.py) | 调用链总入口 | 先看 `Qwen3ForCausalLM` → `Qwen3Model` → `Qwen3DecoderLayer` → `Qwen3Attention` / `Qwen3MLP` |
| 2 | [context.py](file:///home/jq/infraTech/nanovllm/nanovllm/utils/context.py) | 全局 Context 桥梁 | 理解模型层如何拿到 KV cache 相关张量（slot_mapping/block_tables/cu_seqlens 等） |
| 3 | [embed_head.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/embed_head.py) | Embedding / LMHead | 入口处的词嵌入 + 出口处的 logits 投影（词表行切 + all_reduce/gather） |
| 4 | [linear.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/linear.py) | 四种 TP Linear | **核心重点**：Column/Row/Merged/QKV 的切分维度与通信策略 |
| 5 | [rotary_embedding.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/rotary_embedding.py) | RoPE 位置编码 | 只旋转 Q/K 不动 V；`lru_cache(1)` 单例共享 cos_sin_cache |
| 6 | [attention.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/attention.py) | Attention 算子 | **核心重点**：Triton kernel 写 paged cache + prefill/decode 两条 flash attn 分支 |
| 7 | [layernorm.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/layernorm.py) | RMSNorm | 融合残差 `add_rms_forward` 减少 1 次显存往返 |
| 8 | [activation.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/activation.py) | SwiGLU 激活 | `silu(gate) * up` 融合算子 |
| 9 | [sampler.py](file:///home/jq/infraTech/nanovllm/nanovllm/layers/sampler.py) | 采样器 | Gumbel-max 变体：`argmax(probs/Exp(1))` 等价多项式采样 |
| 10 | [loader.py](file:///home/jq/infraTech/nanovllm/nanovllm/utils/loader.py) | 权重加载 | 每个 Parameter 自带 `weight_loader` 钩子；`packed_modules_mapping` 处理 q/k/v、gate/up 合并 |

**一句话导览**：
- 想懂 **TP 并行** → 重点啃 `linear.py`（序号 4）
- 想懂 **Paged KV Cache 与 Attention** → 重点啃 `attention.py`（序号 6）
- 想懂 **数据如何注入模型** → 看 `context.py`（序号 2）
- 想懂 **权重如何加载与合并** → 看 `loader.py`（序号 10）
