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

---

# 基准指标笔记：TTFT / ITL / TPOT

LLM serving 延迟的三个常用时间指标（单位通常是 ms）：

| 记号 | 全称 | 含义 |
|------|------|------|
| **TTFT** | Time To First Token | 发出请求 → **第一个**输出 token 到达 |
| **ITL** | Inter-Token Latency | 相邻两个输出 token 之间的时间间隔（tokenᵢ → tokenᵢ₊₁） |
| **TPOT** | Time Per Output Token | 输出阶段**平均**每 token 耗时（常接近 ITL 的均值） |

关系简述：

- decode 每产出一个新 token 打一次点，相邻两次的时间差 = 一条 **ITL**；ITL 是逐步序列，可看 P50/P90/P99。
- **TPOT** 是整段生成的平均；**ITL 分布**更能暴露抖动。
- **TTFT** 主要反映排队 + prefill；**ITL/TPOT** 主要反映 decode。

长尾 ITL（P99）高，常见原因：prefill 抢占 decode、调度抖动、KV 分配/抢占、CPU 侧打包气泡等（对应计划里 1.3 报 ITL P99、2.4 看调度饿死 decode）。

---

# 基准指标笔记：P50 / P90 / P99

延迟（或吞吐相关数值）的**百分位数**：把所有样本从小到大排序后，落在第 50% / 90% / 99% 位置上的值。

| 记号 | 含义 | 直观理解 |
|------|------|----------|
| **P50** | 中位数 | 一半请求比这快，一半比这慢；看「典型体验」 |
| **P90** | 第 90 百分位 | 90% 的请求 ≤ 这个值；开始暴露偏慢的尾巴 |
| **P99** | 第 99 百分位 | 最慢那 1% 的门槛；看「长尾 / SLO 是否炸」 |

举例：若 TTFT 的 P50=20ms、P90=80ms、P99=200ms，说明多数请求很快，但仍有约 1% 会到 200ms 量级。

在本仓库 baseline（TTFT / TPOT / ITL）里：

- 看吞吐优化 → 常盯 **P50**
- 看服务稳定性、调度饿死 decode → 更看 **P99**（以及极差）

---

# 2.6 笔记：宿主侧 decode 打包消除（面试向）

## 一句话

> CUDA Graph 只消掉了 **GPU kernel launch**；decode 每步在 CPU 上 `torch.tensor` + pin + H2D 的元数据打包仍在同步路径上。2.6 用**常驻 pinned buffer + NumPy 打包 + 直接写入 graph 输入**，把这条 CPU 气泡压掉。

## 问题从哪来

小 batch decode 时 GPU 前向只要 ~1–2ms，CPU 侧「组 input_ids/positions/slots/block_tables」的占比很高。原路径每步：

1. Python list 收集字段  
2. `torch.tensor(..., pin_memory=True)` → 可能触发 host 分配  
3. `.cuda()` 得到**临时** GPU tensor  
4. 再 copy 进 CUDA Graph 的持久 buffer（二次拷贝）

Graph 优化了 launch，**没优化第 2–4 步**。

## 做法（对齐上游 PR #253）

- 新增 `DecodeInputBatch`：启动时分配一组 pin_memory CPU 张量  
- 零拷贝 `.numpy()` 视图上写元数据（避免 PyTorch 逐元素赋值的 dispatch）  
- `copy_` 进 capture 时绑定的 `graph_vars`（无临时 GPU 元数据、无二次 D2D）  
- Prefill / eager / bs>512 仍走旧路径

## 受益怎么讲（面试口径）

| 维度 | 内容 |
|------|------|
| **机制收益** | 稳态 decode：**不反复** `torch.tensor`/cudaHostAlloc；pinned `data_ptr` 跨 step 稳定；打包 CPU 更轻 |
| **可观测数字** | 本机 bs1/8/64：thr 约 +0.4%～+1.6%，step_p50 约 −0.7%～−1.3%；**落在噪声边**（bs1 极差 ≥ 收益） |
| **证据分层** | 主证据 = 架构/profiler 叙事（上游 nsys：稳态零 cudaMalloc）；辅证据 = 小幅正向 latency；**不把 +1% 当主 KPI** |
| **和 2.4/2.1 关系** | 2.4 解决「谁先跑」；2.6 解决「decode 步 CPU 打包税」；若再做 2.1 overlap，须先 2.6 才能单独归因 |

## 面试 30 秒稿

「vLLM 类引擎开了 CUDA Graph 后，小 batch decode 的瓶颈常回到 **CPU 组 batch**。我做了常驻 pinned + NumPy 打包，直接写入 graph 输入，去掉每步临时 tensor。正确性指纹对齐；吞吐只有几个点且在噪声边，所以面试我强调 **profiler/分配语义**，而不是报虚高加速比。」

---

# 2.4 笔记：调度策略消融（面试主戏）

> 完整演示卡见 `DEMO_2.4.md`。证据：`results/opt_2.4_schedule.json`。

## 一句话

> 默认 **prefill 绝对优先** 会在混合负载下饿死 decode（ITL 长尾）。用 `decode_priority` / `token_budget` 画出 **TTFT ↔ ITL/吞吐** 权衡，而不是追求峰值 tok/s。

## 背表（三次中位数，mixed 16+16）

| 策略 | thr | LP TTFT | LD ITL p99 |
|------|-----|---------|------------|
| prefill_priority | 4453 | 145 | **5.07** |
| decode_priority | 4123 (−7%) | 145 | **3.50 (−31%)** |
| budget 256→4096 | 1976→4397 | 736→97.5 | ~4–5 |

## 30 秒稿

「continuous batching 下 waiting 有 prefill 就不跑 decode，长 prompt 一来短请求 ITL 就抖。我加了策略开关：decode 优先把 LD ITL p99 从 5.1 打到 3.5，吞吐只掉几个点；再扫 token budget 给面试官看 TTFT–吞吐旋钮。主证据是这张消融表，不是单一加速比。」

---

# 2.2 笔记：n-gram / prompt-lookup 投机解码

## 一句话

> 用请求自身历史做 n-gram 续写当 draft，一次（逐条）前向验证；贪心接受前缀 + bonus。重复文本接受率高；并行 verify 与逐步 AR 在长生成上可能分叉。

## 机制（面试口径）

1. **Draft**：在 `token_ids` 里找与后缀匹配的 n-gram，取其后续 ≤K 个 token  
2. **Verify**：输入 `[last]+drafts`，prefill 式变长前向，LMHead 保留全部位置 logits  
3. **Accept**：贪心下 draft[i]==argmax 则接受；否则在拒绝位用 argmax（bonus）  
4. **实现**：逐条 verify（不 pack 多 seq）；KV 只 commit `last+accepted`；T>0 回退普通 decode

## 数字（本机）

| 项 | 值 |
|----|-----|
| 重复文本 accept rate | **~0.89** |
| warm tok/s | ~198 → ~227（约 +15%；首跑冷启动极差大） |
| 正确性 | vs 关 ngram：`max_tokens≤40` exact-match；HF align：**≤40** 100%，64 可能在 ~40+ 分叉 |

## 30 秒稿

「投机解码我先做不需草稿模型的 n-gram：历史匹配提案，一次前向验证，讲清接受前缀和拒绝位重采样。重复文本接受率近 90%；并说明 paged KV 多 token verify 与逐步 AR 长生成可能分叉，对齐验收卡在 40 token。」

---

# 2.7 笔记：Hybrid P+D + 相关概念

> 证据：`results/opt_2.7_hybrid.json` · 复现：`python experiments/ablate_schedule.py --save opt_2.7_hybrid`
> 单测：`tests/test_hybrid_schedule.py` · 对齐：默认 `experiments/align_hf.py` + hybrid 并发 4×64

## 一句话

> Continuous batching **已经有了**；2.7 不是「从零做 CB」，而是把 2.4 的「逐步纯 P 或纯 D 轮转」
> 升级成 **同一步 decode + chunked prefill**（Sarathi stall-free / 接近 vLLM v1）。

## 概念辨析

### Continuous batching vs Hybrid P+D

| 概念 | 本仓库状态 | 含义 |
|------|------------|------|
| Continuous batching | ✅ 原有 | 多请求按 step 动态拼批，生命周期独立（waiting/running） |
| 2.4 调度策略 | ✅ | 每步仍纯 P 或纯 D，只改「谁先跑」 |
| 2.7 hybrid | ✅ 新增 | **同一步**里既有 decode 又有 prefill chunk |

### Prefill / Decode 因果 vs「decode 优先」

- **单条请求**：必须先 prefill（或 chunk 推进完）才能 decode——因果顺序不变。
- **`decode_priority` 实质**：管的是 **跨请求** 这一步 GPU 先伺候谁。
  - running 里已有人在 decode → 先跑他们的 decode；
  - waiting 里新来的长 prefill 先让路。
  - **不是**「跳过 prefill 直接 decode」。

### `token_budget_512` 含义

- 策略 = `token_budget`，`prefill_token_budget=512`。
- 每步若跑 prefill，最多 512 个 prompt token；与 decode **轮转**（一步仍纯 P 或纯 D）。
- 消融名 `token_budget_512` 只是扫预算时的标签。

### Hybrid vs token_budget（同预算对照）

| | token_budget_512 | hybrid_512 |
|--|------------------|------------|
| 预算 | 每步 prefill ≤512 | 同样 ≤512 |
| 一步形态 | 纯 P **或** 纯 D（轮转） | **同一步** decode + prefill chunk |
| 对照意义 | 控制变量只有「是否同一步混 batch」 | 2.7 的正确基线 |

### varlen 是什么

- **variable length（变长）**：一批内各序列 query 长度可以不同。
- 用 `cu_seqlens_q/k` 前缀和切开拍平 token；hybrid/prefill 走 `flash_attn_varlen_func`。
- 纯 decode 的 `flash_attn_with_kvcache` + CUDA Graph 假设「每行约 1 token」——混合批不能走这条。

## 背表（本机三次中位数，mixed 16+16）

| 策略 | thr | LP TTFT p50 | LD ITL p99 |
|------|-----|-------------|------------|
| prefill_priority | 4354 | 148 | 5.2 |
| decode_priority | 4040 | 148 | **3.5** |
| token_budget_512 | 2882 | 362 | 4.7 |
| **hybrid_512** | **3104** | **320** | **4.8** |

读表要点：

- **正确对照** = `token_budget_512`：hybrid 吞吐约 +7.7%、TTFT 更好、ITL 持平（增益 > 极差，可信）。
- **不是全面碾压**：decode_priority 仍是 ITL 最优；prefill_priority 仍是峰值吞吐最高。
- 面试主证据 =「同一步混 batch 机制 + 与轮转对照」；不要对 `baseline_v0` 报调度加速比。

## 30 秒稿

「2.4 证明谁先跑改变 ITL 长尾；token_budget 是限预算轮转，一步仍纯 P 或纯 D。
2.7 同一步先 decode 再塞 prefill chunk，走 varlen+block_table。相对同预算轮转吞吐和 TTFT 更好、ITL 不恶化——这才是 stall-free；不是跳过单条的 prefill。」

---

# 2.7 各文件改动的意义

| 文件 | 改了什么 | 意义（为什么必须动） |
|------|----------|----------------------|
| `nanovllm/config.py` | `schedule_policy` 允许 `"hybrid"`；budget 注释扩到 hybrid | 对外旋钮：不改配置层就无法选策略、也无法做消融扫点 |
| `nanovllm/engine/scheduler.py` | `_try_schedule_prefill(max_seqs=)`；`hybrid` 分支先 decode 再填 prefill；`postprocess` 按 per-seq `cached < num_tokens` 决定是否 append | **调度语义核心**：同一步拼出混合 seq 列表；未完成 chunk 不能误 append，否则污染生成 |
| `nanovllm/engine/model_runner.py` | 新增 `prepare_hybrid`；`run` 识别混合批 → varlen eager，禁 graph | **执行路径核心**：混合长度只能走 varlen+block_table；走 decode graph 会形状错误 |
| `nanovllm/engine/llm_engine.py` | hybrid/`is_prefill=True` 不进 ngram；postprocess 前按 P/D 计 `num_tokens` | ngram verify 与混合批假设冲突；进度条/吞吐符号依赖计数约定 |
| `experiments/ablate_schedule.py` | 增加 `hybrid_512` 点；统一 `enable_ngram=False`；默认 save `opt_2.7_hybrid` | 量化验收：同 mixed 负载出对照表，关 ngram 避免混淆归因 |
| `tests/test_hybrid_schedule.py` | 断言同一步可含 P+D；纯 decode 时 `is_prefill=False` | 无 GPU 的调度契约回归，改调度时先挡回归 |
| `results/opt_2.7_hybrid.json` | 三次中位数消融落盘 | 面试/台账证据本体 |
| `IMPROVEMENT_PLAN.md` / `timeline.md` | 台账、M2.5 完成、下一跳 3.1 | 项目叙事与日历，不参与运行时 |

**明确没改的**：`attention.py` 仍用全局 `is_prefill`——hybrid 步一律设 `True` 走 varlen 分支即可，不必第三套 attention API。

---

# 3.1 笔记：INT8 KV Cache（原理 + 做法）

> 证据：`results/opt_3.1_int8_kv.json`（blocks/align）· `opt_3.1_baseline_{auto,int8}_eager.json`（吞吐）  
> 复现：`python experiments/ablate_int8_kv.py` · `python experiments/align_hf.py --kv-cache-dtype int8`  
> 吞吐对照：`python experiments/baseline.py --suites throughput latency mixed --repeats 3 --enforce-eager --kv-cache-dtype {auto,int8} --gpu-memory-utilization 0.85`

## 一句话

> KV 用 **int8 + group-wise scale** 常驻 HBM；attention 前把本步用到的 block **反量化成 bf16** 再进现有 FlashAttention。主收益是同显存 **更多 KV blocks**（本机 ×1.78），不是追求 tok/s；第一版因 gather-dequant 税，eager 对照吞吐约 **−60%～−78%**（见下文「为何变慢」）。

## 原理

### 为什么动 KV

Decode 每步要读整段历史 K/V。并发↑、上下文↑时，**KV 显存**先顶满。权重加载一次；KV 随 token 线性涨。bf16→int8 元素体积约减半，扣掉 scale 后块数仍可接近翻倍。

### 对称量化（group-wise）

对每个 token、每个 KV head，把 `head_dim` 按 **G=16** 分组。组内向量 \(x\)：

1. \(s = \max(|x|) / 127\)
2. \(q = \mathrm{round}(x/s)\) 夹到 \([-128,127]\) → `int8`
3. 反量化：\(\hat{x} = q \cdot s\)

比「整段 D 一个 scale」或「整 head 一个 scale」更细，argmax 更稳。本机：相对 auto **3/4** 条 64-token 贪心 exact-match；prompt0 在 token 5 分叉——**不是 bit-exact**，面试要主动说。

### 与 FA 的合同

现有 FA **不直接吃 int8 paged KV**，所以：

| 阶段 | 行为 |
|------|------|
| 写 | 本步 bf16 K/V → 量化 → int8 cache + fp16 scale |
| 读（decode / prefix / hybrid） | 收集 `block_tables` 里的物理块 → dequant 到 workspace → **重映射** block 下标 → FA |
| 纯 prefill（无 block_table） | 仍用本步 bf16 K/V 算 attention，只额外量化写入 |

### 为何 INT8 不是 FP8 attn

4090 = Ada **SM89**；很多 FP8 attention / FlashInfer 路径面向 **SM90**。本方案：**INT8 只负责存**，算子仍是成熟 bf16 FA。

### 为何关 CUDA Graph

Graph 预捕获假设固定 bf16 paged buffer；int8 + 动态 workspace 重映射与之不兼容。第一版：`kv_cache_dtype=int8` → `use_cudagraph=False`。

## 背表（本机 gpu_util=0.85）

| 项 | auto | int8 |
|----|------|------|
| `num_kvcache_blocks` | 655 | **1166（×1.78）** |
| HF greedy 4×64 | 100% | **3/4**（prompt0 @5 分叉） |
| hybrid 冒烟 | — | 通过 |
| 调度默认 | prefill_priority | 同左（3.1 不扫四种策略） |

## 为何变慢（吞吐对照，eager 公平对比）

> 证据：`results/opt_3.1_baseline_auto_eager.json` vs `opt_3.1_baseline_int8_eager.json`  
> 条件：两边都 `--enforce-eager`（去掉 Graph 混杂）、`gpu_util=0.85`、三次中位数。

| 套件 | 指标 | auto | int8 | 变化 |
|------|------|------|------|------|
| throughput | tok/s | 4436 | 993 | **−78%** |
| throughput | TPOT p50 | 27 ms | 170 ms | +525% |
| latency bs1/8/64 | tok/s | 71 / 518 / 3974 | 27 / 205 / 1598 | **约 −60%** |
| mixed | tok/s | 1148 | 460 | **−60%** |

**结论：3.1 第一版用容量换正确路径，没有带来 tok/s 提升；变慢是机制税，不是测噪。**

### 根因（按贡献排序）

1. **读侧 gather-dequant 同步税（主因）**  
   FA 只吃 bf16。每步 decode/prefix/hybrid 都要：`unique(block_ids)` → 整块 int8→bf16 反量化 → 重映射 `block_tables` → 再调 FA。  
   0.6B 上单步 FA 本身很快，这段 PyTorch 反量化/拷贝占比被放大，ITL/TPOT 明显变差。

2. **写侧 group-wise 量化**  
   每次 store 要算 amax、round、写 int8+scale（相对原 bf16 直写更重）。通常轻于读侧，但是固定附加成本。

3. **CUDA Graph 关闭（产品路径上会再叠一层）**  
   int8 与动态 workspace 不兼容 capture。上面对照已双方 eager，**表内 −60%～−78% 主要不是「少了 Graph」**，而是 dequant 路径本身。  
   若拿默认 auto（开 Graph）去比 int8，差距还会更大——面试要分开说。

4. **硬件/模型尺度**  
   4090 + 0.6B 算力轻、带宽宽：省 KV 流量的收益盖不过额外 kernel/拷贝。计划里也写过「本机 tok/s 可能远小于 3090 报道的 +22%」。主 KPI 仍应是 **blocks ×1.78**，不是加速比。

### 面试怎么说

> 「INT8 KV 第一版验证的是存储合同和显存账：同利用率 blocks 约 1.8×，对齐 3/4。吞吐掉一截是因为 FA 前同步 gather-dequant，小模型上这笔税很显眼。下一步若要补吞吐故事，就该融 dequant / 减整块拷贝，而不是否认这个数字。」

## 做法（代码锚点）

1. `Config.kv_cache_dtype ∈ {auto, int8}`
2. `allocate_kv_cache`：按 int8 元素 + group scale 算 `block_bytes` → 块数↑；注入 `k/v_cache`、`k/v_scale`
3. `store_kvcache_int8`：向量化 per-group 量化写入
4. `dequant_active_kv`：`torch.unique` 活跃块 → workspace → remap `block_tables`
5. `experiments/ablate_int8_kv.py`：落盘 blocks 比与 align 率
6. `experiments/baseline.py --kv-cache-dtype`：吞吐/延迟对照（见上表）

## 30 秒稿

「Decode 显存墙我做了 INT8 paged KV：group-wise 动态量化写入，读前反量化进 FA，4090 不上 FP8 attn。同利用率 blocks 约 1.8 倍，对齐 3/4。吞吐第一版会掉——主因是每步 gather-dequant 税，小模型上很明显；故事主轴是容量合同，吞吐优化是明确的下一步。」
