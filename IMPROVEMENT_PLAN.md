# nano-vllm 面试项目改进路线

> 方法论：每一项改进都遵循 **baseline → 假设 → 实现 → `baseline.py --compare` 量化验证 → 消融** 的闭环，
> 结果记录到 `results/` 与本文件末尾的结果台账，面试时以表格呈现。
>
> 每一项都标注了**现状锚点**（代码里的具体位置），改进前先读对应代码，避免凭印象动工。

## 0. 基线（先跑这个）

```bash
python baseline.py --save baseline_v0     # 建立基线
# 每次改进后:
python baseline.py --save opt_xxx
python baseline.py --compare results/baseline_v0.json results/opt_xxx.json
```

基线指标：总吞吐 / prefill·decode tok/s / TTFT·TPOT (P50/P90/P99) / 前缀缓存加速比 / 正确性指纹。

**基线自身的两个已知局限**（见 1.3，建议与 P0 一起补）：

- 每套件只跑 1 次，无方差信息，<5% 的优化结论不可信；
- ~~所有请求第 0 秒同时入队（饱和压测），测不出排队行为~~ → 已由 `serving` 套件解决
  （2026-08：指数间隔开环注入 + TTFT/TPOT 分位数 + SLO goodput + injection_lag，
  参考上游 PR #29 并修正其泊松间隔误用、TTFT 判定依赖调度器隐含约定、无分位数三处缺陷；
  remaining：mixed 套件尚未加，prefill/decode 干扰仍测不到）。

## 1. P0 — 打地基（低成本，让后续结论可信）

### 1.1 贪心采样 + HF 逐 token 对齐

- **现状锚点**：`sampling_params.py:11` 直接 `assert temperature > 1e-10` 禁止贪心；
  `sampler.py` 的 Gumbel-max 路径对所有温度统一处理。当前 baseline.py 用 1e-6 温度近似，依赖浮点 argmax 稳定性。
- **做法**：`temperature == 0` 走 `logits.argmax(-1)`；同时补 `top_p` / `top_k`（SamplingParams 目前只有 temperature/max_tokens/ignore_eos 三个字段，采样能力本身也是缺口）。
- **验收**：与 HuggingFace `generate(do_sample=False)` 输出 exact-match = 100%（4 条固定 prompt × 64 token）；
  正确性从"指纹自洽"升级为"与权威实现对齐"。`top_p/top_k` 用 HF 采样结果对齐概率分布（固定 seed 对比）。

### 1.2 引擎内指标上报

- **现状锚点**：baseline.py 是外部观测（对 step 计时），拿不到引擎内部状态；
  `block_manager.py` 里 `can_allocate` 命中前缀时、`free_block_ids` 长度变化处都是天然的埋点位置。
- **做法**：暴露 per-request TTFT/TPOT、KV cache 利用率（`len(free_block_ids)/num_blocks` 随时间曲线）、prefix 命中率（命中 block 数 / 请求 block 数）。
- **验收**：baseline 报告新增 cache hit rate、KV utilization；prefix_cache 套件的加速比能和命中率互相印证。

### 1.3 baseline 统计严谨性补强

- **做法**：
  1. 每套件跑 3 次取中位数并记录极差（对齐 vLLM 基准惯例），`--compare` 只对中位数比较；
  2. `run_workload` 里逐 token 记录 ITL 分布（数据在 step 循环里已有，只是没存），报告 ITL P99；
  3. 新增 `mixed` 套件：短 decode + 长_prompt 混合负载，暴露调度问题（为 2.4 提供测量工具）。
- **验收**：同一代码跑两次 baseline，关键指标波动 < 2%；否则说明环境噪声大，需先锁频/排查。

## 2. P1 — 高性价比性能优化（建议至少做 2 个）

### 2.1 Scheduler 与 GPU 执行重叠（overlap scheduling）

- **现状锚点**：`llm_engine.py:49-55` 的 `step()` 是严格串行三段：CPU 调度（`scheduler.schedule()`，
  含 block 分配、xxhash 哈希、Python 循环）→ GPU 前向（`model_runner.call("run")`）→ `postprocess`。
  decode 步 batch 小、前向快（毫秒级），CPU 侧 Python 循环占比高，GPU 有明显气泡。
- **做法**：step N 的 GPU 前向异步发出后，CPU 立即做 step N+1 的调度与输入准备（双缓冲 + CUDA event 同步）。
  注意 `postprocess` 依赖本步采样结果，需要把"采样回传"与"下一步调度"解耦成流水线（vLLM v1 的做法）。
- **预期**：小 batch decode `step_time_ms.p50` 下降 10%~30%。
- **风险**：调度与执行重叠后，KV cache 分配决策发生在前向完成之前，预抢占路径（`scheduler.py:58-65`）需要重新推演正确性。

### 2.2 N-gram / Prompt-lookup 投机解码

- **现状锚点**：decode 每步每序列只产 1 token（`sampler.py` 返回 `[S]`）；
  `prepare_decode` 的输入构造按"每序列 1 token"假设写死，verify 路径需要改成每序列 k+1 token 的变长 prefill 式前向。
- **做法**：用请求自身的已生成内容做 n-gram 匹配草拟 k 个候选 token，一次前向批量验证（不需草稿模型，纯算法改动）。
- **预期**：代码/重复文本场景 decode 吞吐 +30%~80%；新增"接受率"指标（1.2 的埋点顺手记录）。
- **面试亮点**：vLLM/SGLang 都在用，可讲清 verify 的数学等价性（接受前缀 + 按概率重采样首个拒绝位）。

### 2.3 模型注册表 + Llama 移植（多模型支持第一步）

- **现状锚点**：`model_runner.py:9,31` 硬编码 `Qwen3ForCausalLM`；仓库唯一模型文件 `models/qwen3.py`。
- **参考实现**：上游 PR #95（`model_registry.py` 125 行 + `models/__init__.py`）、PR #188（`models/llama.py` 204 行 + `rotary_embedding.py` +78 行）。均未被合并，可整体借鉴。
- **做法（两步）**：
  1. **注册表**：新增 `models/model_registry.py`，按 `hf_config.architectures[0]` 查表（比 `model_type` 稳——密集/MoE/VL 的 architectures 名不同，天然分流，PR #251 也验证了这个方向）；`model_runner.py` 改为 `create_model(hf_config)`，仅 ~10 行改动。
  2. **Llama 移植**：新增 `models/llama.py`，以 PR #188 为底。与 Qwen3 的实际差异很小：无 q/k-norm（`qwen3.py:77-79` 的分支自然不触发）、tied embedding（`qwen3.py:238` 已有处理，Llama-3.2-1B/3B 是 tied）；**真正的工作量在 RoPE**——Llama-3 的 `rope_scaling` 频段重标定（low/high freq 双因子）需在 `rotary_embedding.py` 加解析分支，且 transformers 新旧版本的 rope_scaling 字段格式有变（上游 #198/#214/#182 三个 PR 都在修这个坑）。
- **模型选择**：Llama-3.2-3B（~6GB，单卡，迭代快）做对齐调试，Llama-3.1-8B 做性能报告。
- **验收**：HF `generate(do_sample=False)` 逐 token 对齐 100%（复用 1.1 的对齐框架）；`baseline.py` 全套件在 Llama 上出独立报告，与 Qwen3 共用同一 harness。
- **面试亮点**：能列清 Qwen3/Llama 的架构差异清单（q/k-norm、rope scaling、tied embedding、attention bias），而不是"改了个类名"。

### 2.5 Qwen3-MoE 支持（依赖：2.3 注册表 + 3.6 FP8 权重量化）

- **现状锚点**：`Qwen3MLP`（`qwen3.py:101-133`）是密集 SwiGLU；MoE 即把每层 MLP 替换为 router + 128 专家 top-8 的稀疏块，**注意力/调度/KV cache/prefix cache 全部原样复用**。
- **参考实现**：上游 PR #251（+1311 行：`Qwen3MoeForCausalLM` + fused MoE Triton kernel + top-k 路由 + 注册表接入，最完整，首选研读对象）、PR #116（FuseMoeLinear 版）、PR #236。
- **显存账（硬前提）**：Qwen3-30B-A3B 总参 30.5B，BF16 权重 61GB > 2×4090D 的 48GB；**必须走官方 FP8 checkpoint（约 31GB，TP2 每卡 ~15.5GB）+ weight-only 反量化**，因此 3.6 是本项前置。磁盘：数据盘剩约 47GB，Qwen3-4B(8GB) + 30B-A3B-FP8(~31GB) 可同放但余量小。
- **做法（三块，kernel 是大头）**：
  1. `models/qwen3_moe.py`：`Qwen3MoeSparseMoeBlock`（router gate + experts），config 读 `num_experts`/`num_experts_per_tok`/`moe_intermediate_size`；
  2. top-k 路由：router logits → top-8 → softmax 权重（几十行）；
  3. **fused MoE Triton kernel**：逐专家 Python 循环慢到不可用（PR #116 点名 HF 官方实现的性能问题），需按 expert 重排 token 的 grouped-GEMM 融合 kernel，从 vLLM / PR #251 移植。TP 策略先做"每个 expert 的 intermediate 维切分"，复用现有 `MergedColumnParallelLinear`/`RowParallelLinear` 基建；expert parallel 留作扩展。
- **量化指标**：① decode 吞吐 vs Qwen3-4B/8B 密集版（"3.3B 激活"的算力优势 vs 专家权重搬运的带宽劣势）；② **batch 敏感性曲线**（MoE 特有：小 batch 带宽浪费、大 batch 专家重叠摊薄）；③ 相对 HF transformers MoE 实现的加速比。
- **工作量**：3-5d（kernel 移植占一半以上）。
- **面试亮点**：MoE 是服务侧最主流的架构方向，能讲清"显存看总参、算力看激活参"、fused kernel 为什么必须有、TP/EP 两种专家切分的 trade-off。

### 2.4 调度策略消融：prefill/decode 干扰（新增）

- **现状锚点**：`scheduler.py:54-55`——只要 waiting 队列有可调度的 prefill，本步就不跑 decode（prefill 绝对优先）。
  长_prompt 请求持续到达时 decode 会被饿死，TPOT P99 恶化；这正是 Sarathi-Serve / chunked prefill 要解决的问题。
  现有代码已有 chunked prefill（`scheduler.py:42` 只允许队首序列分块），但**没有任何策略开关和量化数据**。
- **做法**：加调度策略选项：`prefill_priority`（现状）/ `decode_priority` / token-budget 混合（每步给 prefill 固定 token 预算，其余给 decode）。
- **验收**：用 1.3 的 `mixed` 套件画出 TTFT-TPOT 权衡曲线（扫 prefill 预算），证明不同策略在两个指标上的 trade-off。
- **面试亮点**：能从代码层面讲清 continuous batching 下队头阻塞的成因与缓解，直接对标业界论文。

## 3. P2 — 进阶（视时间选做）

| 改进 | 现状锚点 | 内容 | 量化指标 |
|---|---|---|---|
| 3.1 KV cache FP8 量化 | `model_runner.py:115` 按 bf16 一次性分配整块 KV | 写入时量化、attention 读侧反量化（flash-attn 不直接吃 FP8 KV，需在 `store_kvcache_kernel` 写侧量化 + attention 调用前转回，或换 Triton attention） | 同显存下最大 batch/上下文翻倍；长上下文吞吐变化；与 FP16 的 PPL/对齐率差 |
| 3.2 block size 消融 | `config.py:22` 断言 `kvcache_block_size % 256 == 0`，**当前根本不允许 32/64/128**；decode 侧 `flash_attn_with_kvcache` 未显式传 `page_size` | 先放宽断言并给 flash-attn 传 `page_size=block_size`，再扫 256/128/64/32 | 碎片率 vs kernel 效率 trade-off 曲线；注意 block 变小同时让 prefix cache 命中粒度更细，命中率本身也会变——两个变量要分开解释 |
| 3.3 融合 Triton kernel | `layernorm.py` / `rotary_embedding.py` 均为独立 kernel，逐层多次 launch | RMSNorm+residual+RoPE 融合，减少 launch 次数 | eager 模式 `step_time_ms.p50` 下降；`nsys` 里 launch 间隙缩短的证据 |
| 3.4 CUDA Graph 覆盖 prefill | `model_runner.py:197`：prefill 恒走 eager；graph 仅 decode 且 bs≤512 | prefill 按 token 数分桶 capture，或 decode 桶粒度调优（当前 `[1,2,4,8]+16 步进`，512 上限外的 batch 回落 eager） | 桶边界附近的延迟毛刺消除；短 prompt TTFT 变化 |
| 3.5 prefix cache 驱逐与 CoW（新增） | `block_manager.py:31` `hash_to_block_id` 无上限、无 LRU；`free_block_ids.remove()` 是 O(n)；命中块的追加写入未做 copy-on-write（`hash_blocks` 直接改共享块内容） | LRU 驱逐 + O(1) 索引 + 命中块追加前复制 | 长跑（数万请求）下命中率稳定不衰减；`hash_blocks` 边界场景正确性测试（共享块被追加时的污染问题） |
| 3.6 FP8 权重量化（weight-only，**MoE 前置**） | `loader.py` 按原始 dtype 加载，2×4090D 跑不了 30B 级权重 | 加载官方 FP8 checkpoint（如 Qwen3-30B-A3B-FP8，约 31GB），matmul 前 weight-only 反量化回 BF16 计算（参考上游 PR #241）；4090D 是 Ada，后续可进阶 FP8 GEMM | 显存减半后 30B-A3B 能以 TP2 跑起来；与 BF16 权重的输出对齐率；反量化开销 <3% |

## 4. 工程亮点（非性能，但面试加分）

- **OpenAI 兼容 server**（FastAPI + SSE 流式输出）：把引擎变成"产品"，演示多并发下指标曲线。
  有了 server 才能测完整 serving 链路（HTTP/流式）——当前 `serving` 套件是进程内直连引擎，
  只覆盖调度器+引擎层的排队行为。
- **开环压测 + goodput**：✅ 引擎层已完成（baseline.py `serving` 套件，指数到达 + goodput +
  injection_lag）；待做：server 层 HTTP 压测（对标 vLLM benchmark_serving、NVIDIA GenAI-Perf 范式），
  以及用多档速率画吞吐-延迟曲线定位拐点。
- **压测对比**：与 vLLM 同机同模型对比吞吐/延迟，`bench.py` 已留了 vLLM 切换注释。
- **结果文档**：每次优化的 before/after 表格 + 火焰图（`nsys` 或 `torch.profiler`）。
- **正确性回归自动化**：`git hook` / CI 里跑 correctness 套件，指纹不一致禁止提交（比"我记得没改坏"有说服力）。

## 推荐执行顺序

```
1.1 贪心+HF对齐 (0.5d) → 1.3 基线统计补强 (0.5d) → 2.3 注册表+Llama (1.5d)
    → 2.1 overlap (1-2d) → 3.6 FP8 权重量化 (1-2d) → 2.5 Qwen3-MoE (3-5d)
    → 2.4 调度消融 / 2.2 投机解码 视兴趣选 → 其余 3.x
```

依赖链说明：2.5 MoE 依赖 2.3 的注册表（按 architectures 分流）和 3.6 的 FP8 加载（否则 61GB 权重放不进 48GB），所以这两项排在它前面；2.2/2.4 与模型支持相互独立，可按兴趣对调。

## 结果台账（每次优化追加一行）

| 日期 | 优化项 | commit | 对比基线 | 关键指标变化 | 正确性 |
|---|---|---|---|---|---|
| | | | | | |

> 模板：`总吞吐 +X% / TPOT P99 -Y ms / TTFT P50 -Z ms / 前缀命中 +H%`，附 `results/*.json` 文件名。

## 面试叙事主线

**"我复刻了 vLLM 的核心机制（PagedAttention、continuous batching、chunked prefill、CUDA graph），
建立了可复现、带统计置信的基准体系并与 HF 对齐正确性；把引擎泛化成多架构
（模型注册表 + Llama 移植 + MoE with fused Triton kernel + FP8 权重量化），
再用 N 项有量化验证的优化（overlap 调度、调度策略消融等）把 decode 吞吐提升 X%、TPOT P99 降低 Y%，
并能用火焰图解释每一项收益的来源。"**
