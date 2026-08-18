# nano-vllm 面试项目改进路线

> 方法论：每一项改进都遵循 **baseline → 假设 → 实现 → `baseline.py --compare` 量化验证 → 消融** 的闭环，
> 结果记录到 `results/` 与本文件末尾的结果台账，面试时以表格呈现。
>
> 每一项都标注了**现状锚点**（代码里的具体位置），改进前先读对应代码，避免凭印象动工。
>
> **上游状态（2026-08 调研）**：上游 `GeeeekExplorer/nano-vllm` 自 2026-04-26 合并 PR #218（即本 fork 点，
> 等于上游 main 的 HEAD）后零合并、无维护者 review、无 roadmap，已休眠。
> 本 fork 事实上是最活跃的 head，不存在同步负担；上游 51 个 open + 约 60 个 closed-unmerged PR
> 反而是免费的想法库（本文吸收的条目均标注了 PR 号，2026-08-18 调研补充）。

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

### 1.4 上游正确性雷区收编为回归测试（3.5 / 2.4 重写的前置）

- **现状锚点**：上游 issue/PR 沉淀了约 10 个已知边界 bug，多数集中在 prefix cache × chunked prefill
  的交互路径（`block_manager.py` / `scheduler.py`）——正是 3.5（LRU+CoW 重写）和 2.4（调度重写）
  要动的代码。不先收编就重写，等于把别人踩过的坑再踩一遍。
- **清单（上游 PR/issue 号）**：
  1. #199：prompt 100% 被缓存命中时的崩溃（num_tokens == 0 边界）；
  2. #209：同一 prefill 批次内 in-flight 序列的前缀命中；
  3. #211：抢占（preemption）后索引越界；
  4. #219 / #243：同一步骤内多序列生成相同前缀块（重复哈希 / 去重时机）；
  5. #240：`may_append` 晚一个 token 才分配块（写进未分配块）；
  6. #114：prompt 长度恰等于 block size 时崩溃；
  7. #106 / #191：CUDA graph 下 block_tables 形状不匹配（enforce_eager=False 时）；
  8. #246 / #248：TP 多进程 shm-RPC 缺 worker ack，rank0 可能损坏在途命令（独立正确性 bug，可直接修）；
  9. #215：TP 下 num_kvcache_blocks 未 all_reduce 取各 rank 最小值（可直接修）。
- **做法**：每个 bug 从上游修复 diff 反推最小复现场景先写测试；能复现的打补丁，不能复现的记录
  "已验证不适用"；全部并入 correctness 套件，接第 4 节的 CI/git hook 自动化。
- **验收**：correctness 套件新增 ≥8 个边界用例；#248/#215 完成修复并有 TP2 双卡验证。
- **工作量**：0.5-1d。

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
- **第二阶段（学习型草稿，与 n-gram 形成草稿来源消融）**：n-gram 零成本但接受率上限低，
  生产主流是学习型草稿，两条路线**二选一**（都做是 scope creep）：
  - **EAGLE-3（推荐，低风险）**：feature 级 drafter（单层 decoder 吃 target 的 hidden states +
    已采样 embedding），第一版线性草稿即可用现有 target 前向批量验证，不需要特殊 kernel
    （树验证可后置，FlashInfer 有现成 verify kernel）。Qwen3-8B/14B/32B/30B-A3B 均有现成 head
    （如 HF `RedHatAI/Qwen3-8B-Thinking-speculator.eagle3`），vLLM speculators 官方支持，
    生产环境最高 ~2.5×。难点：hidden state 管线接入 CUDA graph 的 decode 路径。3-6d。
  - **MTP（更 novel）**：Qwen3 dense 未发布 MTP 权重；Qwen3.5 系列（0.8B/2B/4B + MoE）自带 MTP
    head，自投机不占额外模型显存，但需先移植 Qwen3.5 text path（原生多模态、新架构），工作量大得多。
  - **验收**：同 batch 同负载产出消融表——无投机 / n-gram / 学习型草稿 的 decode 吞吐与接受长度
    （期望 2-4）；贪心路径 HF 对齐保持 100%。
- **面试亮点**：vLLM/SGLang 都在用，可讲清 verify 的数学等价性（接受前缀 + 按概率重采样首个拒绝位）；
  升级后还能对比"草稿来源"（统计 n-gram vs 学习型 EAGLE-3）这一投机解码的核心设计维度。

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

### 2.6 CUDA graph decode 输入的宿主侧打包消除（上游实测背书，建议排在 2.1 之前）

- **现状锚点**：decode 每步的输入构造（input_ids/positions/slots/block_tables）是逐元素 torch 赋值
  + 每步新分配/pin tensor，全部位于同步路径上；CUDA graph 只消掉了 GPU 侧 launch 开销，
  CPU 侧打包开销仍在，小 batch decode 时占比高。
- **参考实现**：上游 PR #253（open，唯一带实测数据的 perf PR）：`DecodeInputBatch` 常驻 pinned
  CPU tensor + 零拷贝 NumPy 视图，NumPy 侧打包后一次 H2D；PR #176（pinned staging buffer 思路，
  被 #253 取代）；PR #239 另提出把 decode logits 也捕获进 graph。
- **上游实测**（A40, Qwen3-0.6B）：bs 1-16 下 +3.5%~5.8% tok/s，打包耗时 -43%，
  稳态零 cudaMalloc/cudaHostAlloc、无元数据 D2D 拷贝。
- **做法**：decode 输入全部走常驻 pinned buffer + NumPy 视图写入，graph replay 前单次 H2D；
  顺带评估 logits 进 graph（#239）。
- **验收**：小 batch decode `step_time_ms.p50` 下降 ≥3%；nsys 确认稳态无 cudaMalloc/HostAlloc；
  与 2.1 正交——先做本项（更便宜、有上游数据背书），之后 overlap 的剩余收益才能单独归因。
- **工作量**：1-2d。

## 3. P2 — 进阶（视时间选做）

| 改进 | 现状锚点 | 内容 | 量化指标 |
|---|---|---|---|
| 3.1 KV cache 量化（FP8 为主，INT8 对比） | `model_runner.py:115` 按 bf16 一次性分配整块 KV | 写入时量化、attention 读侧反量化（flash-attn 不直接吃 FP8 KV，需在 `store_kvcache_kernel` 写侧量化 + attention 调用前转回，或换 Triton attention）；对比路线：INT8 KV + head-major 布局 + KV 量化与 attention 计算的异步 stream 流水线（上游 PR #229/#184、issue #228，宣称 ~22% 吞吐 / ~48% 显存——上游社区的实际主攻方向），与 FP8 KV 在同一测量框架出对比表 | 同显存下最大 batch/上下文翻倍；长上下文吞吐变化；与 FP16 的 PPL/对齐率差；FP8 vs INT8 的吞吐/精度对比 |
| 3.2 block size 消融 | `config.py:22` 断言 `kvcache_block_size % 256 == 0`，**当前根本不允许 32/64/128**；decode 侧 `flash_attn_with_kvcache` 未显式传 `page_size` | 先放宽断言并给 flash-attn 传 `page_size=block_size`，再扫 256/128/64/32 | 碎片率 vs kernel 效率 trade-off 曲线；注意 block 变小同时让 prefix cache 命中粒度更细，命中率本身也会变——两个变量要分开解释 |
| 3.3 融合 Triton kernel | `layernorm.py` / `rotary_embedding.py` 均为独立 kernel，逐层多次 launch | RMSNorm+residual+RoPE 融合，减少 launch 次数；顺带：weightless RMSNorm / FlashNorm 式权重折叠（norm 权重折进后续 linear，跳过全 1 乘法——上游 PR #224/#250、issue #220，近零成本） | eager 模式 `step_time_ms.p50` 下降；`nsys` 里 launch 间隙缩短的证据 |
| 3.4 CUDA Graph 覆盖 prefill | `model_runner.py:197`：prefill 恒走 eager；graph 仅 decode 且 bs≤512 | prefill 按 token 数分桶 capture，或 decode 桶粒度调优（当前 `[1,2,4,8]+16 步进`，512 上限外的 batch 回落 eager） | 桶边界附近的延迟毛刺消除；短 prompt TTFT 变化 |
| 3.5 prefix cache 驱逐与 CoW（新增） | `block_manager.py:31` `hash_to_block_id` 无上限、无 LRU；`free_block_ids.remove()` 是 O(n)；命中块的追加写入未做 copy-on-write（`hash_blocks` 直接改共享块内容） | LRU 驱逐 + O(1) 索引 + 命中块追加前复制 | 长跑（数万请求）下命中率稳定不衰减；`hash_blocks` 边界场景正确性测试（共享块被追加时的污染问题） |
| 3.6 权重量化消融表（FP8 + INT4/AWQ，**MoE 前置**） | `loader.py` 按原始 dtype 加载，2×4090D 跑不了 30B 级权重 | ① FP8 weight-only：加载官方 FP8 checkpoint（如 Qwen3-30B-A3B-FP8，约 31GB），matmul 前 weight-only 反量化回 BF16 计算（参考上游 PR #241）；4090D 是 Ada，后续可进阶 FP8 GEMM。② INT4/AWQ weight-only：W4A16 走 Triton 路线 GemLite（兼容 CUDA graph）或 torchao `int4_weight_only`（sm80+ tinygemm，bs=1 最优）——**Marlin kernel 覆盖 sm89 可用，Machete 是 Hopper-only 不适用**；官方 Qwen3-32B-AWQ（~18GB）解锁 2×4090D TP2 跑 32B | FP8：显存减半后 30B-A3B 能以 TP2 跑起来；与 BF16 权重的输出对齐率；反量化开销 <3%。INT4：最大可服务模型（32B vs 14B）、decode 吞吐 vs FP8/BF16、显存、对齐率——与 FP8 合成一张"精度-显存-吞吐"消融表 |
| 3.7 分层 KV cache：GPU→CPU offload + swap 抢占（新增） | `scheduler.py:75-79` 抢占只有重计算路径（deallocate 后整体重新 prefill）；3.5 的 LRU 驱逐后冷命中同样要全量重算 | 驱逐块落 pinned CPU buffer（独立 copy stream 异步搬运），重命中异步预取回 GPU；抢占增加 swap 路径（KV 换出 CPU 保留进度）与现有 recompute 抢占做消融——即 vLLM v0 的 swap vs recompute 之争；整体对标 LMCache（vLLM 官方委托的分层缓存方案）的微缩版，纯 PyTorch 实现，无需外部 kernel | 已被驱逐前缀重新命中时的 TTFT：swap vs 重计算；同显存下可并发长上下文序列数；会话回放 trace 的 GPU 命中率 |
| 3.8 TP2 PCIe 自定义 one-shot allreduce（新增，系统项） | TP 通信走 NCCL；4090D 无 NVLink，decode 的小消息（16-256KB）allreduce 正是 NCCL 延迟短板 | CUDA IPC + P2P 写的 2 卡 one-shot allreduce（参考 vLLM custom allreduce、voipmonitor rtx6kpro 的 PCIe oneshot 写作，小消息下快 1.4-6×）；2 卡是最简情形（单 peer）。**否定结论防弯路：torch `symm_mem` 的 multimem/one-shot 需 NVLink 组播，PCIe 上不可用（vLLM #26949/#35724 实证）** | allreduce 延迟 vs 消息大小 vs NCCL（µs）；TP2 bs1-8 的 decode ITL/吞吐增量；nsys 通信/计算重叠证据。坑：switchless 拓扑需驱动 ForceP2P，与 `expandable_segments:True` 不兼容 |
| 3.9 torch.compile piecewise + cascade attention（新增，选做） | 3.4 的手写 graph 是人工路线；共享前缀负载仍走普通 attention | ① vLLM v1 默认的 torch.compile piecewise CUDA graph 作为 3.4 的对照路线（`max-autotune` 顺带自动融合，与 3.3 消融）；② cascade attention：共享前缀部分单次读取 + 各自后缀单独计算，LSE 合并（FlashInfer 支持 SM89），与现有 prefix cache 组合 | ① 相对手写 capture 的代码量与 CPU 开销（nsys 间隙）；② 共享 4-8k system prompt × 16-32 路并行生成的 decode 吞吐 |

### 明确不做清单（防跑偏，2026-08 调研结论）

| 项 | 不做的原因 |
|---|---|
| Medusa | 已被 EAGLE-3/MTP 淘汰，vLLM speculators 不收录，且无 Qwen3 head |
| Qwen3-Next / DeepSeek 的 MTP | 唯一带 MTP 权重的 checkpoint ≥80B 放不进 48GB；Qwen3-Next 的混合 GDN 注意力等于再写一个引擎。要 MTP 用 Qwen3.5 小模型（见 2.2） |
| torch `symm_mem` allreduce | multimem/one-shot 需 NVLink 组播，PCIe 上不工作（vLLM #26949/#35724 实证）；PCIe 优化走 3.8 的 IPC 路线 |
| KIVI 2-bit / SnapKV / H2O / PyramidKV | 生产引擎均未支持（vLLM #36311 仍 open），需逐块注意力统计管线，与 paged KV + CUDA graph 相性差，质量风险大 |
| MInference / Quest / NSA 稀疏注意力 | 需块稀疏 kernel 且伤非稀疏训练模型的精度；24GB 的瓶颈是显存不是注意力 FLOPs |
| multi-step scheduling | 已被 async scheduling 取代（vLLM v0.14 起默认），与 2.1 overlap 重复 |
| NIXL / Mooncake / LMCache 全家桶 | 集群级系统（RDMA、多副本 KV 路由），2 卡单副本无意义；分层缓存思想取 3.7 微缩版即可 |
| 独立立项的 2 卡 1P1D 分离 | 小模型 P/D 阶段太短，传输+编排开销吃掉收益；仅作为 2.4 消融的延伸测量（1P1D vs 同位 TP2 的 TTFT/ITL SLO 权衡） |
| DFlash / EAGLE 3.1 等 2026 前沿投机 | 论文引用即可（arXiv 2602.06036 等），draft checkpoint 少、迭代过快，不宜作项目依赖 |

## 4. 工程亮点（非性能，但面试加分）

- **OpenAI 兼容 server**（FastAPI + SSE 流式输出）：把引擎变成"产品"，演示多并发下指标曲线。
  有了 server 才能测完整 serving 链路（HTTP/流式）——当前 `serving` 套件是进程内直连引擎，
  只覆盖调度器+引擎层的排队行为。
- **xgrammar 结构化输出**（1-3d，性价比极高）：vLLM v1 默认后端，JSON schema/正则/EBNF 约束生成，
  per-token 开销 ~40-50µs 近乎免费。`GrammarCompiler` 每步在 CPU 侧填 token bitmask，采样前
  `apply_token_bitmask_inplace`；与 OpenAI server 的 `response_format`/`json_schema` 参数天然配套。
  验收：schema 合法率 100%，ITL 开销 <5%，与 CUDA graph decode 兼容。
- **增量式 detokenizer**：`llm_engine.py:89` 现状是生成结束后对完整 token 列表一次性 decode；
  流式 server 需要逐 token 增量解码（UTF-8 跨 token 边界、skip_special_tokens、stop string 截断），
  直接决定首字延迟指标。
- **开环压测 + goodput**：✅ 引擎层已完成（baseline.py `serving` 套件，指数到达 + goodput +
  injection_lag）；待做：server 层 HTTP 压测（对标 vLLM benchmark_serving、NVIDIA GenAI-Perf 范式），
  以及用多档速率画吞吐-延迟曲线定位拐点。
- **压测对比**：与 vLLM 同机同模型对比吞吐/延迟，`bench.py` 已留了 vLLM 切换注释。
- **结果文档**：每次优化的 before/after 表格 + 火焰图（`nsys` 或 `torch.profiler`）。
- **正确性回归自动化**：`git hook` / CI 里跑 correctness 套件，指纹不一致禁止提交（比"我记得没改坏"有说服力）。

## 推荐执行顺序

```
1.1 贪心+HF对齐 (0.5d) → 1.3 基线统计补强 (0.5d) → 1.4 上游正确性回归收编 (0.5-1d)
    → 2.3 注册表+Llama (1.5d) → 2.6 宿主侧打包 (1-2d) → 2.1 overlap (1-2d)
    → 3.6 量化消融表 FP8+INT4 (1-2d) → 2.5 Qwen3-MoE (3-5d)
    → xgrammar (1-3d) / 2.4 调度消融 / 2.2 投机解码(n-gram→EAGLE-3) 视兴趣选 → 其余 3.x
```

依赖链说明：2.5 MoE 依赖 2.3 的注册表（按 architectures 分流）和 3.6 的 FP8 加载（否则 61GB 权重放不进 48GB），所以这两项排在它前面；1.4 是 3.5/2.4 两次重写的前置（先收编上游已知 bug 再动手）；2.6 排在 2.1 之前——它更便宜、有上游实测背书，且做完后 overlap 的剩余收益才能单独归因；2.2/2.4 与模型支持相互独立，可按兴趣对调。

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
