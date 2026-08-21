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
>
> **2026-08-19 收束**：主线按面试可中途停的三层故事重排。MoE fused kernel / INT4 / EAGLE-3 /
> 自定义 allreduce / xgrammar **移出主线**（层 C 或停车场）。日历与依赖见 `timeline.md`。
>
> **2026-08-19 晚（M2 之后）**：硬件上限 **2×4090（48GB，PCIe，无 NVLink，SM89 Ada）**；当前常见是单卡开工。
> 下一阶段只加深两条、都做透：**2.7 同一步 hybrid P+D**（把 2.4 的轮转升级成 Sarathi stall-free）→
> **3.1 INT8 KV**（写侧量化；不走 SM89 不稳的 FP8 attention）。顺序不可对调。

## 三层目标（面试导向）

任何一天中断，故事必须仍然完整。面试官会深挖 2～3 项，不会听完 12 项清单。

| 层 | 目标 | 完成态 | 失败是否可接受 |
|---|---|---|---|
| **A 地基**（~3d，必须） | 能量化、敢改 | HF 贪心 exact-match；三次中位数 + ITL P99 + `mixed`；引擎内 cache hit / KV util / per-request TTFT | 否 |
| **B 引擎**（~4–6d，主线） | 懂服务侧引擎 | **2.4/2.6/2.2/2.7 已完成**；**3.1 INT8 KV 已完成**。下一阶段：层 C 可选 | 层 C 可失败 |
| **C 规模**（加分，可失败） | 碰到过规模问题 | 优先 **Qwen3-8B 单卡** 复测 2.7/3.1；Llama-3.2-3B / FP8 loader / MoE 仍可停 | 是。30B TP2 / fused kernel 不是叙事支柱 |

**最小可演示完成态 = 层 A + 2.4 消融表（已达成）。** 2.7/3.1 是 M2 之后的加厚，做成 2.4 的「生产引擎那一跳」+「decode 带宽墙」。

---

## 0. 基线（先跑这个）

```bash
python baseline.py --save baseline_v0     # 已完成：results/baseline_v0.json（2026-08-19，单卡 Qwen3-0.6B）
# 同模型、同 TP 的改进后:
python baseline.py --save opt_xxx
python baseline.py --compare results/baseline_v0.json results/opt_xxx.json
```

基线指标：总吞吐 / prefill·decode tok/s / TTFT·TPOT (P50/P90/P99) / 前缀缓存加速比 / 正确性指纹。

**对比纪律（禁止混表）**：

- `baseline_v0` **只**对比同配置：单卡、Qwen3-0.6B、同一套 `Config`。层 A / 2.4 / 2.6 / 2.1 / 2.2 / **2.7 / 3.1** 走这条线。3.1 若换 8B 必须另建 `baseline_qwen3_8b.json`。
- 换模型或换 TP 必须另建基线，例如 `baseline_llama3b.json`、`baseline_moe_tp2.json`。MoE 吞吐 **不得** 对 0.6B 报百分比。
- `--compare` 只对中位数（见 1.3）。收益小于三次运行极差时，面试里主动说「这项在噪声边缘，证据是 profiler」。

**基线自身的两个已知局限**（见 1.3，与层 A 一起补）：

- 每套件只跑 1 次，无方差信息，<5% 的优化结论不可信；
- ~~所有请求第 0 秒同时入队（饱和压测），测不出排队行为~~ → 已由 `serving` 套件解决
  （2026-08：指数间隔开环注入 + TTFT/TPOT 分位数 + SLO goodput + injection_lag，
  参考上游 PR #29 并修正其泊松间隔误用、TTFT 判定依赖调度器隐含约定、无分位数三处缺陷；
  remaining：mixed 套件尚未加，prefill/decode 干扰仍测不到）。

---

## 1. 层 A — 打地基（必须，让后续结论可信）

### 1.1 贪心采样 + HF 逐 token 对齐

- **现状锚点**：`sampling_params.py:11` 直接 `assert temperature > 1e-10` 禁止贪心；
  `sampler.py` 的 Gumbel-max 路径对所有温度统一处理。当前 baseline.py 用 1e-6 温度近似，依赖浮点 argmax 稳定性。
- **做法（本项范围内）**：`temperature == 0` 走 `logits.argmax(-1)`。正确性从「指纹自洽」升级为「与权威实现对齐」。
- **验收**：与 HuggingFace `generate(do_sample=False)` 输出 exact-match = 100%（4 条固定 prompt × 64 token）。
- **明确后移**：`top_p` / `top_k`（SamplingParams 目前只有 temperature/max_tokens/ignore_eos）是采样能力缺口，但 **不要塞进 1.1 同一下午**。有余力再补，用固定 seed 对齐 HF 概率分布。

### 1.2 引擎内指标上报（克制范围）

- **现状锚点**：baseline.py 是外部观测（对 step 计时），拿不到引擎内部状态；
  `block_manager.py` 里 `can_allocate` 命中前缀时、`free_block_ids` 长度变化处都是天然的埋点位置。
- **做法（只埋三点，不做监控系统）**：
  1. per-request TTFT / TPOT；
  2. KV cache 利用率（`len(free_block_ids)/num_blocks`，报告峰值/均值即可，不强制时序曲线产品化）；
  3. prefix 命中率（命中 block 数 / 请求 block 数）。
- **验收**：baseline 报告新增 cache hit rate、KV utilization；prefix_cache 套件的加速比能和命中率互相印证。

### 1.3 baseline 统计严谨性补强

- **做法**：
  1. 每套件跑 3 次取中位数并记录极差（对齐 vLLM 基准惯例），`--compare` 只对中位数比较；
  2. `run_workload` 里逐 token 记录 ITL 分布（数据在 step 循环里已有，只是没存），报告 ITL P99；
  3. 新增 `mixed` 套件：短 decode + 长 prompt 混合负载，暴露调度问题（为 2.4 提供测量工具）。
- **验收**：同一代码跑两次 baseline，关键指标波动 < 2%；否则说明环境噪声大，需先锁频/排查。
- **与 2.6 的契约**：上游 PR #253 预期 +3.5%～5.8%，正好在噪声区边缘。该项验收以 **三次中位数 + 极差 < 收益** 为准；否则吞吐数字不当面试主证据，改用 nsys「稳态无 cudaMalloc」。

### 1.4 上游正确性雷区：只收会打到后续改动的用例

- **现状锚点**：上游 issue/PR 沉淀了约 10 个已知边界 bug，多数集中在 prefix cache × chunked prefill
  的交互路径（`block_manager.py` / `scheduler.py`）——正是 2.4（调度策略）和日后 3.5（LRU+CoW）要动的代码。
  不先收编就改调度，等于把别人踩过的坑再踩一遍。
- **主线只收 3～4 个（会打到 prefix × chunked / preempt）**，从上游修复 diff 反推最小复现；能复现的打补丁，不能复现的记录「已验证不适用」，并入 correctness 套件：
  1. **#199**：prompt 100% 被缓存命中时的崩溃（`num_tokens == 0` 边界）；
  2. **#240**：`may_append` 晚一个 token 才分配块（写进未分配块）；
  3. **#211**：抢占（preemption）后索引越界；
  4. **#114**：prompt 长度恰等于 block size 时崩溃。
- **停车场（不占层 A 日历）**：#209 同批 in-flight 前缀命中；#219/#243 同一步骤重复哈希；#106/#191 CUDA graph `block_tables` 形状。动 CUDA graph / prefix 哈希时再捡。
- **TP 专属，用 TP 之前再修**：#246/#248 shm-RPC 缺 worker ack；#215 `num_kvcache_blocks` 未 all_reduce 取最小值。与单卡主线无关，挂到层 C 真上 TP2 之前。
- **验收**：correctness 套件新增上述 3～4 个边界用例；面试准备能讲清其中 **2 个根因**，而不是「我移植了 10 个 PR 测试」。
- **工作量**：1–1.5d（原 0.5–1d 收满 10 个不现实）。

---

## 2. 层 B — 服务侧引擎（主线）

**已完成**：2.4 消融 → 2.6 打包（噪声边）→ 2.2 n-gram（≤40 对齐）→ 2.7 hybrid P+D → **3.1 INT8 KV**。
**下一阶段**：层 C 可选（8B 复测 / Llama / FP8 / MoE）。不要再开 2.1 overlap。

### 2.4 调度策略消融：prefill/decode 干扰（主线第一项）

- **现状锚点**：`scheduler.py:54-55`——只要 waiting 队列有可调度的 prefill，本步就不跑 decode（prefill 绝对优先）。
  长 prompt 请求持续到达时 decode 会被饿死，TPOT P99 恶化；这正是 Sarathi-Serve / chunked prefill 要解决的问题。
  现有代码已有 chunked prefill（`scheduler.py:42` 只允许队首序列分块），但**没有任何策略开关和量化数据**。
- **做法**：加调度策略选项：`prefill_priority`（现状）/ `decode_priority` / token-budget 混合（每步给 prefill 固定 token 预算，其余给 decode）。
- **验收**：用 1.3 的 `mixed` 套件画出 TTFT–TPOT 权衡曲线（扫 prefill 预算），证明不同策略在两个指标上的 trade-off。
- **面试亮点**：能从代码层面讲清 continuous batching 下队头阻塞的成因与缓解，直接对标业界论文。性价比高于「移植某个 kernel PR」。
- **工作量**：1.5–2d。依赖 1.3 的 `mixed` 与 1.4 的 preempt/prefix 用例。
- **已完成（2026-08-19）**：策略开关 + mixed 消融表见台账 / `DEMO_2.4.md`。**剩余缺口**：`token_budget` 仍是逐步纯 P 或纯 D 轮转，不是 Sarathi 的同一步混合 batch → **2.7**。

### 2.7 同一步 hybrid P+D（Sarathi stall-free；2.4 加深）

- **现状锚点**：`scheduler.py` 的 `token_budget` 在 waiting+running 同时非空时 **一轮只返回纯 prefill 或纯 decode**（`_budget_turn_prefill` 翻转）。`ModelRunner.run` / `Attention.forward` 用 `is_prefill: bool` 二分，无法在同一次 forward 里既有多 token prefill chunk 又有 per-seq 1 token decode。
- **与 2.4 的差一句**：2.4 证明「谁先跑」改变 ITL 长尾；2.7 证明 **decode 算力有空档，把 prefill chunk 拼进同一步** 更接近 vLLM v1 默认（decode 优先 + 剩余 token 预算给 chunked prefill）。论文：Sarathi-Serve（OSDI’24）stall-free batching。
- **做法**：
  1. 新策略 `hybrid`（或扩展 `token_budget`）：每步先调度全部可跑的 decode（受 `max_num_seqs` 限制），再用剩余 `prefill_token_budget` 塞 waiting 队首的 chunked prefill。
  2. `schedule()` 返回混合 seq 列表 + 每条 `num_scheduled_tokens`（decode=1，prefill=chunk）。
  3. `prepare_hybrid`：走 **varlen prefill 路径**（`flash_attn_varlen_func` + `block_table`），decode 行当作 `seqlen_q=1` 的 query；**不要**走 CUDA Graph decode（graph 是 1 token/row 桶）。
  4. `postprocess` 按条处理：prefill chunk 未结束则不 sample/append；decode 与 prefill 完成条走现有 append。
- **验收**（`ablate_schedule.py` 同 mixed 16+16，三次中位数）：
  1. 正确性：greedy 4×64 HF 对齐不回退（hybrid 关 ngram）。
  2. 对照表至少三列：`prefill_priority` / `token_budget` 轮转 / **`hybrid`**。主看 LD ITL p99 与 LP TTFT、总吞吐。预期 hybrid 相对轮转：**ITL 不差于 decode_priority 量级，TTFT/吞吐好于小预算轮转**（同一步吃掉 decode 空档）。
  3. 若吞吐增益 < 极差：仍以「同一步混 batch 的机制 + 调度痕迹（每步同时有 P 和 D）」为主证据。
- **明确不做**：同一步再叠 2.2 n-gram verify；hybrid 步强制 `enable_ngram=False`。
- **面试亮点**：能画 2.4 轮转 vs 2.7 同 kernel 混 batch 的区别；对标 vLLM `--enable-chunked-prefill` / `max_num_batched_tokens`。
- **工作量**：1.5–2.5d。依赖 2.4 已落地。单卡 0.6B 即可。

### 3.1 INT8 KV cache（写侧量化；Ada/SM89 安全路径）

- **现状锚点**：`model_runner.py` `allocate_kv_cache` 按模型 dtype（bf16）分配 `[2, L, blocks, 256, kv_heads, D]`；`store_kvcache_kernel` 原样散写；decode 读 `flash_attn_with_kvcache`。
- **为何是 INT8 不是 FP8 attn**：4090 是 Ada SM89。vLLM 上 **FlashInfer FP8 attention 多为 SM90**，SM89 上 MTP/FP8 KV 有非法访存先例。flash-attn paged 也不直接吃 FP8 KV。本项路径：**INT8 存 HBM → 读入后反量化成 bf16 再进现有 FA**（或 Triton 读侧；参考上游 issue #228 / PR #229、下游 `naalo2/nano-vllm-kv-compression`，3090+0.6B 曾报 +22%）。
- **做法（克制范围）**：
  1. KV 分配改为 int8 + 每 token（或 per-head）scale 缓冲；`store_kvcache` 写时动态量化。
  2. attention 前把本步需要的 K/V 反量化到 bf16 workspace（decode 可只反量化当前页）。**第一版允许同步反量化**；async store pipeline（#228）有余力再加，不绑验收。
  3. head-major 重排 **可选**，不要和量化第一版绑死。
- **验收**：
  1. 正确性：`align_hf.py` 4×64 **允许少量 argmax 分叉**，须报 exact-match 率；面试主动说「KV 量化不是 bit-exact」。目标：≥ 默认温度 0 下 4 条里至少 3 条满匹配，或分叉位置记入台账。
  2. 显存：同样 `gpu_memory_utilization` 下 `num_kvcache_blocks` 约 **×2**（扣 scale 开销后仍明显增加）。
  3. 性能：0.6B 上 4090 带宽宽，**tok/s 可能 < 3090 的 +22%**。主 KPI = **同显存最大并发 seq 或更长 max_model_len 仍能跑 mixed**；辅 KPI = throughput 套件中位数（极差门闩同 1.3）。换 Qwen3-8B 时另存基线，禁止对 `baseline_v0` 报 %。
- **与 2.7 关系**：必须先 2.7。hybrid 的 varlen+block_table 读路径要在量化后仍正确（反量化 workspace 或 kernel 内 dequant）。
- **明确不做**：FlashInfer SM90 FP8 attn；KIVI 2-bit；与 2.2 同时开。
- **面试亮点**：decode 带宽墙、paged KV 量化合同、为什么 4090 选 INT8 而不是跟风 FP8 KV。
- **工作量**：2–3d。单卡；8B 复测可选（层 C）。


### 2.6 CUDA graph decode 输入的宿主侧打包消除（二选一的默认项）

- **现状锚点**：decode 每步的输入构造（input_ids/positions/slots/block_tables）是逐元素 torch 赋值
  + 每步新分配/pin tensor，全部位于同步路径上；CUDA graph 只消掉了 GPU 侧 launch 开销，
  CPU 侧打包开销仍在，小 batch decode 时占比高。
- **参考实现**：上游 PR #253（open，唯一带实测数据的 perf PR）：`DecodeInputBatch` 常驻 pinned
  CPU tensor + 零拷贝 NumPy 视图，NumPy 侧打包后一次 H2D；PR #176（pinned staging buffer 思路，
  被 #253 取代）；PR #239 另提出把 decode logits 也捕获进 graph。
- **上游实测**（A40, Qwen3-0.6B）：bs 1-16 下 +3.5%～5.8% tok/s，打包耗时 -43%，
  稳态零 cudaMalloc/cudaHostAlloc、无元数据 D2D 拷贝。
- **做法**：decode 输入全部走常驻 pinned buffer + NumPy 视图写入，graph replay 前单次 H2D。
  logits 进 graph（#239）评估即可，不绑进本项验收。
- **验收**：
  1. nsys 确认稳态无 cudaMalloc/HostAlloc（**主证据**）；
  2. 小 batch decode `step_time_ms.p50` 三次中位数下降，且极差小于收益；达不到则吞吐不当主证据，仍可凭 nsys 讲述。
- **为何默认选这项**：比 2.1 便宜、有上游数据背书、profiler 证据硬。
- **工作量**：1–2d。

### 2.1 Scheduler 与 GPU 执行重叠（二选一的高难度项）

- **现状锚点**：`llm_engine.py:49-55` 的 `step()` 是严格串行三段：CPU 调度（`scheduler.schedule()`，
  含 block 分配、xxhash 哈希、Python 循环）→ GPU 前向（`model_runner.call("run")`）→ `postprocess`。
  decode 步 batch 小、前向快（毫秒级），CPU 侧 Python 循环占比高，GPU 有明显气泡。
- **做法**：step N 的 GPU 前向异步发出后，CPU 立即做 step N+1 的调度与输入准备（双缓冲 + CUDA event 同步）。
  注意 `postprocess` 依赖本步采样结果，需要把「采样回传」与「下一步调度」解耦成流水线（vLLM v1 的做法）。
- **预期**：小 batch decode `step_time_ms.p50` 下降 10%～30%（三次中位数）。
- **风险**：调度与执行重叠后，KV cache 分配决策发生在前向完成之前，预抢占路径（`scheduler.py:58-65`）需要重新推演正确性。1.4 没收干净会在这里返工。
- **工作量**：2–3d（原 1–2d 偏紧）。选这项就不把 2.6 当必须项；若 2.6 已做完再做本项，剩余收益才能单独归因。

### 2.2 N-gram / Prompt-lookup 投机解码（层 B 第三条，有余力才做）

- **现状锚点**：decode 每步每序列只产 1 token（`sampler.py` 返回 `[S]`）；
  `prepare_decode` 的输入构造按「每序列 1 token」假设写死，verify 路径需要改成每序列 k+1 token 的变长 prefill 式前向。
- **做法**：用请求自身的已生成内容做 n-gram 匹配草拟 k 个候选 token，一次前向批量验证（不需草稿模型，纯算法改动）。
- **预期**：代码/重复文本场景 decode 吞吐 +30%～80%；新增「接受率」指标（1.2 的埋点顺手记录）。
- **验收**：同 batch 同负载：无投机 vs n-gram 的 decode 吞吐与接受长度；贪心路径 HF 对齐保持 100%。
- **面试亮点**：讲清 verify 的数学等价性（接受前缀 + 按概率重采样首个拒绝位）。n-gram 已足够覆盖这个维度。
- **明确不做（主线）**：EAGLE-3 / MTP。学习型草稿要 hidden state 进 CUDA graph decode，与 2.1 叠难度；草稿 checkpoint 与迭代速度不适合当项目依赖。论文引用即可，细节见第 3 节停车场与「明确不做」。

---

## 3. 层 C — 规模问题（加分，允许不完成）

层 C **不是** 11–16 天主线的高潮。做到「能讲清显存账和没做完的部分」即可。

### 2.3 模型注册表 + Llama-3.2-3B（脚手架，不为 8B 报告绑死）

- **现状锚点**：`model_runner.py:9,31` 硬编码 `Qwen3ForCausalLM`；仓库唯一模型文件 `models/qwen3.py`。
- **参考实现**：上游 PR #95（`model_registry.py` 125 行 + `models/__init__.py`）、PR #188（`models/llama.py` 204 行 + `rotary_embedding.py` +78 行）。均未被合并，可整体借鉴。
- **做法（两步）**：
  1. **注册表**：新增 `models/model_registry.py`，按 `hf_config.architectures[0]` 查表（比 `model_type` 稳——密集/MoE/VL 的 architectures 名不同，天然分流，PR #251 也验证了这个方向）；`model_runner.py` 改为 `create_model(hf_config)`，仅 ~10 行改动。
  2. **Llama 移植**：新增 `models/llama.py`，以 PR #188 为底。与 Qwen3 的实际差异很小：无 q/k-norm（`qwen3.py:77-79` 的分支自然不触发）、tied embedding（`qwen3.py:238` 已有处理，Llama-3.2-1B/3B 是 tied）；**真正的工作量在 RoPE**——Llama-3 的 `rope_scaling` 频段重标定（low/high freq 双因子）需在 `rotary_embedding.py` 加解析分支，且 transformers 新旧版本的 rope_scaling 字段格式有变（上游 #198/#214/#182 三个 PR 都在修这个坑）。
- **模型选择**：只下 **Llama-3.2-3B（~6GB，单卡）** 做对齐。Llama-3.1-8B 性能报告不下、不绑进验收（磁盘与日历都不够）。
- **验收**：HF `generate(do_sample=False)` 逐 token 对齐 100%（复用 1.1 的对齐框架）；另存 `baseline_llama3b.json`，**不** 与 `baseline_v0` 比百分比。
- **面试亮点**：能列清 Qwen3/Llama 的架构差异清单（q/k-norm、rope scaling、tied embedding、attention bias），而不是「改了个类名」。这是脚手架，不是差异化亮点。

### 3.6 FP8 weight-only loader（小模型先做对；INT4 移出主线）

- **现状锚点**：`loader.py` 按原始 dtype 加载。
- **层 C 范围**：用 **小模型 FP8 checkpoint** 把 block-wise 反量化做对（读 `weight_scale_inv`，block-size 128），matmul 前 weight-only 还原 BF16（参考上游 PR #241 的方向；官方 30B-A3B-FP8 是细粒度分块，比 per-channel 重一截）。4090D 是 Ada，后续可进阶 FP8 GEMM，不绑本项。
- **验收**：小模型 FP8 vs 同结构 BF16 输出对齐率；反量化开销有量级概念即可。
- **30B-A3B-FP8（~31GB）**：仅当层 B 已完成、磁盘 ≥32GB 空闲、且准备上 2.5 时再下载。与 INT4/32B-AWQ **禁止同时占盘**。
- **INT4/AWQ（Qwen3-32B-AWQ ~18GB）**：移出主线与日历。W4A16 可走 GemLite 或 torchao `int4_weight_only`（Marlin 覆盖 sm89；Machete 是 Hopper-only 不适用）。需要「精度–显存–吞吐」第三列时再做，做完即删权重。

### 2.5 Qwen3-MoE（胜利条件 = load + 朴素专家循环）

- **依赖**：2.3 注册表 + 3.6 FP8 loader。真上 30B TP2 前先修 1.4 停车场里的 #248/#215。
- **现状锚点**：`Qwen3MLP`（`qwen3.py:101-133`）是密集 SwiGLU；MoE 即把每层 MLP 替换为 router + 128 专家 top-8 的稀疏块，**注意力/调度/KV cache/prefix cache 全部原样复用**。
- **参考实现**：上游 PR #251（+1311 行：`Qwen3MoeForCausalLM` + fused MoE Triton kernel + top-k 路由 + 注册表接入）、PR #116、PR #236。
- **显存账（硬前提）**：Qwen3-30B-A3B 总参 30.5B，BF16 权重 61GB > 2×4090D 的 48GB；跑 30B 必须走官方 FP8（约 31GB，TP2 每卡 ~15.5GB）+ weight-only 反量化。
- **层 C 做法（两档）**：
  1. **完成态（本项胜利）**：`models/qwen3_moe.py`（`Qwen3MoeSparseMoeBlock` + top-k 路由）能 load；朴素逐专家循环能出 token；HF 贪心对齐在可测规模上成立。面试能讲清「显存看总参、算力看激活参」以及朴素循环为什么慢。
  2. **下一步（不绑里程碑）**：fused MoE Triton grouped-GEMM（PR #116 点名 HF 官方实现不可用）。TP 先切每个 expert 的 intermediate 维，复用 `MergedColumnParallelLinear`/`RowParallelLinear`；expert parallel 留作扩展。
- **量化指标（仅当 30B TP2 真跑起来）**：① vs 密集 4B/8B 的 decode 吞吐（激活参 vs 专家权重带宽）；② batch 敏感性曲线；③ vs HF MoE。另存 `baseline_moe_tp2.json`，禁止对 `baseline_v0` 报加速比。
- **工作量**：结构 + 朴素路径 1–2d；fused kernel + TP2 另计 3–5d+，不写入层 B 日历。

---

## 4. 停车场 — 进阶（主线不做；调研保留防跑偏）

层 B 做透之前不要从这里拉项进日历。3.8 / 3.5 / EAGLE-3 尤其容易变成半成品。
**例外**：3.1 INT8 KV 已升格为 M2 之后正式项（见上文）；下表 3.1 行只保留未做的 FP8-attn 变体。

| 改进 | 现状锚点 | 内容 | 量化指标 |
|---|---|---|---|
| 3.1 KV cache 量化 | 已提升为正式项 **3.1 INT8 KV**（见层 B 续）。停车场仅留：FP8 KV + FA3 attn（SM89 不稳）、INT8 vs FP8 精度表 | — | 勿在 4090 上硬上 FlashInfer FP8 attn |
| 3.2 block size 消融 | `config.py:22` 断言 `kvcache_block_size % 256 == 0`，**当前根本不允许 32/64/128**；decode 侧 `flash_attn_with_kvcache` 未显式传 `page_size` | 先放宽断言并给 flash-attn 传 `page_size=block_size`，再扫 256/128/64/32 | 碎片率 vs kernel 效率；block 变小会让 prefix 命中粒度变细——两个变量分开解释 |
| 3.3 融合 Triton kernel | `layernorm.py` / `rotary_embedding.py` 均为独立 kernel | RMSNorm+residual+RoPE 融合；顺带 weightless RMSNorm / FlashNorm 式权重折叠（上游 PR #224/#250、issue #220） | eager `step_time_ms.p50`；nsys launch 间隙 |
| 3.4 CUDA Graph 覆盖 prefill | `model_runner.py:197`：prefill 恒走 eager；graph 仅 decode 且 bs≤512 | prefill 按 token 数分桶 capture，或 decode 桶粒度调优 | 桶边界延迟毛刺；短 prompt TTFT |
| 3.5 prefix cache 驱逐与 CoW | `block_manager.py:31` `hash_to_block_id` 无上限、无 LRU；`free_block_ids.remove()` 是 O(n)；命中块追加未做 CoW | LRU + O(1) 索引 + 追加前复制 | 长跑命中率不衰减；共享块被追加时的污染测试。动这项前把 1.4 停车场里 prefix 用例收齐 |
| 3.6 续：INT4/AWQ 消融 | 见层 C | 官方 Qwen3-32B-AWQ 解锁 TP2 跑 32B；与 FP8 合成精度–显存–吞吐表 | 最大可服务模型、decode 吞吐、对齐率。与 30B-FP8 **不同时占盘** |
| 3.7 分层 KV：GPU→CPU offload + swap 抢占 | `scheduler.py:75-79` 抢占只有重计算 | 驱逐块落 pinned CPU buffer；swap vs recompute 消融（vLLM v0 之争）；对标 LMCache 微缩版 | 冷前缀再命中的 TTFT；可并发长上下文数 |
| 3.8 TP2 PCIe 自定义 one-shot allreduce | TP 走 NCCL；4090D 无 NVLink | CUDA IPC + P2P 写的 2 卡 one-shot（vLLM custom allreduce）。**否定结论：torch `symm_mem` multimem/one-shot 需 NVLink 组播，PCIe 不可用（vLLM #26949/#35724）** | allreduce 延迟 vs NCCL；TP2 bs1-8 ITL。坑：ForceP2P 与 `expandable_segments:True` 不兼容 |
| 3.9 torch.compile piecewise + cascade attention | 3.4 的手写 graph；共享前缀仍走普通 attention | ① piecewise CUDA graph 对照 3.4；② cascade attention（FlashInfer SM89）与 prefix cache 组合 | 相对手写 capture 的 CPU 开销；共享 4–8k system prompt 的 decode 吞吐 |
| EAGLE-3 / MTP | 见原 2.2 | feature 级 drafter；Qwen3 有现成 EAGLE-3 head，但 hidden state 接入 CUDA graph decode 是另一条项目 | 仅论文引用。不要与 n-gram 写进同一条「视兴趣」 |

### 明确不做清单（防跑偏，2026-08 调研 + 2026-08-19 主线收束）

| 项 | 不做的原因 |
|---|---|
| 主线同时做 2.6 **和** 2.1 | 两样都浅不如一样深；2.6 已做且数字在噪声边，**不再开 2.1** |
| 2.7 与 3.1 对调或并行浅做 | hybrid 先钉 attention 混合路径，再量化；两样都浅 = 计划失败 |
| SM89 上 FlashInfer/FA3 **FP8 attention** | 4090 内核不齐，vLLM 有非法访存先例；3.1 只做 INT8 存储 + bf16 计算 |
| 主线 EAGLE-3 / MTP / 2.2「升级到学习型草稿」 | 与 overlap 叠难度；draft 迭代过快。n-gram 已够讲 verify 等价性 |
| 主线 INT4/32B-AWQ、30B fused MoE 当高潮 | 磁盘互斥、日历下界当中位数、半成品会断在最显眼处 |
| 主线 xgrammar | 纯推理岗位优先级低于调度/overlap；要「产品演示」时再做 |
| Llama-3.1-8B 性能报告 | 层 C 只需要 3B 对齐；8B 占盘且不增加面试可讲的架构差异 |
| Medusa | 已被 EAGLE-3/MTP 淘汰，vLLM speculators 不收录，且无 Qwen3 head |
| Qwen3-Next / DeepSeek 的 MTP | 唯一带 MTP 权重的 checkpoint ≥80B 放不进 48GB；Qwen3-Next 的混合 GDN 注意力等于再写一个引擎 |
| torch `symm_mem` allreduce | multimem/one-shot 需 NVLink 组播，PCIe 上不工作；PCIe 优化若做走 3.8 IPC |
| KIVI 2-bit / SnapKV / H2O / PyramidKV | 生产引擎均未支持（vLLM #36311 仍 open），与 paged KV + CUDA graph 相性差 |
| MInference / Quest / NSA 稀疏注意力 | 需块稀疏 kernel 且伤精度；24GB 瓶颈是显存不是注意力 FLOPs |
| multi-step scheduling | 已被 async scheduling 取代（vLLM v0.14 起默认），与 2.1 overlap 重复 |
| NIXL / Mooncake / LMCache 全家桶 | 集群级系统，2 卡单副本无意义；分层缓存思想取 3.7 微缩版即可 |
| 独立立项的 2 卡 1P1D 分离 | 小模型 P/D 阶段太短，开销吃掉收益；仅可作为 2.4 的延伸测量 |
| DFlash / EAGLE 3.1 等 2026 前沿投机 | 论文引用即可（arXiv 2602.06036 等），不宜作项目依赖 |

---

## 5. 工程亮点（非性能；排在层 B 之后）

层 B 的消融表未完成前，不要把日历交给这一节。

- **结果文档**：每次优化的 before/after 表格 + 火焰图（`nsys` 或 `torch.profiler`）。这是面试材料本体，随层 A/B 一起做，不是选做。
- **正确性回归自动化**：`git hook` / CI 里跑 correctness 套件，指纹不一致禁止提交。
- **增量式 detokenizer**：`llm_engine.py:89` 现状是生成结束后对完整 token 列表一次性 decode。只有做流式 server 才需要（UTF-8 跨 token 边界、skip_special_tokens、stop string）。
- **OpenAI 兼容 server**（FastAPI + SSE）：把引擎变成「产品」。当前 `serving` 套件是进程内直连，只覆盖调度器+引擎层排队。
- **xgrammar 结构化输出**（1–3d）：vLLM v1 默认后端；与 server 的 `response_format`/`json_schema` 配套。纯推理面试优先级低于 2.4/2.6。验收：schema 合法率 100%，ITL 开销 <5%，与 CUDA graph decode 兼容。
- **开环压测 + goodput**：✅ 引擎层已完成（baseline.py `serving` 套件）；待做：server 层 HTTP 压测（对标 vLLM benchmark_serving）。
- **压测对比**：与 vLLM 同机同模型对比，`bench.py` 已留切换注释。作为层 B 数字的外部锚点，不是新项目。

---

## 推荐执行顺序

```
层 A（必须，~3d）
  1.1 贪心+HF对齐（不含 top_p/top_k）
  → 1.3 三次中位数 + ITL P99 + mixed  ∥  1.2 三点埋点
  → 1.4 仅 #199/#240/#211/#114

层 B（主线，已完成 2.4/2.6/2.2/2.7）
  → ✅ 2.7 hybrid P+D 同一步
  → 3.1 INT8 KV（写侧量化 + 读前反量化；禁 SM89 FP8 attn）

层 C（加分，可停）
  → 可选 Qwen3-8B 单卡复测 2.7/3.1（`baseline_qwen3_8b`）
  → 2.3 注册表 + Llama-3.2-3B（独立 baseline_llama3b）
  → 3.6 小模型 FP8 loader
  → 2.5 结构 + 朴素专家循环；fused kernel / 30B TP2 是下一步
```

依赖链（不可对调）：

- 没有 `baseline_v0`，后面的百分比都是口头数字；
- 没有 1.1，后面任何调度/kernel 改动都不敢说没改坏；
- 没有 1.3，2.6 那种 +3.5% 落在噪声里；
- 没有 1.3 的 `mixed` + 1.4 的 3～4 个用例，不要改 2.4；
- 2.6 先于 2.1（仅当两者都做时）；**当前不再做 2.1**；
- **2.4 → 2.7 → 3.1**：先有轮转消融，再同一步混 batch，再动 KV dtype；
- 2.5 依赖 2.3 注册表与 3.6 FP8；30B 另建 `baseline_moe_tp2`，禁止对 `baseline_v0` 比加速比。

---

## 结果台账（每次优化追加一行）

| 日期 | 优化项 | commit | 对比基线 | 关键指标变化 | 正确性 |
|---|---|---|---|---|---|
| 2026-08-19 | 第 0 步建基线 | a5fa446 | —（自身） | 单卡 Qwen3-0.6B：throughput 套件 6153 tok/s；latency bs1 TPOT p50 2.28 ms | 指纹自洽（1e-6 温度近似） |
| 2026-08-19 | 1.1 贪心 + HF 对齐（不含 top_p/top_k） | 未提交 | baseline_v0（正确性套件） | 非性能项。`SamplingParams` 允许 T=0；Sampler 走 `logits.argmax`。验收：`python align_hf.py` 4×64 exact-match 100%（裁判=HF `flash_attention_2`；默认 SDPA 会在 argmax 边界分叉）。CUDA Graph 路径同样 100%。指纹：prompt0 因真贪心相对 1e-6 变化（`7ad27c65ccd97ed7`），其余 3 条与 v0 相同。见 `results/opt_1.1_greedy.json` | **HF greedy exact-match 100%** |
| 2026-08-19 | 1.2 引擎三点埋点 + 1.3 统计补强 | 未提交 | 同配置三次中位数（新格式） | **1.2**：`EngineMetrics` 上报 prefix hit / KV util peak·mean / per-request TTFT·TPOT；`clear_prefix_cache` 保证测量独立。prefix_cache：shared hit **0.425** vs no-share **0.0**，speedup **1.59×** 互相印证（首波 prefill 同批无法自命中，0.425=17/32×4/5 属预期）。**1.3**：`--repeats 3` 取中位数+极差；`itl_ms`；新 `mixed` 套件。latency bs1 thr 极差 0.74/413≈0.2%≪2%。见 `results/opt_1.2_1.3_d2.json` | 指纹与 1.1 一致 |
| 2026-08-19 | 1.4 边界回归 #199/#240/#211/#114 | 未提交 | —（正确性） | **#211 实修**：`may_append`/`can_append` 增加 `len(block_table) < num_blocks`，避免抢占后 re-allocate 已备块时再追加导致写错槽。**#199/#114 N/A**：fork 已含 #218（最多命中 `num_blocks-1`），原 `seqlen_q==0` 崩溃不再触发。**#240 N/A**：`==1` 相对 schedule→forward→postprocess 时序正确，跨块 decode 通过。套件 `tests/test_1_4_edges.py` + `baseline --suites edge_cases`；见 `results/opt_1.4_edges.json` | 7/7 pass·n/a；指纹稳定 |
| 2026-08-19 | 2.4 调度策略消融 | 未提交 | mixed 同负载三次中位数 | 新增 `schedule_policy`∈{prefill_priority, decode_priority, token_budget}+`prefill_token_budget`。引擎单步仍纯 P/D；token_budget=限预算 prefill↔decode 轮转。**消融表**（LP=长prefill组 TTFT p50，LD=长decode组）：prefill 4453 tok/s / TTFT145 / ITL_p99 **5.1**；decode 4123 / 145 / **3.5**（decode 不饿死）；budget 256→4096 扫出 TTFT 736→97.5 与 thr 1976→4397 的权衡。见 `results/opt_2.4_schedule.json`，脚本 `ablate_schedule.py` | 1.4 回归仍过 |
| 2026-08-19 | 2.6 宿主侧 decode 打包 | 未提交 | opt_1.2_1.3_d2 latency | 新增 `DecodeInputBatch`（pinned+NumPy→单次 H2D 进 graph buffer）；CUDA Graph decode 不再每步 `torch.tensor(... pin_memory)`。**吞吐**：bs1/8/64 thr +1.6%/+0.4%/+0.8%，step_p50 −1.3%/−0.8%/−0.7%；bs1 step 极差 0.07 ≥ 收益，**吞吐不作主证据**。本机无 nsys；主证据=打包路径零 `torch.tensor` 单测 + 稳态 pinned `data_ptr` 不变。见 `results/opt_2.6_decode_pack.json`、`tests/test_decode_input_batch.py` | 指纹与 1.1 一致 |
| 2026-08-21 | 3.1 INT8 KV cache | 未提交 | 同 gpu_util=0.85 | `kv_cache_dtype=int8`：group-wise(G=16) 写侧量化 + 活跃 block gather-dequant 进 FA；禁 CUDA Graph。**blocks 655→1166（×1.78）**；`align_hf` **3/4** exact（prompt0 @5 分叉）。**吞吐（双方 eager）**：throughput **4436→993 tok/s（−78%）**；latency/mixed 约 **−60%**——主因读侧同步 dequant 税，非测噪。见 `opt_3.1_int8_kv.json`、`opt_3.1_baseline_{auto,int8}_eager.json`、`note.md` | 3/4 HF；1.4 过 |

> 同配置模板：`总吞吐 +X% / TPOT P99 -Y ms / TTFT P50 -Z ms / 前缀命中 +H%`，附 `results/*.json`。
> 换模型/TP：在「对比基线」列写独立文件名，不要填相对 `baseline_v0` 的百分比。

---

## 面试叙事主线

> 我在约 1200 行的 nano-vllm 上复刻了 PagedAttention / continuous batching / chunked prefill / CUDA graph；
> 先做成可复现、带重复测量、并与 HuggingFace 贪心对齐的基准。然后针对 continuous batching 下
> **prefill 饿死 decode** 做了调度消融（`DEMO_2.4.md`），再升级为 **同一步 hybrid P+D（2.7）**。
> 最后做 **INT8 KV（3.1）**：写侧 group-wise 量化、读前反量化进 FA（4090 不上 SM90 FP8 attn），
> 同显存 blocks **×1.78**，贪心对齐 3/4（主动说明非 bit-exact）。宿主侧打包（2.6）本机收益在噪声边只讲机制。
