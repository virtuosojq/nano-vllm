# 02 vLLM V1 源码（第 2–3 周）

## 目标

1. nano-vllm 里的每个概念，都能在 vLLM 里找到对应实现，并说出 vLLM 多做了什么、为什么。
2. 面试中被问到 vLLM 某个机制时，能讲到「哪个类、哪个函数、关键数据结构」这一层。
3. 产出一份「平台插件需要实现哪些接口」的清单，作为第 4 周读 vllm-ascend 的地图。

## 准备

```bash
git clone https://github.com/vllm-project/vllm.git /home/wjq/src/vllm
cd /home/wjq/src/vllm && git checkout v0.31.0   # 和 /home/wjq/tmp/vllm-ascend/Dockerfile.a3 的 VLLM_TAG 对齐
```

下面写的文件路径是按 V1 架构整理的，vLLM 迭代很快，**以 `v0.31.0` 的实际代码为准**。找不到时用类名搜：

```bash
rg -n "class Scheduler\b" vllm/v1
rg -n "class KVCacheManager" vllm/v1
rg -n "class EngineCore\b" vllm/v1
rg -n "class .*ModelRunner" vllm/v1/worker
rg -n "class CUDAGraphMode" vllm
rg -n "class Platform\b" vllm/platforms
```

vllm-ascend 里有 `patch_use_v2_model_runner.py`，说明这个版本的 vLLM 有 V1 和 V2 两套 model runner。读第 4 节前先确认哪个是默认的：`rg -n "use_v2_model_runner|VLLM_USE_V2_MODEL_RUNNER" vllm`。

## 读法

- **带着问题读**，每节下面的问题清单是读完要能回答的，答案写进 `/home/wjq/infra/interview-plan/笔记-vLLM.md`。
- **每节先回忆 nano-vllm 怎么做的**，再看 vLLM，最后写「多了什么、为什么」。
- **只读主路径**。遇到 LoRA、多模态、structured output、pooling 这类分支，先跳过。
- 每读完一节，用自己的话画一张小图（模块关系或数据流），面试时能在白板上画出来。

---

## 第 1 节：请求入口与进程结构（第 2 周，1 天）

**读什么**

- `vllm/entrypoints/llm.py`（离线 `LLM`）、`vllm/entrypoints/openai/` 下的服务入口
- `vllm/v1/engine/async_llm.py`（在线 `AsyncLLM`）、`llm_engine.py`
- `vllm/v1/engine/core.py`（`EngineCore`、`EngineCoreProc`）、`core_client.py`
- 输入处理（分词、参数校验）和 `output_processor.py`（反分词、组装流式输出）

**nano-vllm 对应**：`LLMEngine` 一个类全包了，分词、调度、执行、反分词都在同一个进程、同一个循环里。

**问题清单**

1. 一个请求从 HTTP 进来，到第一个 token 流式返回，经过了哪些进程和哪些类？
2. 为什么要把 `EngineCore` 放到单独进程？（提示：分词和反分词占 CPU，Python 有 GIL，会和调度、下发 kernel 抢时间）
3. 前端进程和 `EngineCore` 之间用什么通信？传的是什么数据？
4. `EngineCore` 的主循环每一轮做哪几件事？
5. 流式输出里的增量反分词是在哪个进程做的？

---

## 第 2 节：调度器（第 2 周，1.5 天）

**读什么**

- `vllm/v1/core/sched/scheduler.py`：`Scheduler.schedule()`、`update_from_output()`
- `vllm/v1/core/sched/output.py`：`SchedulerOutput`
- `vllm/v1/request.py`：`Request` 和它的状态

**nano-vllm 对应**：`scheduler.py`。原版是「有 waiting 就整步做 prefill，否则整步做 decode」；你的 `back` 分支加了 token_budget 和 hybrid。

**核心认识**：V1 调度器不再区分 prefill 步和 decode 步。每个请求只有 `num_computed_tokens`（已算过的 token 数）和总 token 数，调度器每步给每个请求分配「这一步算多少个 token」。decode 就是分配 1 个，prefill 就是分配很多个，chunked prefill 就是分配不完整的一段。你的 hybrid 调度就是在往这个方向走。

**问题清单**

1. `schedule()` 先处理 running 还是 waiting？为什么？
2. token 预算（`max_num_batched_tokens`）是怎么在请求之间分配的？长 prompt 怎么被切块？
3. 什么时候会发生抢占？抢占谁？被抢占的请求怎么恢复？（V1 用重算，不再做 swap 到 CPU；想清楚为什么）
4. `SchedulerOutput` 里为什么要区分「新请求」和「已缓存请求」？只传增量有什么好处？
5. prefix cache 命中的 token 在调度时怎么处理？（提示：直接计入 `num_computed_tokens`）
6. `update_from_output()` 做了哪些事？请求在什么条件下结束？
7. 异步调度（async scheduling）是什么？它想重叠什么和什么？
8. 优先级调度和 FCFS 的区别在代码里体现在哪？

---

## 第 3 节：KV cache 管理（第 2 周，1.5 天）

**读什么**

- `vllm/v1/core/kv_cache_manager.py`：`KVCacheManager`，重点看 `get_computed_blocks()`、`allocate_slots()`、`free()`
- `vllm/v1/core/block_pool.py`：`BlockPool`，空闲块队列、缓存的块
- `vllm/v1/core/kv_cache_utils.py`：块哈希、`FreeKVCacheBlockQueue`、按显存算块数
- `vllm/v1/core/kv_cache_coordinator.py`、`single_type_kv_cache_manager.py`：多种 attention 类型（全注意力、滑窗、Mamba）共存

**nano-vllm 对应**：`block_manager.py`，块大小 256，用 xxhash 对块内 token 加上一个块的哈希做链式哈希。

**问题清单**

1. 块哈希由哪些内容组成？为什么要带上父块的哈希？（多模态、LoRA 的额外 key 是什么作用）
2. 为什么只缓存「满块」？最后一个不满的块怎么处理？
3. 引用计数归零的块会立即被回收吗？空闲队列为什么是 LRU 顺序？命中一个已在空闲队列里的块时会发生什么？
4. 块数是怎么根据显存算出来的？`gpu_memory_utilization` 管的是什么？
5. 为什么 vLLM 默认块大小是 16，nano-vllm 用 256？各自的利弊是什么？（提示：碎片、前缀命中粒度、kernel 对块大小的要求）
6. 滑窗注意力的层，旧的块什么时候可以释放？混合模型怎么让不同层用不同的管理策略？
7. KV cache 满了以后，调度器和 KV 管理器是怎么配合的？

---

## 第 4 节：Worker 与 Model Runner（第 3 周，1.5 天）

**读什么**

- `vllm/v1/worker/gpu_worker.py`：设备初始化、显存 profile、KV cache 分配
- `vllm/v1/worker/gpu_model_runner.py`（或 V2 的对应目录）：`execute_model()`、输入准备
- `vllm/v1/worker/gpu_input_batch.py`：持久化批（persistent batch）
- `vllm/v1/sample/`：采样器
- `vllm/compilation/`、`vllm/config/compilation.py`：`CUDAGraphMode`、捕获尺寸

**nano-vllm 对应**：`model_runner.py`，`prepare_prefill` / `prepare_decode` 构造 `input_ids`、`positions`、`slot_mapping`、`block_tables`、`cu_seqlens`；decode 时用 CUDA Graph，按 batch size 分桶捕获。

**问题清单**

1. 一步执行要准备哪些输入张量？`slot_mapping`、`query_start_loc`、`seq_lens`、`block_table` 各自是什么？
2. 什么是持久化批？为什么每步只更新变化的部分，而不是每步重建？
3. 显存 profile 是怎么做的？怎么保证 CUDA Graph、激活值、采样都留出了空间？
4. `CUDAGraphMode` 有哪几种（NONE、PIECEWISE、FULL、FULL_DECODE_ONLY、FULL_AND_PIECEWISE）？分别在什么场景用？
5. piecewise 模式为什么把 attention 留在图外？full 模式又需要 attention backend 提供什么能力？
6. 运行时 batch 大小不在捕获尺寸里时怎么办？padding 会带来什么代价？
7. `torch.compile` 在这里做了什么？和 CUDA Graph 是什么关系？
8. 采样在哪做？logits 处理（温度、top-k、top-p、惩罚项）的顺序是什么？

---

## 第 5 节：Attention backend（第 3 周，1 天）

**读什么**

- `vllm/v1/attention/backends/`（或 `vllm/attention/backends/`）：`flash_attn.py`、`flashinfer.py`、`triton_attn.py`
- 三个抽象：`AttentionBackend`（工厂）、`AttentionMetadataBuilder`（每步构造元数据）、`AttentionImpl`（真正算）
- attention 选择逻辑：`rg -n "def get_attn_backend" vllm`

**nano-vllm 对应**：`layers/attention.py`，prefill 用 `flash_attn_varlen_func`，decode 用 `flash_attn_with_kvcache`，写 KV 用一个 Triton kernel 按 `slot_mapping` 写入。

**问题清单**

1. 为什么要把「元数据构造」和「计算」分成两个类？（提示：元数据每步算一次、所有层共享；也方便做 CUDA Graph）
2. 同一个 batch 里既有 prefill 又有 decode 时，一次 varlen 调用怎么同时处理？
3. KV 写入 cache 发生在哪？用的是什么 kernel？
4. FlashAttention 和 FlashInfer 的 backend 各有什么限制（块大小、head size、是否支持 full CUDA Graph）？
5. MLA（DeepSeek 系）的 backend 和普通 MHA/GQA 有什么不同？

---

## 第 6 节：投机解码（第 3 周，0.5 天）

**读什么**

- `vllm/v1/spec_decode/`：`ngram_proposer.py`、`eagle.py`（含 MTP）、`metrics.py`
- `vllm/v1/sample/rejection_sampler.py`
- 调度器里给 draft token 预留位置的逻辑：`rg -n "spec_token|num_lookahead" vllm/v1/core`

**nano-vllm 对应**：你在 `back` 上写的 `ngram_speculation.py` 和 `run_ngram`（只支持贪心）。

**问题清单**

1. draft token 是在哪一步提出的？验证是在哪一步做的？一步里能同时「验证上一轮 draft」和「提出下一轮 draft」吗？
2. 拒绝采样怎么保证输出分布和不投机时一致？贪心时退化成什么？
3. n-gram、EAGLE、MTP 三种方法的 draft 从哪来？成本分别是多少？
4. 调度器怎么给 draft token 分配 KV slot？被拒绝的 token 占的 slot 怎么处理？
5. 为什么并发高时投机解码收益会下降甚至变负？

---

## 第 7 节：分布式与平台插件（第 3 周，1.5 天）

**读什么**

- `vllm/distributed/parallel_state.py`：TP、PP、DP、EP 进程组
- `vllm/model_executor/layers/linear.py`：`ColumnParallelLinear`、`RowParallelLinear`
- `vllm/model_executor/layers/fused_moe/`：`FusedMoE` 层、modular kernel 的 prepare / finalize 抽象（**vllm-ascend 的 `prepare_finalize.py`、`token_dispatcher.py` 就是对这里的昇腾实现**）
- `vllm/distributed/kv_transfer/`：KV connector，PD 分离的基础
- `vllm/platforms/interface.py`：`Platform` 基类；`vllm/plugins/`：插件加载

**问题清单**

1. TP 下一层 Transformer 需要几次 all-reduce？分别在哪？
2. EP 下 MoE 层的 dispatch 和 combine 各做什么通信？数据量和什么有关？
3. DP attention + EP 是什么组合？为什么 DeepSeek 这类 MLA 模型偏好这种方式？
4. KV connector 提供了哪些钩子？prefill 实例和 decode 实例之间 KV 是怎么传的？
5. **平台插件**：vLLM 通过什么机制发现外部平台（entry point 组名）？`Platform` 类上有哪些方法是硬件相关的？至少列出：设备类型、attention backend 选择、静态图 wrapper（vllm-ascend 文档里提到的 `get_static_graph_wrapper_cls`）、worker 类、设备通信器、显存和随机数相关的接口。

**本节产出**：`平台插件接口清单`，每个接口写一句「它决定了什么」。第 4 周读 `vllm_ascend/platform.py` 时，逐项对照这份清单看昇腾是怎么实现的。

---

## 第 3 周的 4090 D 实验（开机一次，约 2–3 小时）

复用项目 A 的压测脚本，只起 vLLM，验证源码阅读中的结论。

| 实验 | 做法 | 要验证的结论 |
|---|---|---|
| token 预算扫描 | S3 场景，`--max-num-batched-tokens` 取 2048 / 8192 / 16384 | 预算小 → ITL 稳但 TTFT 高；预算大 → 反之 |
| prefix caching 开关 | S4 场景，加或不加 `--no-enable-prefix-caching` | 命中时 TTFT 大幅下降，命中率可在 `/metrics` 里看到 |
| CUDA Graph 开关 | S2 场景，`--max-concurrency 1` 和 `64`，加或不加 `--enforce-eager` | 小 batch 时 CUDA Graph 收益大（kernel 下发开销占比高），大 batch 时收益变小 |
| decode profile | 用 vLLM 的 torch profiler 支持（按 `v0.31.0` 文档配置 profiler 目录，再用 `vllm bench serve --profile`）或 `nsys` | decode 每步时间里 GEMM 占大头，且接近「权重字节数 / 显存带宽」的下限 |

最后一项要自己算一下下限：Qwen3-4B BF16 权重约 8GB，4090 D 带宽约 1 TB/s，单步 decode 至少约 8ms。实测值和这个下限的比值，就是你的「带宽利用率」，面试时可以讲。

---

## 产出物

1. `笔记-vLLM.md`：7 节问题清单的答案，每节一张小图。
2. nano-vllm ↔ vLLM 对照表（模板如下）。
3. 平台插件接口清单。
4. 4090 D 实验的结果表和一句话结论。

### 对照表模板

| 概念 | nano-vllm | vLLM V1 | vLLM 多了什么 / 为什么 |
|---|---|---|---|
| 进程结构 | 单进程 `LLMEngine` | 前端进程 + `EngineCore` 进程 | |
| 调度粒度 | 整步 prefill 或整步 decode（`back` 上有 hybrid） | 统一 token 预算，按 `num_computed_tokens` 分配 | |
| chunked prefill | 无（hybrid 部分实现） | 默认开启 | |
| 抢占 | | 重算 | |
| 块大小 | 256 | 默认 16 | |
| 前缀缓存 | 链式哈希、满块 | 链式哈希 + 额外 key、LRU 空闲队列 | |
| 输入准备 | 每步重建 | 持久化批 | |
| CUDA Graph | 仅 decode，按 bs 分桶 | 多种模式，piecewise / full | |
| attention | flash-attn 两个函数 | 可插拔 backend，元数据 builder | |
| 投机解码 | n-gram，仅贪心 | n-gram / EAGLE / MTP，拒绝采样 | |
| 并行 | TP | TP / PP / DP / EP、KV connector | |
| 硬件扩展 | 无 | 平台插件 | |
