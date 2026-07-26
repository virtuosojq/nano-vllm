# nano-vllm 学习路径

> 面向初学者的循序渐进学习路线。整个项目约 1200 行代码，分 **8 个阶段**，每个阶段聚焦一个核心模块。
> 建议每个阶段：**读代码 → 理解概念 → 回答自检问题 → 进入下一阶段**。

---

## 阶段 0：项目全景与运行环境

**目标**：搞清楚"它是什么、能做什么"，先把项目跑起来获得感性认识。

- 阅读：[README.md](file:///root/autodl-tmp/nano-vllm/README.md)、[pyproject.toml](file:///root/autodl-tmp/nano-vllm/pyproject.toml)
- 运行：[example.py](file:///root/autodl-tmp/nano-vllm/example.py)（先 `enforce_eager=True` 跑通）
- 关键认知：
  - nano-vllm 是 vLLM 的极简复刻，约 1200 行 Python
  - 当前仅支持 Qwen3 系列
  - 四大支柱：**Continuous Batching / Paged KV Cache / Prefix Caching / TP + CUDA Graph**

**自检**：能用一句话说清 nano-vllm 与 vLLM 的关系吗？依赖了哪些关键库（torch、triton、flash-attn、xxhash）？

---

## 阶段 1：对外 API 层（最外层入口）

**目标**：理解一次 `llm.generate(prompts, sp)` 是怎么从外部接入的。

- 文件：
  - [nanovllm/__init__.py](file:///root/autodl-tmp/nano-vllm/nanovllm/__init__.py)（仅 2 行，导出 `LLM` 和 `SamplingParams`）
  - [nanovllm/llm.py](file:///root/autodl-tmp/nano-vllm/nanovllm/llm.py)（`LLM` 只是 `LLMEngine` 的空壳）
  - [nanovllm/sampling_params.py](file:///root/autodl-tmp/nano-vllm/nanovllm/sampling_params.py)（temperature / max_tokens / ignore_eos）
- 关键认知：`LLM` 的所有真实逻辑都在父类 `LLMEngine` 中。

**自检**：为什么 `LLM` 类只有一个 `pass`？这样设计有何好处？

---

## 阶段 2：引擎主循环（编排核心）

**目标**：理解 [llm_engine.py](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/llm_engine.py) 如何把"调度器"和"模型执行器"串起来。

- 重点方法：
  - `__init__`：启动 TP worker 进程、建 tokenizer、建 scheduler
  - `add_request`：把 prompt 编码后包装成 `Sequence` 丢给 scheduler
  - `step`（[L49-55](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/llm_engine.py#L49-L55)）：**引擎心跳**，三步走 `schedule → run → postprocess`
  - `generate`：循环调用 `step()` 直到所有序列完成

- 核心数据流：
  ```
  generate() → add_request() → [scheduler.waiting]
       ↓
  循环 step():
    scheduler.schedule()  →  返回 [Sequence], is_prefill
    model_runner.run()    →  返回 token_ids
    scheduler.postprocess() → 更新序列状态
  ```

**自检**：`num_tokens` 在 prefill 和 decode 阶段分别取什么值？为什么 decode 用负数（[L51](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/llm_engine.py#L51)）？

---

## 阶段 3：序列状态（基础数据结构）

**目标**：理解一条请求的全生命周期状态。这是后续所有模块的操作对象。

- 文件：[sequence.py](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/sequence.py)
- 关键概念：
  - 三种状态 `WAITING → RUNNING → FINISHED`
  - 字段含义：`num_tokens / num_prompt_tokens / num_cached_tokens / num_scheduled_tokens`
  - `block_table`：该序列引用的 KV 块 ID 列表（PagedAttention 的灵魂）
  - `__getstate__ / __setstate__`：为 TP 多进程传输做的精简序列化（只传必要字段）

**自检**：`prompt_token_ids` 和 `completion_token_ids` 的分界线是哪个字段？为什么序列化时要区分 prefill 和 decode（[L72-83](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/sequence.py#L72-L83)）？

---

## 阶段 4：调度器（Continuous Batching）

**目标**：理解多请求如何动态拼批，这是 vLLM 性能的第一来源。

- 文件：[scheduler.py](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/scheduler.py)
- 关键方法：
  - `schedule`（[L25-73](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/scheduler.py#L25-L73)）：**优先 prefill，其次 decode**
    - prefill 阶段受 `max_num_batched_tokens / max_num_seqs` 约束，仅对首条允许 **chunked prefill**
    - decode 阶段每条序列出 1 token；显存不够时触发 `preempt`
  - `preempt`（[L75-79](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/scheduler.py#L75-L79)）：释放该序列的 KV 块，退回 waiting 重算
  - `postprocess`（[L81-92](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/scheduler.py#L81-L92)）：登记哈希、追加 token、判断 EOS/max_tokens 终止

**自检**：为什么 chunked prefill 只允许作用于队列首条序列？preempt 为什么是"释放重算"而不是"换出到 CPU"？

---

## 阶段 5：块管理器（最核心、最精彩）

**目标**：吃透 Paged KV Cache + Prefix Caching 的实现。**这是本项目的精华，建议反复读**。

- 文件：[block_manager.py](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/block_manager.py)
- 两个核心抽象：
  1. **分页管理**：把 KV 显存切成等大块（默认 256 token/块），序列通过 `block_table` 非连续引用，按 `ref_count` 回收
  2. **Prefix Caching**：用 xxhash 对填满的块算哈希，且**把前一块的哈希作为后一块哈希的输入**形成链式前缀；新请求 prefill 前先比对哈希命中即可复用 KV

- 按方法逐个攻克：
  - `Block` 类（[L8-23](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/block_manager.py#L8-L23)）：块的内部状态
  - `compute_hash`（[L35-41](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/block_manager.py#L35-L41)）：链式哈希
  - `can_allocate / allocate`（[L58-92](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/block_manager.py#L58-L92)）：prefill 前的查询与分配
  - `can_append / may_append`（[L103-108](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/block_manager.py#L103-L108)）：decode 时按需追加块
  - `hash_blocks`（[L110-119](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/block_manager.py#L110-L119)）：把新填满的块登记入哈希表

**自检**：为什么哈希要带 `prefix` 参数做成链式？`can_allocate` 返回 -1 表示什么？为什么最后一个块不参与哈希计算？

---

## 阶段 6：模型执行器（GPU 上的真活）

**目标**：理解 GPU 端如何准备张量、跑前向、采样，以及 TP 与 CUDA Graph 的集成。

- 文件：[model_runner.py](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/model_runner.py)
- 分四块读：
  1. **初始化**（`__init__`、`allocate_kv_cache`、`warmup_model`）：NCCL 初始化、根据剩余显存反推 KV 块数
  2. **数据准备**（`prepare_prefill / prepare_decode / prepare_sample`）：把 Sequence 列表拍平成张量，计算 `slot_mapping`、`cu_seqlens` 等
  3. **执行**（`run_model` [L195-212](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/model_runner.py#L195-L212)）：prefill 直接跑；decode 走 CUDA Graph 回放
  4. **CUDA Graph 捕获**（`capture_cudagraph` [L222-256](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/model_runner.py#L222-L256)）：按多个 batch size 预捕获图
  5. **TP 通信**（`loop / read_shm / write_shm / call` [L61-89](file:///root/autodl-tmp/nano-vllm/nanovllm/engine/model_runner.py#L61-L89)）：rank0 通过共享内存广播方法调用给其他 rank

- 配合阅读：[utils/context.py](file:///root/autodl-tmp/nano-vllm/nanovllm/utils/context.py)（用全局 Context 在 layers 与 runner 间传参）

**自检**：`slot_mapping` 的作用是什么？CUDA Graph 为什么只用于 decode 不用于 prefill？TP 下 rank0 和 rank>0 的代码路径有何不同？

---

## 阶段 7：模型与算子层（底层实现）

**目标**：理解模型结构、TP 切分、Attention/FlashAttn 集成。

- 模型：[models/qwen3.py](file:///root/autodl-tmp/nano-vllm/nanovllm/models/qwen3.py)
  - `Qwen3DecoderLayer`：标准 pre-norm 残差结构
  - `packed_modules_mapping`：配合 [utils/loader.py](file:///root/autodl-tmp/nano-vllm/nanovllm/utils/loader.py) 把 HF 的 q/k/v/gate/up 权重合并加载到 packed 参数

- 算子（按需深入）：
  - [layers/linear.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/linear.py)：Column/Row/QKV/Merged 四种 TP 并行 Linear
  - [layers/attention.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/attention.py)：prefill 用 `flash_attn_varlen_func`，decode 用 `flash_attn_with_kvcache`；Triton 写的 `store_kvcache_kernel` 把 K/V 散写到 paged cache
  - 其他：[rotary_embedding.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/rotary_embedding.py)、[layernorm.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/layernorm.py)、[activation.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/activation.py)、[sampler.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/sampler.py)、[embed_head.py](file:///root/autodl-tmp/nano-vllm/nanovllm/layers/embed_head.py)

**自检**：ColumnParallel 和 RowParallel 的 TP 切分维度有何不同？为什么 RowParallel 需要 `all_reduce`？

---

## 推荐学习顺序图

```
外层入口                      编排层                    数据/控制层                  底层算子
─────────────────────────────────────────────────────────────────────────────────
[阶段1] API              →    [阶段2] 引擎主循环    →    [阶段4] Scheduler       →   [阶段7] 模型/算子
   LLM.generate()              step() 三步走              Continuous Batching        Qwen3/Attention/Linear
                                     ↓                          ↓
                                [阶段3] Sequence        →    [阶段5] BlockManager  →   [阶段6] ModelRunner
                                状态机基础                  Paged KV + PrefixCache     TP + CUDA Graph
```

## 学习建议

1. **第一遍粗读**：按阶段 0→7 走马观花，建立全局地图，不纠结细节
2. **第二遍精读**：重点啃 **阶段 4 / 5 / 6**（Scheduler + BlockManager + ModelRunner），这是引擎精髓
3. **动手验证**：
   - 在 `step()` 加日志，打印每步是 prefill 还是 decode、batch 大小
   - 把 `enforce_eager` 切换 True/False 对比 CUDA Graph 加速效果
   - 用相同 system prompt 发多条请求，观察 `num_cached_tokens` 变化验证 Prefix Caching
4. **画图辅助**：自己画一遍 block_table / slot_mapping / KV cache 三者关系图
