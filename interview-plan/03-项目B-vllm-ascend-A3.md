# 03 项目 B：A3 上的 vllm-ascend MoE 推理剖析与优化（第 4–6 周）

## 目标与面试故事

**一句话**：在 Atlas A3（910C）上用 vllm-ascend 部署 Qwen3-30B-A3B（MoE），用和 vLLM 一致的评测口径建基线，剖析出 attention、MoE 计算、专家通信各自的耗时，并从调度层一路追到 MC2 通信算子，做出至少一项有数字、能讲清原理的优化。

**为什么这个故事有竞争力**：它把三层串起来了：

1. **引擎层**（vLLM 调度、KV cache、图模式）——第 2–3 周读过的东西
2. **插件层**（vllm-ascend 怎么替换 vLLM 的执行层、MoE 通信方式怎么选）
3. **算子与通信层**（`torch_npu` 算子、`ops-transformer/mc2`、HCCL）——你已有的工作经验

**完成标准**：

- 至少 3 个配置的基线数字（同一口径、同一组场景）
- 一张 prefill 和 decode 的耗时拆解表
- 一张「MoE 层从 vLLM 调度到 NPU kernel」的全链路图
- 至少一项优化的前后对比 + 原理解释；或者（降级）一份完整的性能剖析报告

---

## 一、模型和配置

### 为什么选 Qwen3-30B-A3B

- MoE 结构：总参数 30.5B，每个 token 激活 3.3B；48 层，128 个专家，每个 token 选 8 个。正好落在 EP、MoE 通信、MC2 上。
- vllm-ascend 有官方教程（本地 `docs/source/tutorials/models/Qwen3-30B-A3B.md`），有现成的 W8A8 权重和 EAGLE3 draft 模型，环境问题最少。
- 官方给了参照精度：A3 上 W8A8 的 GSM8K（4-shot CoT，AISBench）为 92.87%，可以用来检验你的部署是否正常。

### 显存估算（面试时也会被问）

- BF16 权重：30.5B × 2 字节 ≈ 61GB
- W8A8 权重：约 31GB
- KV cache 每个 token：`2 × 48 层 × 4 个 KV 头 × 128 head_dim × 2 字节 = 98,304 字节 ≈ 96KB`
- A3 一台 8 卡、双 die 设计，系统里是 16 个 `/dev/davinci*`。每个 die 的显存以 `npu-smi info` 为准（官方教程按 64GB 写）。
- 结论：**BF16 一个 die 放不下**（权重之外还要留 KV 和激活），至少要 2 个 die；**W8A8 一个 die 可以跑**，剩余约 30GB 给 KV，大约能放 30 万个 token。

### 配置矩阵

| 编号 | 权重 | die 数 | 并行 | 图模式 | MoE 通信（预期） | 用途 |
|---|---|---|---|---|---|---|
| C1 | W8A8 | 1 | TP1 | ACL Graph | 无跨卡通信 | 单 die 基线，没有通信干扰 |
| C2 | BF16 | 2 | TP2 | ACL Graph | ALLGATHER | TP 基线 |
| C3 | BF16 | 2 | TP2 + EP | ACL Graph | 按 token 数在 ALLTOALL / MC2 间选择 | EP 和 MC2 的效果 |
| C4 | 同 C3 | 2 | TP2 + EP | eager（`--enforce-eager`） | 同 C3 | 图模式的效果 |
| C5（可选） | W8A8 | 2 | TP2 + EP | ACL Graph | 同 C3 | 量化 + EP |
| C6（可选） | 同 C3 或 C5 | 2 | TP2 + EP | ACL Graph | 同 C3 | EAGLE3 投机解码，和项目 A 的 n-gram 呼应 |

「MoE 通信（预期）」一列来自代码：`vllm_ascend/ascend_forward_context.py` 的 `select_moe_comm_method()`，不开 EP 或 EP 组大小为 1 时用 ALLGATHER；开了 EP 后按硬件策略、当前 token 数和 MC2 容量选择。实际用了哪种，要从日志或 profiler 里确认。

**比较不同 die 数的配置时，要看「每个 die 的吞吐」**，否则 C2/C3 用了两倍的硬件，直接比总吞吐没有意义。

---

## 二、第 4 周：跑通、基线、读插件层

### 2.1 起容器（只映射要用的 die）

在官方教程命令的基础上，只映射 2 个 die，避免影响机器上的其他人：

```bash
export IMAGE=quay.io/ascend/vllm-ascend:<版本>-a3   # 选和 vLLM v0.31.0 匹配的版本；国内可加 m.daocloud.io/ 前缀
docker run --name vllm-ascend-env --ipc host --net host \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/sbin:/usr/local/sbin \
  -v /path/to/models:/models -v /path/to/work:/work \
  -it -d $IMAGE bash
```

具体用哪两个 die，按 `npu-smi info` 里空闲的来。

### 2.2 起服务

参照官方教程的单机部署命令，按配置矩阵改并行度：

```bash
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"          # A3 需要，A2 不需要
export ASCEND_RT_VISIBLE_DEVICES=0,1         # 容器内的编号
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# C3：BF16、TP2 + EP、只在 decode 用整图
vllm serve /models/Qwen3-30B-A3B --served-model-name Qwen3-30B-A3B \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --distributed_executor_backend mp \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --gpu-memory-utilization 0.9 --port 8000

# C1：W8A8、单 die：去掉 TP 和 EP，加 --quantization ascend，ASCEND_RT_VISIBLE_DEVICES=0
# C2：去掉 --enable-expert-parallel
# C4：加 --enforce-eager
# C6：加 --speculative-config '{"method": "eagle3", "model": "/models/Qwen3-a3B_eagle3", "num_speculative_tokens": 3}'
```

官方教程还用了 `--additional-config '{"weight_nz_mode": 2}'`（权重转 NZ 格式）和 `--no-enable-prefix-caching`。建议：**基线先完全照官方配置跑一次**，确认能复现正常性能；然后再按你的矩阵逐项改，每次只改一个变量。

### 2.3 基线压测

复用项目 A 的场景定义和客户端参数（vllm-ascend 的性能文档也明确用 vLLM 自带的 benchmark 保持对齐），模型名和 tokenizer 换成 Qwen3-30B-A3B：

- S1：random 1024→128，rate=inf
- S2：random 128→1024，rate=inf
- S3：ShareGPT，按 C3 的 `R_max` 取 4 个速率
- S6（C6 用）：InstructCoder，`--max-concurrency 1 / 4`

ShareGPT 数据集要提前传到 A3 上（`00-环境与工作流.md` 的同步方式）。

### 2.4 精度基线

两种工具选一种，C1 和 C3 各跑一次：

- **AISBench**（官方教程用的，能直接和 92.87% 对比）：`ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_str --mode all`
- **lm-eval**（和项目 A 用的一致）：参照本地 `docs/source/developer_guide/evaluation/using_lm_eval.md`

### 2.5 本周读的代码

按第 3 周的「平台插件接口清单」逐项对照：

| 文件 | 看什么 |
|---|---|
| `vllm_ascend/platform.py` | `NPUPlatform` 实现了清单里的哪些接口；attention backend、静态图 wrapper、worker 类分别指向哪 |
| `vllm_ascend/patch/` 和 `docs/.../Design_Documents/patch.md` | 对 vLLM 打了哪些补丁；挑 3 个，说出「vLLM 原来的抽象哪里不够用」 |
| `vllm_ascend/worker/worker.py`、`worker/v1/model_runner.py` | 和 vLLM 的 GPU model runner 比，输入准备、图捕获、KV 分配有哪些不同 |
| `vllm_ascend/attention/attention_v1.py` | 用的是哪个 `torch_npu` 注意力算子；prefill 和 decode 是否走不同算子 |
| `docs/.../Design_Documents/ModelRunner_prepare_inputs.md` | 官方对输入准备的说明 |

---

## 三、第 5 周：剖析

### 3.1 配置对比

先把配置矩阵的数据跑齐，回答：

1. C1 vs C2：TP2 带来的每 die 吞吐损失有多大？损失来自哪里？
2. C2 vs C3：开 EP 后 decode 的 TPOT 变化多少？prefill 呢？
3. C3 vs C4：ACL Graph 对 decode 的收益有多大？在小并发和大并发下分别如何？
4. 哪个配置的 goodput 最高？

### 3.2 Profiler 时间线

参照本地 `docs/source/developer_guide/performance_and_debug/service_profiling_guide.md`：

- **Ascend PyTorch Profiler**：看算子级时间线，用 MindStudio Insight 打开。分别抓一段纯 prefill（S1）和一段纯 decode（S2）。
- **MS Service Profiler**：看服务级（调度、排队、批组成），用来解释 TTFT。

产出一张耗时拆解表：

| 阶段 | attention | MoE 路由 / 重排 | MoE 专家计算（GMM） | dispatch / combine 通信 | 其他计算（norm、RoPE 等） | 空闲（host 下发等待） |
|---|---|---|---|---|---|---|
| prefill（C3） | | | | | | |
| decode（C3） | | | | | | |
| decode（C4，eager） | | | | | | |

要能回答：

1. decode 是计算瓶颈、访存瓶颈、通信瓶颈，还是 host 下发瓶颈？（时间线上 kernel 之间有大量空隙，就是 host 瓶颈，这也是 ACL Graph 的价值所在）
2. dispatch / combine 在 prefill 和 decode 中各占多少？为什么不同？
3. 通信和计算有没有重叠？

### 3.3 MoE 全链路追踪

这是项目 B 最核心的部分，也是你最有优势的部分。沿着下面的路径读代码，每一层写一句「这一层做了什么决定」：

```
vLLM: model_executor/layers/fused_moe/  FusedMoE 层（通用抽象）
  │
  ▼
vllm-ascend: ops/fused_moe/fused_moe.py      昇腾版 FusedMoE
  │
  ├── ascend_forward_context.py: select_moe_comm_method()
  │     不开 EP → ALLGATHER；开 EP → 按策略、token 数、MC2 容量选 ALLTOALL / MC2 / FUSED_MC2
  │
  ├── ops/fused_moe/moe_comm_method.py
  │     AllGatherCommImpl / AlltoAllCommImpl / MC2CommImpl / FusedMC2CommImpl
  │
  ├── ops/fused_moe/token_dispatcher.py
  │     MC2 路径：torch_npu.npu_moe_distribute_dispatch_v2 / npu_moe_distribute_combine_v2
  │     AllGather 路径：torch_npu.npu_moe_init_routing_v2
  │     AlltoAll 路径：torch_npu.npu_moe_token_permute / npu_moe_token_unpermute
  │
  ├── ops/fused_moe/moe_mlp.py
  │     专家 FFN：grouped matmul + torch_npu.npu_swiglu
  │
  ▼
torch_npu 算子注册（对照本地 /home/wjq/opsCode/op-plugin）
  │
  ▼
CANN 算子（本地 /home/wjq/opsCode/ops-transformer）
      mc2/moe_distribute_dispatch_v2、mc2/moe_distribute_combine_v2
      moe/ 下的路由与重排算子、gmm/ 下的 grouped matmul
```

追踪时要回答的问题：

1. 为什么 decode 倾向用 MC2，而 prefill 的 token 多时用 ALLTOALL？（提示：MC2 有 token 容量上限；decode 的 token 少、延迟敏感，适合把通信和计算融合进一个算子，减少 host 同步）
2. FUSED_MC2 比 MC2 多融合了什么？（`moe_comm_method.py` 的注释提到 dispatch、FFN、combine 融合成一个算子）
3. `moe_distribute_dispatch_v2` 在 kernel 内部是怎么做通信的？和 HCCL 集合通信库调用有什么区别？（这一层用你在 `ops-transformer` 和 HCCL 上的经验来讲）
4. 为什么 MC2 要求各 rank 的 token 数一致？（`vllm_ascend/utils.py` 里有相关注释）
5. ALLGATHER 路径在 TP 下多做了哪些冗余计算或通信？

### 3.4 降级判断点

**第 5 周末还没有稳定的基线**，项目 B 的目标改为「完整的性能剖析报告」：配置对比 + 耗时拆解 + 全链路分析 + 瓶颈判断。在面试中讲清楚「瓶颈在哪、为什么、如果有时间会怎么优化」同样有说服力。

---

## 四、第 6 周：优化与收尾

### 4.1 优化候选（选 1–2 项）

优先选能讲到算子或通信层的。每一项都要有前后对比数字和原理解释。

| 优先级 | 优化 | 做法 | 预期讲法 |
|---|---|---|---|
| 高 | MoE 通信方式 | 对比 ALLGATHER / ALLTOALL / MC2 在 prefill 和 decode 下的耗时；如果策略可配，尝试调整切换阈值 | 为什么小 token 数用 MC2 更好；切换点由什么决定 |
| 高 | 图模式与捕获尺寸 | 对比 eager / `FULL_DECODE_ONLY` / piecewise；调整 `cudagraph_capture_sizes` 贴合实际 batch 分布 | host 下发开销、padding 浪费、捕获尺寸受 runtime 资源限制（见 `ACL_Graph.md`） |
| 中 | host 侧优化 | 按 `optimization_and_tuning.md` 和 `cpu_binding.md`：tcmalloc、`torch_npu` 的下发队列相关环境变量、CPU 绑核 | 时间线上 kernel 空隙变小；对 host 瓶颈的场景效果明显 |
| 中 | 量化 | W8A8 vs BF16 的每 die 吞吐和精度 | 动态 W8A8 的激活量化开销（`npu_dynamic_quant`）；专家权重量化对访存的影响 |
| 中 | 调度预算 | `--max-num-batched-tokens` 扫描，和 4090 上的结论对比 | 同一机制在不同硬件上的权衡点为什么不同 |
| 低 | 投机解码 | C6：EAGLE3 在不同并发下的收益 | 和项目 A 的 n-gram 对照：draft 质量、成本、并发的影响 |
| 加分 | 提 PR | 从剖析中发现的问题修一个；或者用 `npugraph_ex` 的自定义 fusion pass 做一个小融合（见 `npugraph_ex.md`） | 能讲「我在社区提过什么」 |

### 4.2 收尾

- [ ] 项目 B 仓库的 `RESULTS.md`：配置矩阵结果表、耗时拆解表、优化前后对比
- [ ] 全链路图（可以用 Mermaid 或手画后拍照）
- [ ] 项目 B 的一页说明（模板见 `05-手写代码与系统设计.md`）
- [ ] 准备 10 个追问（见 05 的项目追问清单）

---

## 五、vllm-ascend 本地文档阅读清单

路径都在 `/home/wjq/tmp/vllm-ascend/docs/source/` 下。

| 文档 | 什么时候读 | 为什么 |
|---|---|---|
| `tutorials/models/Qwen3-30B-A3B.md` | 第 4 周第 1 天 | 部署命令、硬件需求、参照精度 |
| `developer_guide/performance_and_debug/performance_benchmark.md` | 第 4 周 | 压测方法，和 vLLM 对齐 |
| `developer_guide/evaluation/using_lm_eval.md`、`using_ais_bench.md` | 第 4 周 | 精度评测 |
| `developer_guide/Design_Documents/patch.md` | 第 4 周 | 补丁机制 |
| `developer_guide/Design_Documents/ModelRunner_prepare_inputs.md` | 第 4 周 | 输入准备 |
| `user_guide/configuration/additional_config.md`、`env_vars.md` | 第 4 周 | 昇腾专有配置项 |
| `developer_guide/Design_Documents/ACL_Graph.md` | 第 5 周 | 图模式在昇腾上的实现和限制 |
| `developer_guide/performance_and_debug/service_profiling_guide.md` | 第 5 周 | profiler 用法 |
| `developer_guide/performance_and_debug/optimization_and_tuning.md` | 第 6 周 | host 侧调优 |
| `developer_guide/Design_Documents/npugraph_ex.md` | 第 6 周（做融合时） | FX 图融合 pass |
| `developer_guide/Design_Documents/model_runner_v1_eplb.md` | 有余力时 | 专家负载均衡 |
| `developer_guide/Design_Documents/disaggregated_prefill.md` | 有余力时 | PD 分离在昇腾上的实现 |
| `developer_guide/Design_Documents/quantization.md` | 做量化对比时 | W8A8 的实现方式 |

---

## 本阶段要掌握的知识点

详细内容见 `04-面试知识点.md` 第 7、8 章，这里列出本阶段必须能讲清楚的：

- 910C / A3 的硬件组成：双 die、AI Core 里 Cube 和 Vector 的分工、片上存储层次、HCCS 互联
- ND 和 NZ 数据格式，为什么权重要转 NZ（`weight_nz_mode`）
- ACL Graph 和 CUDA Graph 的异同；FULL_DECODE_ONLY 为什么是常用选择
- MoE 推理的三种通信方式：AllGather、AlltoAll、MC2，各自的适用场景
- MC2 的含义：把计算和通信融合进同一个算子，减少同步和 host 介入
- EP 的负载不均问题和 EPLB（冗余专家）
- W8A8 静态量化和动态量化的区别（Qwen3-30B-A3B-W8A8 里 attention 用静态、专家用动态）
- `HCCL_BUFFSIZE`、`HCCL_OP_EXPANSION_MODE` 这两个环境变量是干什么的
