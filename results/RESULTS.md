# 项目 A 评测结果（4090 D / Qwen3-4B）

硬件：AutoDL RTX 4090 D 24GB，驱动 580.105.08  
模型：Qwen3-4B BF16  
客户端：两侧都是 `vllm bench serve`（vLLM 0.31.0）  
SLO：TTFT ≤ 500ms 且 TPOT ≤ 50ms  
对齐配置：`max_model_len=4096`，`max_num_seqs=256`，`max_num_batched_tokens=8192`，`gpu_mem=0.9`  
原始 JSON：`results/v2/4090d/`  
日期：2026-11-17，单次跑（未做三次中位数）

---

## 总表（面试先看这一张）

在线口径 S3（ShareGPT，同一组到达速率）：

| 引擎 / 变体 | 4.11 goodput | 8.22 goodput | 12.33 goodput | 14.80 goodput | 8.22 ITL p99 |
|---|---|---|---|---|---|
| nano 原版 prefill_priority | 3.91 | **7.26** | 1.81 | 0.66 | 110 |
| nano token_budget=512 | 3.95 | 4.94 | 0.86 | 0.34 | 109 |
| nano hybrid=512 | **3.96** | 7.22 | 1.67 | 1.10 | **61** |
| nano INT8 KV + hybrid | 0 | 0 | 0 | 0 | 307 |
| vLLM 0.31 对齐 | **4.00** | **7.56** | **10.68** | **12.32** | **36** |

饱和口径（rate=inf，goodput≈0 是排队，只比吞吐）：

| | S1 req/s | S2 out tok/s | S4 out tok/s |
|---|---|---|---|
| nano 原版 | 6.32 | 1672 | 2779 |
| nano token_budget | 4.10 | 1462 | 2088 |
| nano hybrid | 4.83 | 1641 | 2090 |
| nano INT8 | 2.26 | 596 | 924 |
| vLLM 对齐 | **7.54** | **3758** | **4029** |

精度 GSM8K 5-shot completions（strict-match）：

| | strict | flexible |
|---|---|---|
| nano 原版 | **85.82%** | 84.61% |
| nano INT8 KV | 85.29% | 84.61% |
| vLLM 对齐 | 84.99% | 84.61% |

一句话：原版 nano 在线 SLO 拐点大约 8 req/s，vLLM 大约 15 req/s。hybrid 把低负载 ITL p99 压下来了，但没有把拐点推到 12 req/s。INT8 和 n-gram 在这次实现/配置下没有可报的加速。

---

## 每个场景

### S1 prefill 重（random 1024→128，inf，1000 prompts）

| | req/s | out tok/s | ITL p50 / p99 |
|---|---|---|---|
| nano 原版 | 6.32 | 809 | 38.6 / 1231 |
| token_budget | 4.10 | 525 | — / 261 |
| hybrid | 4.83 | 618 | — / 235 |
| INT8 | 2.26 | 289 | — / 408 |
| vLLM 对齐 | 7.54 | 966 | 25.6 / 515 |
| vLLM 默认 | 7.24 | 926 | 26.7 / 163 |

token_budget / hybrid 把 ITL p99 从 1231ms 打到 200–260ms，符合「prefill 不再整步独占」。代价是饱和 req/s 下降。

### S2 decode 重（random 128→1024，inf，500 prompts）

| | out tok/s | TPOT p50 |
|---|---|---|
| nano 原版 | 1672 | 111 |
| token_budget | 1462 | 140 |
| hybrid | 1641 | 104 |
| INT8 | 596 | 315 |
| vLLM 对齐 | 3758 | 46.7 |

decode 是访存瓶颈。nano 原版只有 vLLM 的 44%。INT8 反量化把 decode 又砍掉 2/3。

### S3 在线 ShareGPT（四档速率，每档 600 prompts）

R_max 来自 vLLM 默认 S3 inf：**16.44 req/s**。四档 4.11 / 8.22 / 12.33 / 14.80。

详见总表。补充延迟：

**4.11 req/s（低负载）**

| | goodput | TTFT p99 | TPOT p50 | ITL p99 |
|---|---|---|---|---|
| 原版 | 3.91 | 653 | 16.9 | 65 |
| token_budget | 3.95 | 179 | 17.5 | 61 |
| hybrid | 3.96 | 120 | 14.9 | **41** |
| vLLM | 4.00 | 94 | 11.4 | 26 |

**8.22 req/s（nano 拐点）**

| | goodput | TTFT p99 | TPOT p50 | ITL p99 |
|---|---|---|---|---|
| 原版 | **7.26** | 122 | 25.4 | 110 |
| token_budget | 4.94 | 513 | 40.0 | 109 |
| hybrid | 7.22 | 228 | 24.6 | **61** |
| vLLM | 7.56 | 112 | 13.7 | 36 |

**12.33 / 14.80**：nano 各变体 TTFT p99 都到秒级，goodput 崩溃；vLLM 14.80 仍 goodput 12.32、TTFT p99 175ms。

### S4 共享前缀（prefix 1024，inf，256 prompts）

| | out tok/s | TTFT p50 |
|---|---|---|
| 原版 | 2779 | 3216 |
| token_budget / hybrid | ~2090 | ~4800 |
| INT8 | 924 | 6585 |
| vLLM | 4029 | 1705 |

nano block size=256，vLLM=16，前缀命中粒度不同。

### S6 n-gram（InstructCoder，conc=1 / 4）

| | conc=1 out tok/s | TPOT p50 | conc=4 out tok/s | 接受率 |
|---|---|---|---|---|
| nano hybrid+ngram | 40 | 22.6 | 46 | **0**（draft_tokens=0） |
| vLLM ngram | 96 | 10.4 | 338 | ~8.9% |

nano 侧 n-gram 和 hybrid 叠在一起时没有真正 draft，这次 **不能** 当投机解码成功案例讲。vLLM 侧 conc=4 的 TTFT p99 从 65ms 跳到 1958ms，符合「高并发投机收益消失、长尾先坏」。

### S0

按计划外的要求跳过 0.6B，没有复现 6153 tok/s。

---

## 每项优化四句话

### 原版 prefill 优先

做了什么：一步里要么全 prefill 要么全 decode。  
场景：S3、S1。  
现象：8.22 还能吃满到达速率；12.33 起 TTFT p99 到 11s。S1 ITL p99=1231ms。  
为什么：新请求的长 prefill 独占一步，decode 被卡住。

### token_budget=512

做了什么：每步 prefill ≤512 token，再和 decode 轮转（同一步仍是纯 P 或纯 D）。  
场景：S1、S3。  
现象：S1 ITL p99 1231→261；S3 8.22 goodput 7.26→4.94，TTFT p99 升到 513ms。  
代价：用 TTFT / 吞吐换 ITL。预算偏小，在线 SLO 变差。

### hybrid=512

做了什么：同一步先给 decode，剩余预算给 prefill 分块。  
场景：S3、S1。  
现象：S3 4.11 ITL p99 65→41，8.22 ITL p99 110→61，goodput 几乎不变；S1 ITL p99 1231→235，饱和 req/s 6.32→4.83。  
为什么：和 vLLM chunked prefill 同思路。这次把卡顿压下去了，但没有把可持续 QPS 拉到 vLLM 附近。  
代价：prefill 吞吐下降；还没追上 vLLM 的 kernel / Graph / 异步调度。

### INT8 KV

做了什么：写入量化、读取反量化。  
场景：S2、S3、GSM8K。  
现象：KV 块数 337→**219（×0.65）**，不是旧口径的 ×1.78。S2 1672→596 tok/s。GSM8K 85.82%→85.29%（掉 0.5 点，精度过关）。  
代价：当前 per-group scale 把容量吃掉了，再加反量化计算，高负载 goodput 为 0。**这次不报加速比。**

### n-gram

做了什么：prompt lookup，K=4。  
场景：S6。  
现象：nano 侧 `ngram_draft_tokens=0`；vLLM conc=1 TPOT 10ms，conc=4 TTFT p99 爆到 2s。  
代价：nano 要改成纯 decode 路径再测；不要把 0 接受率写成「投机无效」，那是配置没打到投机路径。

### decode pack

旧引擎内消融已确认无收益，新口径未重测。差异预期在噪声内，不报加速比。

---

## 正确性

- 调度类应对齐 HF greedy：4B 上 `align_hf.py` 原版已经只有 1/4 exact（transformers 4.57 + flash-attn 与引擎分叉）。不能用 HF 当 4B 的 100% 金标。引擎内 GPU correctness 指纹此前通过。
- INT8 允许非 bit-exact；GSM8K 掉点 <1。
- 三次中位数未做，极差未知。差几个点的吞吐不要说成「稳定加速」。

---

## 8B 抽查

Qwen3-8B，S3 rate=4.11，300 prompts，`max_num_seqs=128`，`max_num_batched_tokens=4096`（24GB 上 8B 权重约 16GB，KV 很紧）。

| | goodput | TTFT p50 / p99 | TPOT p50 | ITL p99 | out tok/s |
|---|---|---|---|---|---|
| nano hybrid | **3.52** | 77 / 1192 | 27.4 | 76 | 789 |
| vLLM 对齐 | 2.25 | 126 / **13271** | 24.4 | 95 | 813 |

抽查结论：低速率下两边 req/s 都约 3.6–3.7，接近到达速率；vLLM 的 TTFT p99=13s 像是 KV 不够触发抢占/排队，**不能**据此说 nano 反超 8B。4B 上的「vLLM 在线容量大约高一倍」仍然是主结论。需要更小的 `max_model_len` 或更少并发再比 8B。

---

## 未做

- S0 0.6B
- S1/S3 三次中位数
- S5 混合双客户端
- GitHub 默认分支改 `back`（本机无 `gh`、GitHub API 不通）
- nano n-gram 在纯 decode 策略下的对照
- 把 INT8 scale 布局改回能涨 KV 容量的实现
