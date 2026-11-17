# 项目 A 一页说明（面试前过一遍）

平台：AutoDL RTX 4090 D · 模型：Qwen3-4B BF16 · 对照：vLLM 0.31.0  
测量：给 nano-vllm 包 OpenAI `/v1/completions`，两侧都用 `vllm bench serve`  
SLO：TTFT≤500ms，TPOT≤50ms · 日期：2026-11-17

---

【一句话】  
我在最小推理引擎 nano-vllm 上做了调度 / KV / 投机三类优化，并用 vLLM 官方压测工具在同一张 4090 D 上做同口径对比。原版在线 SLO 可持续吞吐大约是 vLLM 的一半（~8 vs ~15 req/s）；hybrid 改善了 ITL 长尾，但没有把 QPS 拐点推上去；INT8 和 n-gram 在这次实现里没有可报的加速。GSM8K 5-shot 原版 85.8%，INT8 85.3%，vLLM 85.0%。

【背景与动机】  
第一版评测在引擎 `step()` 后打点，离线 TTFT 其实是排队，没法和 vLLM 比。投推理引擎岗位必须先把测量做对，再用同一把尺子看优化有没有用。

【我做了什么】

1. **OpenAI 兼容服务**  
   流式增量、`stop` 截断并 abort 序列、`usage`。否则客户端测的 TTFT/ITL 全是错的。

2. **hybrid 调度（对应 vLLM chunked prefill）**  
   原版一步纯 prefill 或纯 decode，长 prompt 会卡住 decode（S1 ITL p99=1231ms）。改成同一步 decode 优先、剩余预算给 prefill 分块。S3 8.22 req/s 的 ITL p99 从 110ms 降到 61ms，goodput 仍约 7.2；饱和 req/s 从 6.32 降到 4.83。卡顿改善了，容量没追上 vLLM。

3. **token_budget**  
   每步 prefill≤512 再和 decode 轮转。S1 ITL p99 降到 261ms，但 S3 8.22 goodput 掉到 4.94，TTFT 换 ITL 的代价在线上不划算。

4. **INT8 KV**  
   写入量化、读取反量化。当前 per-group scale 把 KV 块数从 337 减到 219（×0.65），S2 吞吐 1672→596 tok/s。GSM8K 只掉 0.5 个点。精度过关，性能这次是负优化，不报加速比。

5. **评测重做本身**  
   用 inf 定饱和容量、用受控到达 + goodput 定线上 QPS。vLLM inf 有 16.4 req/s，但 TTFT 已经 2.5s，goodput≈0.16。饱和吞吐 ≠ 能承诺的 QPS。

【怎么验证的】  
`vllm bench serve`；ShareGPT / random / InstructCoder；goodput 阈值写死。精度用 lm-eval GSM8K 5-shot，不用逐 token 对齐验收量化。单次跑，没有三次中位数。

【结果】见下表（S3 ShareGPT）：

| | 8.22 goodput | 8.22 ITL p99 | 14.80 goodput | GSM8K strict |
|---|---|---|---|---|
| nano 原版 | 7.26 | 110ms | 0.66 | 85.82% |
| nano hybrid | 7.22 | 61ms | 1.10 | — |
| nano INT8 | 0 | 307ms | 0 | 85.29% |
| vLLM 对齐 | 7.56 | 36ms | **12.32** | 84.99% |

S2 decode 饱和：nano 1672 vs vLLM 3758 out tok/s。

【没做成 / 代价】  
- hybrid 没有把可持续 QPS 拉到 vLLM。差距在执行层（FlashAttention 分桶、CUDA Graph、异步调度），不是再调一个预算就能抹平。  
- INT8 容量不增反减，要改 scale 布局才能讲「KV ×1.78」。  
- nano n-gram 和 hybrid 叠在一起时 draft 为 0，这次没有投机收益数字。  
- 没跑 0.6B S0、没三次中位数、没 S5。8B 只抽了 S3 4.11 一档，24GB 上 KV 很紧，vLLM TTFT p99 异常，不当主结论。  
- decode pack 旧结论是无收益，新口径未重测。

【学到了什么】  
1. 性能结论首先取决于测量方法：引擎内打点和客户端 SLO 讲的是两件不同的事。  
2. 调度优化先改的是长尾（ITL p99），不一定改拐点 QPS；报数字前先问自己报的是饱和吞吐还是 goodput。
