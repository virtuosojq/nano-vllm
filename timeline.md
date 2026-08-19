# nano-vllm 改进计划执行 Timeline

> 生成于 2026-08-18，2026-08-19 按面试三层故事收束。
> 分工：改进项的「做法/验收/参考 PR」看 `IMPROVEMENT_PLAN.md`；本文件只回答
> **什么时候做什么、用什么配置跑、动手前准备什么、做到哪可以停**。
>
> **进度（2026-08-19）**：**M0 / M1 / M2 已落地**。下一阶段日历 = **2.7 → 3.1**（单卡即可）。
> 硬件上限 2×4090（同机 48GB，PCIe 无 NVLink）。开工前 `nvidia-smi` 确认卡数——跨机两张卡不能 TP。
> 层 C 若碰 30B：官方 FP8 ~31GB 需 **同机 TP2**；不要把 fused MoE 当里程碑。

## 总览

**主线约 7–9 个工作日**（层 A + 层 B）**已走完**。层 C 另计，允许不做完。

```
第0步   baseline_v0          ✅ 已完成（2026-08-19，单卡 Qwen3-0.6B）
层 A    打地基               ✅ 已完成 1.1–1.4  → M1
层 B    引擎主线             ✅ 已完成 2.4 / 2.6 / 2.2  → M2
层 B+   M2 之后加厚          ⏳ 2.7 hybrid P+D → 3.1 INT8 KV（都做透）
层 C    规模加分（可失败）     未开始：可选 8B 复测 → Llama-3.2-3B → FP8 loader → MoE
停车场  明确不进日历           INT4/32B-AWQ、fused MoE 当里程碑、EAGLE-3、xgrammar、3.8、SM89 FP8 attn
```

**最小可演示完成态 = 层 A + 2.4 消融表（已达成）。** 2.7/3.1 加厚「生产调度 + 带宽墙」，不要为 30B 把这两项做浅。

关键依赖链（不可对调）：

- **第0步 → 一切**：没有 `baseline_v0`，后面每项优化都无法量化归因（✅）；
- **1.1 → 所有改动**：HF exact-match 是全局正确性验收标准（✅）；
- **1.3 → 层 B**：单次运行 <5% 的差异不可信（✅；2.6 验证了噪声门闩）；
- **1.3 mixed + 1.4 → 2.4**：没有混合负载和 preempt/prefix 用例，不要改调度（✅）；
- **2.6 → 2.1**：**不修 2.1**（2.6 吞吐已在噪声边）；
- **2.4 → 2.7 → 3.1**：轮转消融（✅）→ 同一步混 batch（⏳）→ INT8 KV（⏳）。禁止对调；
- **2.3 + 3.6 → 2.5**：层 C 内部依赖；禁止用 MoE 数字对 `baseline_v0` 报百分比。

---

## 已完成清单（2026-08-19，单卡 Qwen3-0.6B）

| 项 | 结果文件 / 入口 | 面试一句话 |
|---|---|---|
| D0 基线 | `results/baseline_v0.json` | 同配置对比锚点；throughput ~6153 tok/s |
| 1.1 贪心 | `results/opt_1.1_greedy.json`、`align_hf.py` | HF flash-attn greedy 4×64 **100%** |
| 1.2+1.3 | `results/opt_1.2_1.3_d2.json` | 三次中位数+极差；ITL P99；`mixed`；prefix hit 与 1.59× 互证 |
| 1.4 边界 | `results/opt_1.4_edges.json`、`tests/test_1_4_edges.py` | **#211 实修**；#199/#240/#114 已验证 N/A |
| 2.4 调度 | `results/opt_2.4_schedule.json`、`DEMO_2.4.md` | LD ITL p99 **5.07→3.50**；budget 扫 TTFT↔吞吐 |
| 2.6 打包 | `results/opt_2.6_decode_pack.json` | 常驻 pinned；**吞吐在噪声边**，不报加速比 |
| 2.2 n-gram | `results/opt_2.2_ngram.json` | 重复文本 accept **~0.89**；HF 对齐 **≤40**；64 会分叉 |

未做（明确跳过）：**2.1 overlap**。未做（下一阶段）：**2.7、3.1**。层 C 全未开始。

---

## 逐日安排

### 已完成（D0–D8）

| 工作日 | 层 | 内容 | 卡 | 落盘 |
|---|---|---|---|---|
| **D0 ✅** | 0 | `baseline.py --save baseline_v0` | 单卡 | `results/baseline_v0.json` |
| **D1 ✅** | A | **1.1** 贪心 + HF exact-match（不含 top_p/top_k） | 单卡 | `opt_1.1_greedy.json`；HF 4×64 = 100% |
| **D2 ✅** | A | **1.3** 三次中位数、ITL P99、`mixed`；**1.2** cache hit / KV util / per-req TTFT | 单卡 | `opt_1.2_1.3_d2.json`；bs1 极差 ≪2% |
| **D3 ✅** | A | **1.4** #199 / #240 / #211 / #114 | 单卡 | #211 修；其余 N/A；`opt_1.4_edges.json` |
| **D4–D5 ✅** | B | **2.4** `prefill_priority` / `decode_priority` / token-budget | 单卡 | `opt_2.4_schedule.json`；`DEMO_2.4.md` |
| **D6–D7 ✅** | B | **2.6** 宿主侧 decode 打包 | 单卡 | 无 nsys；thr 噪声边；`opt_2.6_decode_pack.json` |
| **D8 ✅** | B | **2.2** n-gram / prompt-lookup | 单卡 | accept ~0.89；HF≤40；`opt_2.2_ngram.json` |

### 待做（D9+）

| 工作日 | 层 | 内容 | 卡 | 动手前准备 |
|---|---|---|---|---|
| D9–D10 | B+ | **2.7** hybrid：同一步 decode + prefill chunk；`ablate_schedule.py` 加 hybrid 列 | 单卡 | 2.4 回归仍过；hybrid 步关 ngram |
| D11–D13 | B+ | **3.1** INT8 KV：store 量化 + FA 前反量化；blocks 约 ×2；align 允许记分叉 | 单卡 | **必须 2.7 已合**。禁止 FlashInfer FP8 attn |
| D14+ | C | 可选 Qwen3-8B 复测 2.7/3.1（`baseline_qwen3_8b`）；或 2.3 Llama-3.2-3B | 单卡；30B 才 TP2 | 见权重清单 |

P0 尾声 **不必** 为了 3.8 去跑 `baseline_v0_tp2`。真上 TP（层 C 的 30B 或修 #248/#215）当天再采一份同配置基线。

**2.7 与 3.1**：都做透，先 2.7 后 3.1。不要并行浅做。2.1 overlap 不再排期。

---

## 单卡 / 双卡与基线文件

**层 A/B 一律单卡 Qwen3-0.6B，对比 `baseline_v0`。** 换模型或换 TP 必须新文件，禁止混进同一张加速比表。

| 阶段/事项 | 配置 | 对比哪份基线 | 原因 |
|---|---|---|---|
| 第0步 + 1.1/1.2/1.3/1.4 | 单卡 0.6B | `baseline_v0` | 测量与正确性基建 |
| 2.4 / 2.6 / 2.1 / 2.2 / **2.7 / 3.1** | 单卡 0.6B | `baseline_v0` | 与 TP 无关；3.1 主 KPI 含 KV blocks 数 |
| 3.1 若换 Qwen3-8B | 单卡 8B | **`baseline_qwen3_8b`** | 不同模型，不能对 0.6B 报 % |
| 2.3 Llama-3.2-3B | 单卡 3B | **`baseline_llama3b`**（新建） | 不同模型，不能对 0.6B 报 % |
| 3.6 小模型 FP8 | 单卡，同结构 BF16 vs FP8 | 该项自己的配对基线 | loader 正确性，不是吞吐主线 |
| 2.5 若上 30B-A3B-FP8 | **双卡 TP2** | **`baseline_moe_tp2`**（新建） | 31GB 单卡放不下；禁止对 `baseline_v0` 比加速比 |
| #248/#215 | **双卡 TP2** | 当天的 TP 基线 | 用 TP 之前再修，不占层 A |

---

## 权重获取清单（按需下载，不提前占盘）

| 模型 | 大小 | 用途 | 何时下载 |
|---|---|---|---|
| Qwen3-0.6B | 1.5GB | 第0步 / 层 A / 层 B / 2.7 / 3.1 | ✅ 已在 `huggingface/` |
| Qwen3-8B | ~16GB BF16 | 可选：复测 2.7/3.1 | **3.1 在 0.6B 上 tok/s 不明显时再下**；另存 `baseline_qwen3_8b` |
| Llama-3.2-3B | ~6GB | 层 C 的 2.3 对齐 | **M2 已完成，仍等开层 C 再下** |
| 小模型 FP8 checkpoint | 视选定 | 层 C 的 3.6 loader | 2.3 之后、30B 之前 |
| Qwen3-30B-A3B-Instruct-2507-FP8 | ~31GB | 层 C 的 2.5，可选 | 朴素 MoE 路径已在小规模跑通、磁盘 ≥32GB |
| Llama-3.1-8B | ~16GB | ~~2.3 性能报告~~ | **不下** |
| Qwen3-32B-AWQ | ~18GB | ~~D10 INT4~~ | **主线不下**；与 30B-FP8 禁止同时占盘 |

30B-FP8 下载命令（仅层 C 需要时；本机 HF 直连超时，走镜像或 ModelScope）：

```bash
# hf-mirror
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local-dir /root/autodl-tmp/huggingface/Qwen3-30B-A3B-Instruct-2507-FP8/

# 或 ModelScope（国内直连）
modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local_dir /root/autodl-tmp/huggingface/Qwen3-30B-A3B-Instruct-2507-FP8/
```

该 FP8 checkpoint 是 **block-wise 细粒度量化**（每 128 权重一组 scale，`weight_scale_inv`），
比上游 PR #241 的 per-channel 复杂。所以 3.6 必须先在小模型上做对，再考虑 30B。

更小的官方 HF MoE？没有；社区 GPTQ-Int4 ~17GB 非官方且与已移出主线的 INT4 重复。层 C 若做 30B，走官方 FP8；做不完就停在朴素路径 + 显存账。

---

## 执行纪律（每项固定闭环）

```bash
# 同配置（层 A/B，Qwen3-0.6B 单卡）:
python baseline.py --save opt_xxx
python baseline.py --compare results/baseline_v0.json results/opt_xxx.json

# 换模型或 TP（层 C）:
python baseline.py --save baseline_llama3b    # 或 baseline_moe_tp2
python baseline.py --save opt_xxx
python baseline.py --compare results/baseline_llama3b.json results/opt_xxx.json
# 禁止: compare baseline_v0.json 与 MoE/Llama 结果

# → IMPROVEMENT_PLAN.md 文末「结果台账」追加一行（指标变化 + commit + 正确性）
# → git commit（correctness 套件通过才允许提交）
```

**2.6 / 3.1 数字门闩**：三次中位数的收益必须大于极差，才在面试里报 tok/s。3.1 即使 tok/s 落在噪声里，仍可报 **num_kvcache_blocks 约 ×2**。

任何时刻中断都可以停。停在 M1 仍有「可信基准」；停在 M2 是完整面试项目；**2.7 完成 = 调度故事闭环**；3.1 完成 = 再加带宽墙。M3 是加分。

---

## 里程碑

- **M0**（✅ 已完成）：`baseline_v0.json` 落盘 —— 同配置对比的锚点。
- **M1**（✅ D3，层 A）：HF 贪心对齐 + 三次中位数/`mixed` + 3～4 个边界回归。叙事：「我能量化、敢改」。
- **M2**（✅ D8，层 B，**面试最小完成态**）：2.4 TTFT–ITL 曲线 + 2.6（机制/噪声）+ 2.2（accept + ≤40 对齐）。叙事：「我懂服务侧引擎」。
- **M2.5**（⏳ D9–D10，**2.7**）：轮转 vs **同一步 hybrid** 对照表。叙事：「这才是 vLLM v1 / Sarathi 的 stall-free」。
- **M2.6**（⏳ D11–D13，**3.1**）：INT8 KV 显存 ×2 + 对齐率。叙事：「decode 带宽墙，4090 上不跟风 SM90 FP8 attn」。
- **M3**（D14+，层 C，可缺席）：8B 复测和/或 Llama-3.2-3B / FP8 / MoE load。fused kernel 未完成不影响 M2。
