# nano-vllm 改进计划执行 Timeline

> 生成于 2026-08-18，基于 `IMPROVEMENT_PLAN.md`（含 2026-08-18 上游+业界调研补充）整理的**执行时间线**。
> 分工：改进项的"做法/验收/参考 PR"看 `IMPROVEMENT_PLAN.md`；本文件只回答"**什么时候做什么、用什么配置跑、动手前准备什么**"。
>
> 硬件前提：2×4090D（同机，共 48GB，PCIe 无 NVLink）。开工前先 `nvidia-smi` 确认是**一台机器两张卡**——
> 若实际是两台各一张卡的独立机器，跨机 TP 不可行，阶段三的 30B-A3B 需改走社区 GPTQ-Int4（~17GB）单卡路线。

## 总览

主线约 **11-16 个工作日**，选做项另计。

```
第0步  建基线 baseline_v0（半小时，单卡）
阶段一  P0 打地基（1.5-2d）      1.1 → 1.2 ∥ 1.3 → 1.4
阶段二  泛化+低垂果实（4-5d）    2.3 注册表+Llama → 2.6 宿主侧打包 → 2.1 overlap
阶段三  量化→MoE 主线（4-7d）    3.6 FP8(+INT4) → 2.5 Qwen3-MoE
阶段四  选做（与主线独立）        xgrammar / 2.4 调度消融 / 2.2 投机解码(n-gram→EAGLE-3) → 其余 3.x
```

关键依赖链（不可对调的部分）：

- **第0步 → 一切**：没有 baseline_v0，后面每项优化都无法量化归因；
- **1.1 → 所有改动**：HF exact-match 对齐是全局正确性验收标准；
- **1.3 → P1 各项**：单次运行 <5% 的差异不可信（2.6 预期 +3.5~5.8%，正好在噪声区边缘）；
- **1.4 → 3.5/2.4**：先收编上游 ~10 个已知 bug 为回归测试，再重写 block_manager/scheduler；
- **2.6 → 2.1**：先做便宜的宿主侧打包，overlap 的剩余收益才能单独归因；
- **2.3 + 3.6 → 2.5**：MoE 依赖注册表分流；61GB BF16 权重放不进 48GB，FP8 加载是硬前置。

## 逐日安排

| 工作日 | 内容 | 卡 | 动手前准备 |
|---|---|---|---|
| D0（0.5h） | **第0步**：`python baseline.py --save baseline_v0`，commit 进 `results/` | 单卡 | 无（Qwen3-0.6B 已就位） |
| D1 | **1.1 贪心采样 + HF 对齐**（0.5d）；下午开始 **1.2 引擎内指标上报** | 单卡 | 固定 4 条 prompt × 64 token 的对齐用例 |
| D2 | **1.3 baseline 统计补强**（3 次取中位数、ITL P99、mixed 套件）；完成 1.2 | 单卡 | 波动 >2% 先锁频排查 |
| D3 | **1.4 正确性雷区收编**：单卡用例（#199/#209/#211/#219/#240/#114 等）+ 修 TP 双 bug #248/#215 | 单卡 → **双卡验证 TP bug** | 对照上游 PR diff 反推复现场景 |
| D4-D5 | **2.3 模型注册表 + Llama 移植**（1.5d） | 单卡 | 下载 Llama-3.2-3B（~6GB） |
| D6 | **2.6 宿主侧 decode 输入打包消除**（1-2d，参考上游 PR #253） | 单卡 | nsys 可用，用于验证稳态零 cudaMalloc |
| D7-D8 | **2.1 overlap 调度**（双缓冲 + CUDA event；重推演抢占路径正确性，依赖 D3 的回归测试） | 单卡 | — |
| D9 | **3.6 FP8 weight-only**（block-wise scale，读 `weight_scale_inv`） | 单卡调试 | 用小模型 FP8 checkpoint 验证 loader |
| D10 | **3.6 INT4/AWQ 扩展**（可后移到阶段四；若只做 FP8 则 D10 直接进 2.5） | 单卡 | Qwen3-32B-AWQ（~18GB，官方） |
| D11 | 下载 **Qwen3-30B-A3B-Instruct-2507-FP8**（~31GB）+ **2.5** 开工：`qwen3_moe.py` 模型结构 + top-k 路由 | **双卡 TP2** | 磁盘需 ≥32GB 空闲（当前 ~49GB ✓） |
| D12-D15 | **2.5 续**：fused MoE Triton kernel 移植（工作量一半以上）→ TP2 联调 → 量化指标三件套 | **双卡 TP2** | 磁盘余量小，勿同时存放大模型 |
| D16+ | 阶段四选做：xgrammar（1-3d）/ 2.4 调度消融 / 2.2 n-gram→EAGLE-3 → 其余 3.x | 视项而定 | EAGLE-3 head（`RedHatAI/Qwen3-8B-Thinking-speculator.eagle3`） |

P0 尾声可选项：用 TP2 跑一份 `baseline_v0_tp2` 存档（几分钟），为后续 TP 相关优化（3.8）准备对照数据。

## 单卡 / 双卡使用原则

**基线一致性原则：`baseline_v0` 与所有 `opt_xxx` 必须同配置积累，中途不换。选定单卡为基准配置（变量最少，无 TP 通信噪声）。**

| 阶段/事项 | 配置 | 原因 |
|---|---|---|
| 第0步 + 1.1/1.2/1.3 | 单卡 | 测量与正确性基建，TP 只会引入噪声 |
| 1.4 大部分用例 | 单卡 | prefix cache / chunked prefill 的单卡路径 bug |
| 1.4 的 #248/#215 | **双卡 TP2** | TP 专属 bug，修复后双卡验证 |
| 2.3/2.6/2.1、2.2、2.4 | 单卡 | 与 TP 无关或先单卡调通 |
| 3.6 量化验证、2.5 MoE | **双卡 TP2** | 30B-A3B-FP8 权重 31GB，单卡放不下（每卡 ~15.5GB，剩 ~8GB KV） |
| 3.8 自定义 allreduce | **双卡 TP2** | 本身就是 TP 通信优化 |

## 权重获取清单（按需下载，不提前占盘）

| 模型 | 大小 | 用途 | 状态 |
|---|---|---|---|
| Qwen3-0.6B | 1.5GB | 第0步/P0 基线 | ✅ 已在 `huggingface/` |
| Llama-3.2-3B | ~6GB | 2.3 对齐调试 | D4 前下载 |
| Llama-3.1-8B | ~16GB | 2.3 性能报告（可选） | 视磁盘 |
| Qwen3-32B-AWQ | ~18GB | 3.6 INT4 扩展（官方） | D10 前确认 |
| Qwen3-30B-A3B-Instruct-2507-FP8 | ~31GB | 2.5 MoE 主线（官方，block-size 128 细粒度 FP8） | D11 下载 |

下载命令（本机 HF 直连超时，走镜像或 ModelScope）：

```bash
# hf-mirror
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local-dir /root/autodl-tmp/huggingface/Qwen3-30B-A3B-Instruct-2507-FP8/

# 或 ModelScope（国内直连）
modelscope download --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --local_dir /root/autodl-tmp/huggingface/Qwen3-30B-A3B-Instruct-2507-FP8/
```

注意：该 FP8 checkpoint 是 **block-wise 细粒度量化**（每 128 权重一组 scale，`weight_scale_inv` 字段），
3.6 的反量化 loader 要按块还原 BF16，比上游 PR #241 的 per-channel 实现略复杂。

更小的 MoE 权重？官方 HF 格式最小就是 31GB FP8；社区 GPTQ-Int4/AWQ ~17GB（非官方、loader 工作量更大，
且与 3.6 的 INT4 扩展重复）；GGUF 不可用（非 HF 格式）。**2.5 按计划走官方 FP8，INT4 留给 3.6 消融。**

## 执行纪律（每项固定闭环）

```bash
# 每完成一项：
python baseline.py --save opt_xxx
python baseline.py --compare results/baseline_v0.json results/opt_xxx.json
# → IMPROVEMENT_PLAN.md 文末"结果台账"追加一行（指标变化 + commit + 正确性状态）
# → git commit（correctness 套件通过才允许提交，见第 4 节回归自动化）
```

任何时刻中断，做到 D 几并不重要——台账里每项都有 before/after，面试材料始终完整。

## 里程碑

- **M0**（D0）：`baseline_v0.json` 落盘并 commit —— 一切对比的锚点；
- **M1**（D3）：P0 完成 —— 拥有"统计可信的基准体系 + HF 对齐的正确性 + 回归测试兜底"；
- **M2**（D8）：多架构（Llama）+ decode 路径优化（打包/overlap）落地，首批台账数据成型；
- **M3**（D15）：2.5 MoE TP2 跑通 —— 项目最大亮点落地（fused Triton kernel + FP8 + 显存/算力叙事）；
- **M4+**：选做项按兴趣推进，每项独立成表。
