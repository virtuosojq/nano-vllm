# 社招面试冲刺计划：推理引擎（vLLM / nano-vllm / vllm-ascend）

> 制定于 2026-11-07。目标是社招面试，不是系统性学完所有内容。
> 这份计划取代 `learn/` 下 24 周课表在「面试」这个目标上的位置；`learn/` 保留作参考，不再按它排进度。

## 目标与定位

**主投方向：NPU 推理（vllm-ascend、昇腾推理优化）**，GPU 推理（vLLM / SGLang）作为必修基础。

理由：你已有 CANN 算子（`ops-transformer`）、HCCL、MC2、torchair、op-plugin 的实际经验。NPU 推理岗的面试官一样会问 vLLM 原理，但能把「上层引擎调度」和「底层算子、通信」串起来讲的人很少，这是你的差异化。

简历上讲两个项目：

| 项目 | 内容 | 平台 | 状态 |
|---|---|---|---|
| **A** | 从零优化 nano-vllm：调度、KV cache、投机解码，用 vLLM 标准口径和 vLLM 同卡对比 | AutoDL 4090 D | 功能已完成（`back` 分支），**评测要重做** |
| **B** | 在 A3（910C）上用 vllm-ascend 部署 MoE 模型，端到端性能剖析与优化，追到 MC2 算子 | 远程 A3 | 待做 |

## 文件索引

| 文件 | 内容 | 什么时候看 |
|---|---|---|
| [00-环境与工作流.md](00-环境与工作流.md) | 本地 / AutoDL / 远程 A3 三个环境怎么分工、怎么同步、怎么省钱 | 第 1 周开始前 |
| [01-项目A-评测重做.md](01-项目A-评测重做.md) | 现有评测的问题、OpenAI 兼容服务设计、场景矩阵、精度验证 | 第 1–1.5 周 |
| [02-vLLM-V1源码.md](02-vLLM-V1源码.md) | 按面试题读 vLLM V1 源码的地图和问题清单 | 第 2–3 周 |
| [03-项目B-vllm-ascend-A3.md](03-项目B-vllm-ascend-A3.md) | A3 上的部署、基线、剖析、优化步骤，源码和算子追踪路线 | 第 4–6 周 |
| [04-面试知识点.md](04-面试知识点.md) | 八股知识点详解，每条附要点和常见追问 | 第 2 周起每天 30 分钟 |
| [05-手写代码与系统设计.md](05-手写代码与系统设计.md) | 手写代码题单、系统设计模板和例题、项目讲述模板 | 第 2 周起穿插，第 6 周起重点 |

## 时间表（每个工作日 1–1.5 小时，共 8–9 周）

| 周 | 主线 | 平台 | 每天穿插 |
|---|---|---|---|
| 1–1.5 | 项目 A 评测重做 | 本地写代码，AutoDL 开 2 次机 | — |
| 2–3 | vLLM V1 源码 | 本地为主，AutoDL 开 1–2 次机 | 八股 30 分钟 |
| 4–6 | 项目 B | 远程 A3 | 八股 / 手写代码 30 分钟 |
| 6–8 | 面试冲刺、开始投递 | 本地 | 模拟面试 |

每天投入更多时间的话，可以压到 6–7 周。

## 每周检查点

勾完一周的所有项再进入下一周。卡住超过两天，看下面的「风险与降级」。

### 第 1 周（含 1.5 周的缓冲）：项目 A 评测重做

- [ ] GitHub 上 `virtuosojq/nano-vllm` 的默认分支改成 `back`（fork 网页改可见性会失败；`cu` 的 README 已链到 `back`/`cu`）
- [x] 写好 nano-vllm 的 OpenAI 兼容服务（`/v1/completions`，流式和非流式，`stop`、`ignore_eos`、`usage`）
- [x] 本地用 `curl` 和一个假模型/小模型验证服务协议正确（不需要 GPU 的部分先在本地做完）
- [x] 写好场景矩阵脚本，一条命令跑完 nano-vllm 和 vLLM 两侧
- [ ] AutoDL 第 1 次开机：用旧 `baseline.py` 复现旧的离线吞吐（约 6153 tok/s，Qwen3-0.6B），确认环境一致
- [ ] AutoDL 第 1 次开机：跑 vLLM 和 nano-vllm 基线（未优化版本）的完整场景矩阵
- [ ] AutoDL 第 2 次开机：跑 nano-vllm 各项优化的场景矩阵，INT8 KV 跑 lm-eval GSM8K
- [ ] 写出项目 A 的一页说明（见 `05-手写代码与系统设计.md` 的项目讲述模板）

### 第 2 周：vLLM V1 引擎与调度

- [ ] clone vLLM `v0.31.0`（和本地 vllm-ascend 的 `Dockerfile.a3` 对齐）
- [ ] 读完 `02-vLLM-V1源码.md` 第 1、2、3 节，回答每节的问题清单
- [ ] 完成 nano-vllm ↔ vLLM 模块对照表
- [ ] 八股：`04-面试知识点.md` 第 1–3 章

### 第 3 周：vLLM V1 执行层与扩展点

- [ ] 读完 `02-vLLM-V1源码.md` 第 4–7 节
- [ ] AutoDL 开 1 次机：chunked prefill 预算扫描、prefix caching 开关、eager vs CUDA Graph、nsys 抓 decode
- [ ] 产出「平台插件需要实现哪些接口」清单（第三阶段的地图）
- [ ] 八股：第 4–6 章；手写代码：attention、online softmax、采样

### 第 4 周：项目 B 跑通与基线

- [ ] 远程 A3 环境确认：`npu-smi info`、驱动和 CANN 版本、能否联网、能否拉镜像
- [ ] 用 vllm-ascend 镜像部署 Qwen3-30B-A3B（W8A8 单 die 和 BF16 双 die 两个配置）
- [ ] 用 `vllm bench serve` 跑出基线（场景矩阵复用项目 A 的）
- [ ] lm-eval GSM8K 跑出精度基线
- [ ] 读 `vllm_ascend/platform.py`、`patch/`、`worker/`
- [ ] 八股：第 7 章（昇腾专项）

### 第 5 周：项目 B 剖析

- [ ] eager vs ACL Graph、TP vs EP 的对比数据
- [ ] 用 Ascend PyTorch Profiler 抓 prefill 和 decode 时间线，算出 attention / MoE 计算 / dispatch-combine 通信 / 其他 的耗时占比
- [ ] 完成 MoE 全链路追踪：vllm-ascend → torch_npu → ops-transformer MC2 算子
- [ ] **降级判断点**：本周末还没有稳定基线，项目 B 目标改成「完整性能剖析报告」

### 第 6 周：项目 B 优化与收尾

- [ ] 至少一项有数字、能解释原理的优化（候选见 `03-项目B-vllm-ascend-A3.md`）
- [ ] 写出项目 B 的一页说明和全链路图
- [ ] （可选）向 vllm-ascend 提一个 PR
- [ ] 开始投递几家练手公司

### 第 7–8 周：面试冲刺

- [ ] 两个项目各准备 10 个以上的追问并能脱稿回答
- [ ] 手写代码题单全部能脱稿写出
- [ ] 系统设计例题完整讲一遍（计时 30 分钟）
- [ ] 每次面试后把被问住的问题记到 `面试复盘.md`，24 小时内补上
- [ ] 正式投递目标公司

## 风险与降级

| 风险 | 信号 | 降级方案 |
|---|---|---|
| 项目 A 服务包装卡住 | 第 1 周过半还没跑通 `vllm bench serve` | 先只做离线口径：用 vLLM 的数据集采样逻辑 + `vllm bench throughput` 的指标定义，nano-vllm 侧自己实现同口径统计 |
| A3 环境搭建慢 | 第 4 周结束还没跑出第一个请求 | 先用官方镜像 + 官方教程的原样命令，不做任何自定义；仍不行就把项目 B 改成「源码 + 算子链路分析」，实验数据缩到最少 |
| 项目 B 没有优化收益 | 第 5 周末没找到可优化点 | 目标改为性能剖析报告：瓶颈定位、各配置对比、原因分析。面试里讲清楚「为什么没有收益」同样有价值 |
| 时间不够 | 任一周延期超过 3 天 | 先砍 02 里的第 6–7 节（PD 分离、平台插件细节），这两块靠八股补 |

## 和旧计划的关系

- `learn/` 下 W04–W12（kernel 相关）：对 NPU 推理岗价值有限，你在 CANN 侧已有实际经验，跳过。
- W13–W17（nano-vllm、continuous batching、paged attention、scheduler、prefix）：已经并入项目 A 和 02 的源码阅读。
- W18–W23（并行、PD 分离、SGLang radix、投机解码、量化）：并入 `04-面试知识点.md`。
- W24 profiling capstone：由项目 B 替代。
