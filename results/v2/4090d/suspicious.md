# 评测可疑点（第 1 次开机，边跑边记）

## 已观察到

1. **vLLM 0.31.0 的 torch 是 2.13.0+cu130，机器 CUDA 工具链是 12.4**。客户端不占卡时没问题；起服务时可能因 CUDA runtime 不匹配失败。驱动 580.105.08，或许能跑 CUDA 13 runtime，需实测。
2. **nano 与 vLLM 的 transformers 主版本不同**（nano 4.57.6 / vLLM 5.17.0）。分词口径可能有偏差；两侧都用同一本地 tokenizer 路径，但库实现不同。
3. **S1/S2/S4 的 `--request-rate inf` 会在 t=0 打满**，TTFT/goodput 反映排队而不是稳态服务。goodput=0 不能解读成「达不到 SLO」。S3 在线口径才有意义。
4. **nano S3 只跑过 4 req/s 一档**，尚未按 R_max 的 0.25/0.5/0.75/0.9 扫描。
5. **S1/S3 计划要 3 次取中位数**，当前都是单次。
6. **run_matrix.sh 原先把结果文件都标成 `nano_`**，vLLM `VARIANT=baseline` 会覆盖 nano JSON。已改为 `${ENGINE}_${VARIANT}`。
7. **压测客户端 wrapper 设了 `CUDA_VISIBLE_DEVICES=`**。若同一变量去起 vLLM 服务，服务看不见 GPU。已拆成 `VLLM_PY`（服务）和 `CLIENT_PY`（客户端）。
8. **nano 服务结束后 SIGTERM 可能退不干净**，上次 S1–S4 跑完后进程仍占 22GB 显存。后续每次场景结束会强制 `pkill`。
9. **S0 / Qwen3-0.6B 按你的要求跳过**，无法用 6153 tok/s 确认和旧环境完全一致。
10. **env.json 只记录了 nano 环境**。vLLM 侧应单独写 `env_vllm.json`。
11. **vLLM 0.31 默认 `max_model_len=40960`**（模型 config 上限），对齐配置才会收到 4096。默认组与对齐组的 KV 容量因此不同，不能直接和 nano 比。
12. **vLLM 起服务失败：EngineCore 子进程 PATH 里没有 `ninja`**（flashinfer JIT）。venv 里有 `/envs/vllm/bin/ninja`，但没进 PATH。修复：启动前 `PATH=/root/autodl-tmp/envs/vllm/bin:$PATH`。
13. **`python -m vllm.entrypoints.openai.api_server` 在 0.31 已标 deprecated**，官方改成 `vllm server`。功能仍可用。
14. **shell 变量 `VLLM_PY` 被 vLLM 当成未知环境变量警告**。无功能影响。
15. **CUDA graph memory profiler 把 `--gpu-memory-utilization 0.9` 实际折成约 0.902/0.92**。对齐组显式 0.9 时也要注意有效 KV 变小。
16. **flashinfer JIT 失败：`nvcc fatal: Unknown option '--compress-mode=size'`**。vLLM 0.31 / flashinfer 0.7 按 CUDA 13 的 nvcc 选项编译，系统 `/usr/local/cuda` 是 12.4。处理：`VLLM_USE_FLASHINFER_SAMPLER=0`，采样走 vLLM 内置 kernel。Attention 日志已是 FLASH_ATTN v2，bench 又是 temperature=0，对头条数字影响应有限，但「vLLM 默认」不再是 100% 开箱路径。
17. **vLLM 默认 S3 inf 的 R_max = 16.44 req/s**（300 prompts）。四档速率取 4.11 / 8.22 / 12.33 / 14.80。该 R_max 下 TTFT p50 已 2.57s、goodput 0.16，说明 inf 饱和吞吐远高于 SLO 可持续吞吐。
18. **S3 inf 期间出现 CUDACachingAllocator 警告**（vLLM default）。后续对齐组 `max_model_len=4096` 后观察是否消失。
19. **run_matrix 在 `[matrix] done` 后经常卡住**：服务 SIGTERM 不退出，tee 等管道。每次跑完需 `pkill` EngineCore / api_server。
20. **`eval $(vllm_cmd)` 会拆掉 `--speculative-config` JSON**，S6 启动报 `Value method:ngram cannot be converted`。已改成数组直接 exec，不再 eval。
21. **S6 客户端报 `OfflineModeIsEnabled`**，即使 shell 里设了 HF_ENDPOINT。必须显式 `HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0`。InstructCoder 已从 hf-mirror 下到本地 cache。
22. **vLLM S6 ngram conc=4 的 TTFT p99=1958ms**，conc=1 只有 65ms。高并发下投机解码的排队/批形状明显变差，和计划里「并发升高后收益消失」一致。
23. **S3 四档 NUM_PROMPTS 按 rate×150 会到 2220**。时间受限先每档 600（仍 ≥300）。
24. **nano S3 在 8.22 req/s 仍接近满 SLO（goodput 7.26），12.33 起崩溃**（TTFT p99 11s，goodput 1.81）。vLLM 对齐在 14.80 仍几乎满 SLO（goodput 12.32，TTFT p99 175ms）。在线口径下 nano 原版大约只能吃到 vLLM 一半的可持续速率。
25. **vLLM 环境里 `lm-eval` 没装 `[api]` extra**。`local-completions` 初始化直接报 `ModuleNotFoundError: tenacity`。`pip install lm-eval[api]` 后 tenacity 9.2.1 / aiohttp 3.14.5 可用。第一次 GSM8K 因此空跑失败，需重跑。
26. **hf-mirror 对 `openai/gsm8k` 的 Hub API 429**（5 分钟 10000 次额度打满，Retry-After 188s）。`datasets.load_dataset("openai/gsm8k")` 卡在 `repo_info`。parquet 的 resolve URL 仍可下，已落到 `/root/autodl-tmp/datasets/gsm8k/{train,test}.parquet`（7473/1319）。lm-eval 用本地 yaml（`dataset_path: parquet`）绕过 Hub。
27. **lm-eval `local-completions` 默认 `max_length=2048`、`max_gen_toks=256`**。计划命令没改这两项。`tokenized_requests=False` 时不检查 context+gen 是否超长。5-shot GSM8K 通常够用；若 Qwen3 在 completions 里自发长推理，256 可能截断。先按计划默认跑，看输出是否大量 `length`。
28. **`lm_eval | tee` 不加 `pipefail` 时，lm-eval 失败仍 `SMOKE_EXIT:0`**。后续脚本加 `set -o pipefail`。
29. **lm-eval 会打 `GET /tokenizer_info`，nano_server 返回 404**。随后回退到本地 huggingface tokenizer，功能正常。vLLM 是否实现该端点待看。
30. **nano GSM8K smoke（limit=5）strict/flexible exact_match = 0.8**。stop 生效、5-shot 能出 `####` 格式。全量 1319 才是基线。
31. **第一次失败的 lm-eval 脚本启动的 vLLM（pid 23672）被 reparent 到 pid 1，pkill 漏掉**。占着 :8000 但不占显存。第二次起 vLLM 时差点抢端口。已手动杀掉；后续每次开机前要 `pkill -f vllm.entrypoints` + `pkill -f EngineCore`。
32. **nano 全量 GSM8K：strict-match 0.8582 > flexible-extract 0.8461**。通常 flexible 更高。对照后：nano 有 24 条 strict 对、flexible 错（`####` 抽对，末位数字 regex 抽到中间计算）。vLLM 同样有 15 条。不是服务 bug，是 GSM8K flexible 过滤器的已知噪音。
33. **贪心 completions 两侧文本并不一致**：1319 条里 446 条（33.8%）生成原文不同，但 398 条 strict 抽出的数字仍相同。真正 pred 不同只有 48 条（nano strict 独赢 23、vLLM 独赢 12）。kernel / batching 浮点差会导致推理路径分叉，精度差 0.8 个点，可接受。
34. **默认 `max_gen_toks=256` 截断**：两侧各 38 条没有 `####`，重叠 32 条，结尾都是半截算式（例如 `<<180-135=45>>4`）。这是计划命令没设 `max_gen_toks` 的代价，不是 stop 失效（没有 `Question:` 泄漏，也没有 `<think>`）。若要贴近官方 Qwen3 数字，下次应把 `max_gen_toks` 提到 512/1024。
35. **flexible-extract 两侧同分 0.8461**（各独赢 23 条对消）。面试讲精度时以 **strict-match** 为主：nano **85.82%** / vLLM **84.99%**。
