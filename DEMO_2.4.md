# 2.4 面试演示卡（钉死版）

> 证据：`results/opt_2.4_schedule.json` · 复现：`python ablate_schedule.py`  
> 负载：mixed `16×(1024→32)` + `16×(64→256)` · 三次中位数 · Qwen3-0.6B 单卡

---

## 30 秒开场

Continuous batching 里若 **waiting 有 prefill 就绝不跑 decode**，长 prompt 持续到达时，短请求的 decode 会被插队 → **ITL 长尾**。  
我加了三种策略开关，用同一 mixed 负载画权衡：**不是单纯提吞吐，而是暴露 TTFT ↔ ITL 的 trade-off**。

---

## 消融表（背这张）

| 策略 | 吞吐 tok/s | LP TTFT p50 | LD ITL p99 | 一句话 |
|------|------------|-------------|------------|--------|
| **prefill_priority**（默认） | 4453（极差大） | 145 ms | **5.07 ms** | 吞吐高，decode 易被饿 → 长尾 |
| **decode_priority** | 4123（−7%） | 145 ms | **3.50 ms** | 吞吐略降，**LD 长尾 −31%** |
| token_budget **256** | 1976 | **736** | 4.10 | prefill 太碎 → TTFT 炸 |
| token_budget **1024** | 3751 | 182 | 5.04 | 中间点 |
| token_budget **4096** | 4397 | **97.5** | 5.11 | 接近满 prefill，行为近默认 |

- **LP** = 长 prefill 组（1024→32）；**LD** = 长 decode 组（64→256）  
- 主对比只盯两行：**prefill vs decode_priority 的 LD ITL p99**  
- budget 扫线讲「旋钮」：小预算伤 TTFT/吞吐，大预算退回 prefill 优先

---

## 根因（指代码）

```103:111:nanovllm/engine/scheduler.py
        if policy == "prefill_priority":
            pref = self._try_schedule_prefill(None)
            if pref is not None:
                self._sample_kv()
                return pref, True
            dec = self._try_schedule_decode()
```

`decode_priority`：running 非空先 decode。  
`token_budget`：每步 prefill ≤ `prefill_token_budget`，再与 decode **轮转**（单步仍纯 P 或纯 D，近似 Sarathi 限预算，非同一步混 batch）。

---

## 现场怎么跑（可选）

```bash
# 重跑消融（约数分钟）
python ablate_schedule.py --repeats 3

# 或只打印已落盘表
python - <<'PY'
import json
d=json.load(open("results/opt_2.4_schedule.json"))
for k,v in d["points"].items():
    m=v["median"]; ld=m["long_decode"]; lp=m["long_prefill"]
    print(f"{k:22} thr={m['throughput_tok_s']:7.1f}  "
          f"LP_TTFT={lp['ttft_ms']['p50']:6.1f}  LD_ITL99={ld['itl_ms']['p99']:5.2f}")
PY
```

---

## 面试官追问（标准答）

| 问 | 答 |
|----|----|
| 为何 LP TTFT 在 P/D 优先几乎一样？ | 本负载两端同时入队；长 prompt 的首 token 主要被「自己的 prefill 排队」决定，差异更明显在 **已在 running 的 LD 的 ITL**。 |
| 为何不报 +X% 吞吐？ | prefill 优先吞吐最高，但 **极差 675** 远大于 decode 的 18；服务场景看 **长尾与稳定性**，不是峰值 tok/s。 |
| 和 chunked prefill / Sarathi？ | 已有 chunked；2.4 补的是 **策略开关 + 量化曲线**。token_budget ≈ 限预算 prefill + 不饿死 decode（本实现逐步轮转，非同 kernel 混 batch）。 |
| 和 2.6 关系？ | 2.4=谁先跑；2.6=decode 步 CPU 打包。2.6 本机收益在噪声边，**演示主戏是这张表**。 |

---

## 不要讲的

- 不要把 token_budget 中档的 `all_itl_p99≈23` 当主 KPI（含 LP 组被拉长的间隔，叙事易糊）  
- 不要对 `baseline_v0` 报「调度加速百分之几」——对比对象是 **同 mixed 下策略互比**
