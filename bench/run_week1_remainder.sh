#!/usr/bin/env bash
# Week-1 remainder: nano variants + align_hf + INT8 GSM8K + n-gram S6.
# Does not change GitHub default branch; does not run 0.6B S0.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PATH="/root/autodl-tmp/envs/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0

NANO_PY="${NANO_PY:-/root/autodl-tmp/envs/nano/bin/python}"
CLIENT="${CLIENT:-/root/autodl-tmp/envs/vllm/bin/vllm-client}"
MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3-4B}"
SERVED="${SERVED:-Qwen3-4B}"
OUT="${OUT:-$ROOT/results/v2/4090d}"
SHAREGPT="${SHAREGPT:-/root/autodl-tmp/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
TASKS="$ROOT/bench/lmeval_tasks"
PORT=8001
DATE="$(date +%Y%m%d)"
mkdir -p "$OUT"

kill_servers() {
  pkill -f 'bench.nano_server' || true
  pkill -f 'vllm.entrypoints' || true
  pkill -f 'EngineCore' || true
  sleep 2
}

wait_health() {
  local url="$1" pid="$2"
  local i
  for i in $(seq 1 180); do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "died $url"
      return 1
    fi
    sleep 2
  done
  echo "health timeout $url"
  return 1
}

COMMON=(
  --backend openai --endpoint /v1/completions
  --model "$SERVED" --tokenizer "$MODEL"
  --ignore-eos --seed 0
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99
  --goodput ttft:500 tpot:50 --save-result --result-dir "$OUT"
)

bench_scene() {
  local variant="$1" scene="$2" extra_name="$3"; shift 3
  echo "[bench] nano $variant $scene $extra_name"
  "$CLIENT" -m vllm.entrypoints.cli.main bench serve \
    --host 127.0.0.1 --port "$PORT" \
    --label "nano_${variant}_${scene}${extra_name}" \
    --result-filename "${DATE}_${scene}_nano_${variant}${extra_name}.json" \
    "${COMMON[@]}" "$@"
}

start_nano() {
  local variant="$1"; shift
  kill_servers
  echo "[start] nano $variant $*"
  "$NANO_PY" -m bench.nano_server --model "$MODEL" --served-model-name "$SERVED" \
    --port "$PORT" --max-model-len 4096 --max-num-seqs 256 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 \
    "$@" > "$OUT/server_nano_${variant}.log" 2>&1 &
  local pid=$!
  wait_health "http://127.0.0.1:${PORT}/health" "$pid" || {
    tail -50 "$OUT/server_nano_${variant}.log"
    return 1
  }
}

run_s1s2s4_s3rates() {
  local variant="$1"
  bench_scene "$variant" S1 "" --dataset-name random --random-input-len 1024 --random-output-len 128 --request-rate inf --num-prompts 1000
  bench_scene "$variant" S2 "" --dataset-name random --random-input-len 128 --random-output-len 1024 --request-rate inf --num-prompts 500
  bench_scene "$variant" S4 "" --dataset-name random --random-input-len 128 --random-output-len 128 --random-prefix-len 1024 --request-rate inf --num-prompts 256
  local rate tag
  for rate in 4.11 8.22 12.33 14.80; do
    tag="_r${rate/./p}"
    bench_scene "$variant" S3 "$tag" --dataset-name sharegpt --dataset-path "$SHAREGPT" --request-rate "$rate" --num-prompts 600
  done
}

if [[ "${WEEK1_SKIP_KV:-0}" == "1" ]]; then
  echo "===== skip KV blocks (WEEK1_SKIP_KV=1) ====="
else
echo "===== KV blocks auto vs int8 ====="
set +e
"$NANO_PY" - <<'PY' | tee "$OUT/kv_blocks_auto_vs_int8.json"
import json, torch
from nanovllm import LLM
model = "/root/autodl-tmp/models/Qwen3-4B"
out = {}
for dtype in ("auto", "int8"):
    print(f"measuring {dtype} ...", flush=True)
    llm = LLM(
        model,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        kv_cache_dtype=dtype,
        enforce_eager=True,
    )
    out[dtype] = {
        "num_kvcache_blocks": llm.config.num_kvcache_blocks,
        "block_size": llm.config.kvcache_block_size,
        "tokens": llm.config.num_kvcache_blocks * llm.config.kvcache_block_size,
    }
    print(json.dumps(out[dtype]), flush=True)
    llm.exit()
    del llm
    torch.cuda.empty_cache()
out["ratio_int8_over_auto"] = out["int8"]["num_kvcache_blocks"] / max(out["auto"]["num_kvcache_blocks"], 1)
print(json.dumps(out, indent=2))
PY
kv_rc=$?
set -e
if [[ "$kv_rc" -ne 0 ]]; then
  echo "KV block probe failed (rc=$kv_rc); continue. Will read blocks from server logs later."
fi
fi
kill_servers

if [[ "${WEEK1_SKIP_HF_ALIGN:-0}" == "1" ]]; then
  echo "===== skip HF align (WEEK1_SKIP_HF_ALIGN=1) ====="
else
echo "===== align_hf vs HF (4B may not be 100%; do not abort) ====="
ALIGN_LOG="$OUT/align_hf_policies.log"
: > "$ALIGN_LOG"
for pol in prefill_priority token_budget hybrid; do
  echo "----- $pol vs HF -----" | tee -a "$ALIGN_LOG"
  "$NANO_PY" experiments/align_hf.py --model "$MODEL" --max-tokens 64 \
    --schedule-policy "$pol" --prefill-token-budget 512 \
    2>&1 | tee -a "$ALIGN_LOG" || true
  kill_servers
done
echo "----- int8 vs HF (expect not 100%) -----" | tee -a "$ALIGN_LOG"
"$NANO_PY" experiments/align_hf.py --model "$MODEL" --max-tokens 64 \
  --kv-cache-dtype int8 --schedule-policy hybrid \
  2>&1 | tee -a "$ALIGN_LOG" || true
kill_servers
fi

if [[ "${WEEK1_SKIP_POLICY_ALIGN:-0}" == "1" ]]; then
  echo "===== skip policy-vs-policy align ====="
else
echo "===== schedule policies vs nano baseline (the real correctness bar) ====="
"$NANO_PY" - <<'PY' | tee "$OUT/align_policies_vs_baseline.log"
import json
from nanovllm import LLM, SamplingParams
model = "/root/autodl-tmp/models/Qwen3-4B"
prompts = [
    "The capital of France is",
    "def quicksort(arr):",
    "1 + 2 + 3 + ... + 100 =",
    "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
]
sp = SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True)

def run(policy, kv="auto"):
    llm = LLM(model, enforce_eager=True, max_model_len=4096,
              schedule_policy=policy, prefill_token_budget=512,
              enable_ngram=False, kv_cache_dtype=kv)
    ids = [llm.tokenizer.encode(p) for p in prompts]
    outs = [o["token_ids"] for o in llm.generate(ids, sp, use_tqdm=False)]
    llm.exit()
    return outs

base = run("prefill_priority")
print("baseline tokens ready", [len(x) for x in base], flush=True)
for pol in ("token_budget", "hybrid"):
    cur = run(pol)
    n = sum(a == b for a, b in zip(base, cur))
    print(f"{pol} vs prefill_priority: {n}/4 exact", flush=True)
    for i, (a, b) in enumerate(zip(base, cur)):
        if a != b:
            d = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f"  diverge prompt[{i}] @{d}")
print("done", flush=True)
PY
fi
kill_servers

echo "===== token_budget S1-S4 + S3 rates ====="
start_nano token_budget --schedule-policy token_budget --prefill-token-budget 512
run_s1s2s4_s3rates token_budget
kill_servers

echo "===== hybrid S1-S4 + S3 rates ====="
start_nano hybrid --schedule-policy hybrid --prefill-token-budget 512
run_s1s2s4_s3rates hybrid
kill_servers

echo "===== int8 KV (hybrid) S1-S4 + S3 rates ====="
start_nano int8 --schedule-policy hybrid --prefill-token-budget 512 --kv-cache-dtype int8
run_s1s2s4_s3rates int8
echo "===== INT8 GSM8K ====="
set -o pipefail
/root/autodl-tmp/envs/vllm/bin/python -m lm_eval --model local-completions \
  --model_args model=Qwen3-4B,base_url=http://127.0.0.1:${PORT}/v1/completions,tokenizer=$MODEL,num_concurrent=32,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5 --include_path "$TASKS" --log_samples \
  --output_path "$OUT/lmeval_nano_int8" \
  2>&1 | tee "$OUT/lmeval_nano_int8.log"
kill_servers

echo "===== nano ngram S6 ====="
start_nano ngram --schedule-policy hybrid --prefill-token-budget 512 --enable-ngram \
  --ngram-prompt-lookup-max 4 --ngram-prompt-lookup-min 1 --ngram-max-ngram-size 3
bench_scene ngram S6 "_c1" --dataset-name hf --dataset-path likaixin/InstructCoder --max-concurrency 1 --request-rate inf --num-prompts 100
curl -fsS "http://127.0.0.1:${PORT}/nano/metrics" | tee "$OUT/nano_ngram_c1_metrics.json" || true
curl -fsS -X POST "http://127.0.0.1:${PORT}/nano/reset_metrics" >/dev/null || true
bench_scene ngram S6 "_c4" --dataset-name hf --dataset-path likaixin/InstructCoder --max-concurrency 4 --request-rate inf --num-prompts 100
curl -fsS "http://127.0.0.1:${PORT}/nano/metrics" | tee "$OUT/nano_ngram_c4_metrics.json" || true
kill_servers

echo WEEK1_REMAINDER_EXIT:0
nvidia-smi --query-gpu=memory.used --format=csv
ls -1 "$OUT"/20261117_*nano_{token_budget,hybrid,int8,ngram}* 2>/dev/null || true
