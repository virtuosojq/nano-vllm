#!/usr/bin/env bash
# Scenario matrix from interview-plan/01-项目A-评测重做.md
# Local:  DRY_RUN=1 bash bench/run_matrix.sh
# AutoDL: bash bench/run_matrix.sh --engine both --variant baseline
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/root/autodl-tmp/models/Qwen3-4B}"
TOKENIZER="${TOKENIZER:-$MODEL}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B}"
RESULT_DIR="${RESULT_DIR:-$ROOT/results/v2/4090d}"
SHAREGPT="${SHAREGPT:-/root/autodl-tmp/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
NANO_PORT="${NANO_PORT:-8001}"
VLLM_PORT="${VLLM_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
DRY_RUN="${DRY_RUN:-0}"
ENGINE="${ENGINE:-nano}"          # nano | vllm | both
VARIANT="${VARIANT:-baseline}"    # baseline | token_budget | hybrid | int8 | ngram | vllm_default | vllm_aligned
SCENES="${SCENES:-S1,S2,S4}"      # comma-separated; S0/S3/S6 need extra data

NANO_PY="${NANO_PY:-python}"
VLLM_PY="${VLLM_PY:-python}"

mkdir -p "$RESULT_DIR"

COMMON=(
  --backend openai
  --endpoint /v1/completions
  --model "$SERVED_NAME"
  --tokenizer "$TOKENIZER"
  --ignore-eos
  --seed 0
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,90,99
  --goodput ttft:500 tpot:50
  --save-result
  --result-dir "$RESULT_DIR"
)

log() { printf '[matrix] %s\n' "$*"; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN: $*"
    return 0
  fi
  "$@"
}

wait_health() {
  local url="$1"
  local i
  for i in $(seq 1 120); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  log "health check failed: $url"
  return 1
}

nano_cmd() {
  local extra=()
  case "$VARIANT" in
    baseline) extra=(--schedule-policy prefill_priority) ;;
    token_budget) extra=(--schedule-policy token_budget --prefill-token-budget 512) ;;
    hybrid) extra=(--schedule-policy hybrid --prefill-token-budget 512) ;;
    int8) extra=(--schedule-policy hybrid --kv-cache-dtype int8) ;;
    ngram) extra=(--schedule-policy hybrid --enable-ngram) ;;
    *) extra=(--schedule-policy prefill_priority) ;;
  esac
  echo "$NANO_PY" -m bench.nano_server --model "$MODEL" --served-model-name "$SERVED_NAME" \
    --port "$NANO_PORT" --max-model-len 4096 --max-num-seqs 256 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.9 "${extra[@]}"
}

vllm_cmd() {
  local extra=()
  if [[ "$VARIANT" == "vllm_default" ]]; then
    extra=()
  elif [[ "$VARIANT" == "ngram" ]]; then
    extra=(--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":3,"prompt_lookup_min":1}')
  else
    extra=(--max-model-len 4096 --max-num-seqs 256 --max-num-batched-tokens 8192
           --gpu-memory-utilization 0.9 --dtype bfloat16)
  fi
  echo "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$MODEL" \
    --served-model-name "$SERVED_NAME" --port "$VLLM_PORT" "${extra[@]}"
}

bench_one() {
  local port="$1" scene="$2" extra=()
  case "$scene" in
    S1) extra=(--dataset-name random --random-input-len 1024 --random-output-len 128 --request-rate inf --num-prompts 1000) ;;
    S2) extra=(--dataset-name random --random-input-len 128 --random-output-len 1024 --request-rate inf --num-prompts 500) ;;
    S3) extra=(--dataset-name sharegpt --dataset-path "$SHAREGPT" --request-rate "${RATE:-4}" --num-prompts "${NUM_PROMPTS:-300}") ;;
    S4) extra=(--dataset-name random --random-input-len 128 --random-output-len 128 --random-prefix-len 1024 --request-rate inf --num-prompts 256) ;;
    S6) extra=(--dataset-name hf --dataset-path likaixin/InstructCoder --max-concurrency "${MAX_CONCURRENCY:-1}" --request-rate inf --num-prompts 100) ;;
    *) log "unknown scene $scene"; return 1 ;;
  esac
  run "$VLLM_PY" -m vllm.entrypoints.cli.main bench serve \
    --host "$HOST" --port "$port" "${COMMON[@]}" "${extra[@]}"
}

start_nano() {
  log "start nano-vllm ($VARIANT) on :$NANO_PORT"
  if [[ "$DRY_RUN" == "1" ]]; then
    nano_cmd
    return 0
  fi
  # shellcheck disable=SC2046
  eval $(nano_cmd) &
  NANO_PID=$!
  wait_health "http://$HOST:$NANO_PORT/health"
}

start_vllm() {
  log "start vLLM ($VARIANT) on :$VLLM_PORT"
  if [[ "$DRY_RUN" == "1" ]]; then
    vllm_cmd
    return 0
  fi
  eval $(vllm_cmd) &
  VLLM_PID=$!
  wait_health "http://$HOST:$VLLM_PORT/health"
}

stop_servers() {
  if [[ "${NANO_PID:-}" ]]; then kill "$NANO_PID" 2>/dev/null || true; fi
  if [[ "${VLLM_PID:-}" ]]; then kill "$VLLM_PID" 2>/dev/null || true; fi
}
trap stop_servers EXIT

log "collect env"
run "$NANO_PY" "$ROOT/tools/collect_env.py" --model "$MODEL" -o "$RESULT_DIR/env.json"

IFS=',' read -r -a SCENE_ARR <<< "$SCENES"

if [[ "$ENGINE" == "nano" || "$ENGINE" == "both" ]]; then
  start_nano
  for scene in "${SCENE_ARR[@]}"; do
    [[ "$scene" == "S0" ]] && continue
    log "nano $VARIANT $scene"
    bench_one "$NANO_PORT" "$scene"
  done
  if [[ "$DRY_RUN" != "1" && "${NANO_PID:-}" ]]; then
    kill "$NANO_PID" 2>/dev/null || true
    wait "$NANO_PID" 2>/dev/null || true
    unset NANO_PID
  fi
fi

if [[ "$ENGINE" == "vllm" || "$ENGINE" == "both" ]]; then
  start_vllm
  for scene in "${SCENE_ARR[@]}"; do
    [[ "$scene" == "S0" ]] && continue
    log "vllm $VARIANT $scene"
    bench_one "$VLLM_PORT" "$scene"
  done
fi

if [[ ",$SCENES," == *",S0,"* ]]; then
  log "S0: reproduce offline baseline with old script"
  run "$NANO_PY" "$ROOT/baseline.py" --model "${BASELINE_MODEL:-/root/autodl-tmp/huggingface/Qwen3-0.6B/}" \
    --suites throughput --save "$RESULT_DIR/s0_old_baseline.json"
fi

log "done"
