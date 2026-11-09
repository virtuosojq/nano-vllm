# OpenAI-compatible serving + eval harness

Implements `interview-plan/01-项目A-评测重做.md`: wrap nano-vllm so `vllm bench serve` can hit both engines.

## Local protocol check (no GPU)

```bash
pip install -e '.[serve]'
python -m bench.nano_server --fake --port 8001 --served-model-name Qwen3-4B
```

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B","prompt":"Hello","max_tokens":8,"temperature":0}'
curl -N http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B","prompt":"Hello","max_tokens":4,"stream":true,"stream_options":{"include_usage":true}}'
```

```bash
python -m pytest tests/test_openai_server.py tests/test_scheduler_abort.py -q
```

## AutoDL (4090 D)

Two conda envs (`nano` / `vllm`). Client always runs in `vllm`:

```bash
python tools/collect_env.py --model /root/autodl-tmp/models/Qwen3-4B -o results/v2/4090d/env.json
ENGINE=nano VARIANT=hybrid SCENES=S1,S2,S4 bash bench/run_matrix.sh
DRY_RUN=1 bash bench/run_matrix.sh   # print commands only
```
