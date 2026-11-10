# Offline experiments

Fork-only GPU scripts for correctness and ablation. They call `LLM.generate()`
directly and write JSON under `results/` at the repo root.

The HTTP serving harness lives in `bench/`. Upstream's microbench is still
`bench.py` at the repo root.

Run from the repository root (editable install optional; `_root.py` puts the
repo on `sys.path`):

```bash
python experiments/baseline.py --save opt_xxx
python experiments/baseline.py --compare results/baseline_v0.json results/opt_xxx.json
python experiments/align_hf.py
python experiments/align_hf_hybrid.py
python experiments/ablate_schedule.py
python experiments/ablate_ngram.py
python experiments/ablate_int8_kv.py
```

`--save` / `--out` names land in `<repo>/results/<name>.json` regardless of cwd.
Absolute paths are kept as-is.
