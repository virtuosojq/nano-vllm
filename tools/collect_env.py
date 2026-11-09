#!/usr/bin/env python3
"""Collect hardware / software / git / model fingerprints for a bench run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _nvidia() -> dict:
    if not shutil.which("nvidia-smi"):
        return {}
    query = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ])
    cuda = _run(["nvidia-smi"])
    gpus = []
    if query:
        for line in query.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory": parts[1], "driver": parts[2]})
    return {"gpus": gpus, "nvidia_smi_ok": bool(query), "cuda_from_smi": bool(cuda)}


def _npu() -> dict:
    if not shutil.which("npu-smi"):
        return {}
    info = _run(["npu-smi", "info"])
    return {"npu_smi_ok": bool(info), "npu_smi_info_head": (info or "").splitlines()[:20]}


def _pkg(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def collect(model: str | None = None) -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = _run(["git", "-C", root, "rev-parse", "HEAD"])
    branch = _run(["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"])
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "git": {"commit": commit, "branch": branch, "root": root},
        "packages": {
            "torch": _pkg("torch"),
            "transformers": _pkg("transformers"),
            "triton": _pkg("triton"),
            "vllm": _pkg("vllm"),
            "flash_attn": _pkg("flash_attn"),
        },
        "nvidia": _nvidia(),
        "npu": _npu(),
        "model": model,
        "env_subset": {
            k: os.environ.get(k)
            for k in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES", "HF_ENDPOINT")
            if os.environ.get(k)
        },
    }
    try:
        import torch
        env["torch_cuda"] = {
            "version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        env["torch_cuda"] = None
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="")
    parser.add_argument("-o", "--output", default="")
    args = parser.parse_args()
    data = collect(args.model or None)
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
