"""Protocol tests against the fake engine (no GPU)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from bench.fake_engine import FakeEngine
from bench.nano_server import apply_stop, create_app, split_visible, tokens_covering_text


def _client(output_text: str = "abcdefghijklmnopqrstuvwxyz", max_model_len: int = 4096) -> TestClient:
    engine = FakeEngine(output_text=output_text, max_model_len=max_model_len, step_sleep=0.0)
    app = create_app(engine, served_model_name="Qwen3-4B", max_model_len=max_model_len)
    return TestClient(app)


def test_apply_stop_keeps_prefix():
    text, matched = apply_stop("hello Question: rest", ["Question:"])
    assert matched == "Question:"
    assert text == "hello "


def test_split_visible_holds_stop_prefix():
    visible, matched = split_visible("hello Q", ["Question:"], flush=False)
    assert matched is None and visible == "hello "
    visible, matched = split_visible("hello Question:", ["Question:"], flush=False)
    assert matched == "Question:" and visible == "hello "


def test_health_and_models():
    with _client() as c:
        h = c.get("/health")
        assert h.status_code == 200 and h.json()["status"] == "ok"
        models = c.get("/v1/models").json()
        assert models["data"][0]["id"] == "Qwen3-4B"


def test_non_stream_completion():
    with _client(output_text="xyzxyz") as c:
        r = c.post("/v1/completions", json={
            "model": "Qwen3-4B",
            "prompt": "Hi",
            "max_tokens": 4,
            "temperature": 0,
            "ignore_eos": True,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion"
        text = body["choices"][0]["text"]
        assert text == "xyzx"
        assert body["choices"][0]["finish_reason"] == "length"
        usage = body["usage"]
        assert usage["prompt_tokens"] == 2
        assert usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 6


def test_stream_sse_and_usage():
    with _client(output_text="abcd") as c:
        r = c.post("/v1/completions", json={
            "model": "Qwen3-4B",
            "prompt": "Z",
            "max_tokens": 3,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
        })
        assert r.status_code == 200, r.text
        raw = r.text
    lines = [ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    chunks = [json.loads(ln) for ln in lines[:-1]]
    texts = [ch["choices"][0]["text"] for ch in chunks if ch.get("choices")]
    assert "".join(texts).startswith("abc")
    usage_chunks = [ch for ch in chunks if ch.get("usage") and not ch.get("choices")]
    assert usage_chunks, raw
    usage = usage_chunks[-1]["usage"]
    assert usage["completion_tokens"] == 3
    assert usage["prompt_tokens"] == 1


def test_stop_string_truncates():
    with _client(output_text="hello Question: rest") as c:
        r = c.post("/v1/completions", json={
            "model": "Qwen3-4B",
            "prompt": "p",
            "max_tokens": 40,
            "stop": "Question:",
            "ignore_eos": True,
            "temperature": 0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["choices"][0]["text"] == "hello "
        assert body["choices"][0]["finish_reason"] == "stop"


def test_max_model_len_rejected():
    with _client(max_model_len=8) as c:
        r = c.post("/v1/completions", json={
            "model": "Qwen3-4B",
            "prompt": "abcdefgh",
            "max_tokens": 4,
        })
        assert r.status_code == 400
        assert "max_model_len" in r.json()["detail"]


def test_metrics_reset():
    with _client() as c:
        c.post("/v1/completions", json={
            "model": "Qwen3-4B", "prompt": "Hi", "max_tokens": 2, "ignore_eos": True,
        })
        metrics = c.get("/nano/metrics").json()
        assert metrics["num_finished_requests"] >= 1
        assert c.post("/nano/reset_metrics").status_code == 200
        assert c.get("/nano/metrics").json()["num_finished_requests"] == 0


def test_tokens_covering_text():
    class T:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    n = tokens_covering_text(T(), [104, 105, 32, 81], "hi ")
    assert n == 3
