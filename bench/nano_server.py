"""OpenAI-compatible HTTP front-end for nano-vllm.

Architecture: FastAPI event loop never calls engine.*; a dedicated thread
owns add_request / step / abort and pushes token deltas back through
asyncio.Queue via call_soon_threadsafe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nanovllm.sampling_params import SamplingParams

SHUTDOWN = object()


def _dumps(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def apply_stop(text: str, stops: list[str]) -> tuple[str, str | None]:
    matched = None
    cut = None
    for stop in stops:
        if not stop:
            continue
        idx = text.find(stop)
        if idx >= 0 and (cut is None or idx < cut):
            cut = idx
            matched = stop
    if cut is None:
        return text, None
    return text[:cut], matched


def split_visible(text: str, stops: list[str], *, flush: bool) -> tuple[str, str | None]:
    """Text that is safe to emit. Hold back a suffix that is a stop prefix unless flushing."""
    truncated, matched = apply_stop(text, stops)
    if matched or flush or not stops:
        return truncated, matched
    hold = 0
    for stop in stops:
        limit = min(len(stop) - 1, len(truncated))
        for k in range(limit, 0, -1):
            if truncated.endswith(stop[:k]):
                hold = max(hold, k)
                break
    if hold:
        return truncated[:-hold], None
    return truncated, None


def tokens_covering_text(tokenizer, token_ids: list[int], text: str) -> int:
    if not text:
        return 0
    for n in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:n], skip_special_tokens=True)
        if decoded == text or decoded.startswith(text):
            return n
    return len(token_ids)


@dataclass
class ServeRequest:
    req_id: str
    prompt: str | list[int]
    sampling_params: SamplingParams
    stop: list[str]
    include_usage: bool
    arrival_time: float
    loop: asyncio.AbstractEventLoop
    out_q: asyncio.Queue
    seq_id: int | None = None
    prompt_tokens: int = 0
    pushed_tokens: int = 0
    decoded_text: str = ""
    finished: bool = False


@dataclass
class AbortRequest:
    req_id: str | None = None
    seq_id: int | None = None


@dataclass
class EngineThread:
    engine: Any
    served_model_name: str
    max_model_len: int
    inbox: queue.Queue = field(default_factory=queue.Queue)
    inflight: dict[int, ServeRequest] = field(default_factory=dict)
    by_req_id: dict[str, ServeRequest] = field(default_factory=dict)
    pending_aborts: set[str] = field(default_factory=set)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="nano-engine", daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.inbox.put(SHUTDOWN)
        if self.thread is not None:
            self.thread.join(timeout=5)
        exit_fn = getattr(self.engine, "exit", None)
        if callable(exit_fn):
            exit_fn()

    def submit(self, item: ServeRequest | AbortRequest) -> None:
        self.inbox.put(item)

    def _push(self, req: ServeRequest, item: dict) -> None:
        def _put() -> None:
            req.out_q.put_nowait(item)

        req.loop.call_soon_threadsafe(_put)

    def _handle_new(self, req: ServeRequest) -> None:
        if req.req_id in self.pending_aborts:
            self.pending_aborts.discard(req.req_id)
            req.finished = True
            return
        seq_id = self.engine.add_request(req.prompt, req.sampling_params)
        seq = self.engine.get_seq(seq_id)
        req.seq_id = seq_id
        req.prompt_tokens = seq.num_prompt_tokens if seq is not None else 0
        if req.prompt_tokens + req.sampling_params.max_tokens > self.max_model_len:
            self.engine.abort(seq_id)
            self._push(req, {
                "type": "error",
                "status": 400,
                "message": (
                    f"prompt ({req.prompt_tokens}) + max_tokens "
                    f"({req.sampling_params.max_tokens}) exceeds max_model_len "
                    f"({self.max_model_len})"
                ),
            })
            req.finished = True
            return
        self.inflight[seq_id] = req
        self.by_req_id[req.req_id] = req

    def _handle_abort(self, item: AbortRequest) -> None:
        req = None
        if item.seq_id is not None:
            req = self.inflight.get(item.seq_id)
        if req is None and item.req_id is not None:
            req = self.by_req_id.get(item.req_id)
        if req is None or req.finished:
            if item.req_id:
                self.pending_aborts.add(item.req_id)
            return
        if req.seq_id is not None:
            self.engine.abort(req.seq_id)
            self.inflight.pop(req.seq_id, None)
        self.by_req_id.pop(req.req_id, None)
        req.finished = True

    def _finish_reason(self, req: ServeRequest, token_ids: list[int], matched_stop: str | None) -> str:
        if matched_stop:
            return "stop"
        eos = getattr(self.engine.tokenizer, "eos_token_id", None)
        if (
            token_ids
            and eos is not None
            and token_ids[-1] == eos
            and not req.sampling_params.ignore_eos
        ):
            return "stop"
        return "length"

    def _publish(self, req: ServeRequest, token_ids: list[int], engine_finished: bool) -> None:
        if req.finished:
            return
        full = self.engine.tokenizer.decode(token_ids, skip_special_tokens=True)
        visible, matched = split_visible(full, req.stop, flush=engine_finished)
        if matched:
            token_ids = token_ids[: tokens_covering_text(self.engine.tokenizer, token_ids, visible)]
            engine_finished = True
            if req.seq_id is not None:
                self.engine.abort(req.seq_id)
        if visible.startswith(req.decoded_text):
            delta = visible[len(req.decoded_text):]
        else:
            delta = ""
        req.decoded_text = visible
        req.pushed_tokens = len(token_ids)
        if delta:
            self._push(req, {"type": "delta", "text": delta, "finish_reason": None})
        if not engine_finished:
            return
        reason = self._finish_reason(req, token_ids, matched)
        usage = {
            "prompt_tokens": req.prompt_tokens,
            "completion_tokens": len(token_ids),
            "total_tokens": req.prompt_tokens + len(token_ids),
        }
        self._push(req, {
            "type": "done",
            "finish_reason": reason,
            "usage": usage,
            "include_usage": req.include_usage,
        })
        req.finished = True
        if req.seq_id is not None:
            self.inflight.pop(req.seq_id, None)
        self.by_req_id.pop(req.req_id, None)

    def _scan(self, step_outputs: list[tuple[int, list[int]]]) -> None:
        finished_map = {seq_id: ids for seq_id, ids in step_outputs}
        for seq_id, req in list(self.inflight.items()):
            seq = self.engine.get_seq(seq_id)
            if seq is not None:
                token_ids = seq.completion_token_ids
                engine_finished = seq.is_finished
            elif seq_id in finished_map:
                token_ids = finished_map[seq_id]
                engine_finished = True
            else:
                continue
            if not token_ids and not engine_finished:
                continue
            self._publish(req, list(token_ids), engine_finished)

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            batch: list[Any] = []
            idle = self.engine.is_finished() and not self.inflight
            if idle:
                try:
                    batch.append(self.inbox.get(timeout=0.5))
                except queue.Empty:
                    continue
            while True:
                try:
                    batch.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            stop = False
            for item in batch:
                if item is SHUTDOWN:
                    stop = True
                    break
                if isinstance(item, ServeRequest):
                    self._handle_new(item)
                elif isinstance(item, AbortRequest):
                    self._handle_abort(item)
            if stop:
                break
            if self.inflight and not self.engine.is_finished():
                outputs, _ = self.engine.step()
                self._scan(outputs)
            elif self.inflight:
                self._scan([])


def create_app(engine: Any, served_model_name: str, max_model_len: int | None = None) -> FastAPI:
    if max_model_len is None:
        cfg = getattr(engine, "config", None)
        max_model_len = getattr(cfg, "max_model_len", 4096)
    worker = EngineThread(engine=engine, served_model_name=served_model_name, max_model_len=max_model_len)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker.start()
        yield
        worker.shutdown()

    app = FastAPI(title="nano-vllm OpenAI server", version="0.2.0", lifespan=lifespan)
    app.state.worker = worker
    app.state.served_model_name = served_model_name

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> dict:
        return {
            "object": "list",
            "data": [{
                "id": served_model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nano-vllm",
            }],
        }

    @app.get("/nano/metrics")
    def nano_metrics() -> dict:
        return engine.get_metrics()

    @app.post("/nano/reset_metrics")
    def reset_metrics() -> dict:
        engine.reset_metrics()
        return {"status": "ok"}

    def _normalize_prompt(prompt: Any) -> str | list[int]:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return [int(x) for x in prompt]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
            return prompt[0]
        raise HTTPException(status_code=400, detail="prompt must be a string or list of token ids")

    def _normalize_stop(stop: Any) -> list[str]:
        if stop is None:
            return []
        if isinstance(stop, str):
            return [stop]
        if isinstance(stop, list):
            return [str(s) for s in stop if s]
        return []

    def _completion_payload(
        req_id: str,
        text: str,
        finish_reason: str | None,
        usage: dict | None,
        *,
        stream: bool,
        include_text: bool = True,
    ) -> dict:
        created = int(time.time())
        body: dict[str, Any] = {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": served_model_name,
        }
        if include_text:
            body["choices"] = [{
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": finish_reason,
            }]
        else:
            body["choices"] = []
        if usage is not None:
            body["usage"] = usage
        return body

    async def _consume_done(req: ServeRequest, http_request: Request, stream: bool):
        try:
            while True:
                if await http_request.is_disconnected():
                    worker.submit(AbortRequest(req_id=req.req_id, seq_id=req.seq_id))
                    return
                try:
                    item = await asyncio.wait_for(req.out_q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if item.get("type") == "error":
                    raise HTTPException(status_code=item.get("status", 500), detail=item["message"])
                if item.get("type") == "delta":
                    if stream:
                        yield _dumps(_completion_payload(
                            req.req_id, item["text"], None, None, stream=True,
                        ))
                    else:
                        yield item
                elif item.get("type") == "done":
                    yield item
                    return
        except asyncio.CancelledError:
            worker.submit(AbortRequest(req_id=req.req_id, seq_id=req.seq_id))
            raise

    @app.post("/v1/completions")
    async def completions(raw: Request):
        body = await raw.json()
        prompt = _normalize_prompt(body.get("prompt"))
        max_tokens = int(body.get("max_tokens") or 16)
        temperature = float(body.get("temperature", 1.0))
        ignore_eos = bool(body.get("ignore_eos", False))
        stream = bool(body.get("stream", False))
        stream_options = body.get("stream_options") or {}
        include_usage = bool(stream_options.get("include_usage", True))
        stop = _normalize_stop(body.get("stop"))
        model_name = body.get("model") or served_model_name
        if model_name != served_model_name:
            # vllm bench serve sends --model; accept any name but serve this one.
            pass

        if isinstance(prompt, str):
            n_prompt = len(engine.tokenizer.encode(prompt))
        else:
            n_prompt = len(prompt)
        if n_prompt + max_tokens > worker.max_model_len:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"prompt ({n_prompt}) + max_tokens ({max_tokens}) exceeds "
                    f"max_model_len ({worker.max_model_len})"
                ),
            )
        if n_prompt <= 0:
            raise HTTPException(status_code=400, detail="empty prompt")

        sp = SamplingParams(temperature=max(temperature, 0.0), max_tokens=max_tokens, ignore_eos=ignore_eos)
        req_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        serve_req = ServeRequest(
            req_id=req_id,
            prompt=prompt,
            sampling_params=sp,
            stop=stop,
            include_usage=include_usage,
            arrival_time=time.perf_counter(),
            loop=asyncio.get_running_loop(),
            out_q=asyncio.Queue(),
        )
        worker.submit(serve_req)

        if stream:
            async def event_stream():
                async for item in _consume_done(serve_req, raw, stream=True):
                    if isinstance(item, str):
                        yield f"data: {item}\n\n"
                        continue
                    finish_reason = item["finish_reason"]
                    usage = item["usage"]
                    yield f"data: {_dumps(_completion_payload(req_id, '', finish_reason, None, stream=True))}\n\n"
                    if include_usage:
                        yield f"data: {_dumps(_completion_payload(req_id, '', None, usage, stream=True, include_text=False))}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        finish_reason = "length"
        usage = {"prompt_tokens": n_prompt, "completion_tokens": 0, "total_tokens": n_prompt}
        async for item in _consume_done(serve_req, raw, stream=False):
            if isinstance(item, dict) and item.get("type") == "done":
                finish_reason = item["finish_reason"]
                usage = item["usage"]
        return JSONResponse(
            _completion_payload(req_id, serve_req.decoded_text, finish_reason, usage, stream=False)
        )

    return app


def build_engine(args: argparse.Namespace):
    if args.fake:
        from bench.fake_engine import FakeEngine
        return FakeEngine(
            output_text=args.fake_output_text,
            step_sleep=args.fake_step_sleep,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
        )
    from nanovllm import LLM
    return LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        schedule_policy=args.schedule_policy,
        prefill_token_budget=args.prefill_token_budget,
        enable_ngram=args.enable_ngram,
        ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
        ngram_prompt_lookup_min=args.ngram_prompt_lookup_min,
        ngram_max_ngram_size=args.ngram_max_ngram_size,
        kv_cache_dtype=args.kv_cache_dtype,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenAI-compatible server for nano-vllm")
    p.add_argument("--model", default="", help="HF model directory (required unless --fake)")
    p.add_argument("--served-model-name", default="nano-vllm")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--schedule-policy", default="prefill_priority",
                   choices=["prefill_priority", "decode_priority", "token_budget", "hybrid"])
    p.add_argument("--prefill-token-budget", type=int, default=512)
    p.add_argument("--enable-ngram", action="store_true")
    p.add_argument("--ngram-prompt-lookup-max", type=int, default=4)
    p.add_argument("--ngram-prompt-lookup-min", type=int, default=1)
    p.add_argument("--ngram-max-ngram-size", type=int, default=3)
    p.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "int8"])
    p.add_argument("--fake", action="store_true", help="GPU-free fake engine for protocol tests")
    p.add_argument("--fake-output-text", default="x" * 4096)
    p.add_argument("--fake-step-sleep", type=float, default=0.0)
    args = p.parse_args(argv)
    if not args.fake and not args.model:
        p.error("--model is required unless --fake is set")
    return args


def main(argv: list[str] | None = None) -> None:
    import uvicorn
    args = parse_args(argv)
    engine = build_engine(args)
    app = create_app(engine, args.served_model_name, args.max_model_len)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
