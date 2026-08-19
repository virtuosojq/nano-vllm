"""引擎内轻量指标（1.2）：只埋三点，不做监控系统。"""

from __future__ import annotations

from dataclasses import dataclass, field


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summarize(vals: list[float]) -> dict:
    vals = sorted(vals)
    return {
        "mean": sum(vals) / len(vals) if vals else 0.0,
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p99": _percentile(vals, 99),
    }


@dataclass
class EngineMetrics:
    prefix_hit_blocks: int = 0
    prefix_req_blocks: int = 0
    kv_util_samples: list[float] = field(default_factory=list)
    request_ttft_s: list[float] = field(default_factory=list)
    request_tpot_s: list[float] = field(default_factory=list)
    # 2.2 n-gram
    ngram_draft_tokens: int = 0
    ngram_accepted_drafts: int = 0
    ngram_verify_steps: int = 0

    def reset(self) -> None:
        self.prefix_hit_blocks = 0
        self.prefix_req_blocks = 0
        self.kv_util_samples.clear()
        self.request_ttft_s.clear()
        self.request_tpot_s.clear()
        self.ngram_draft_tokens = 0
        self.ngram_accepted_drafts = 0
        self.ngram_verify_steps = 0

    def note_prefix(self, hit_blocks: int, req_blocks: int) -> None:
        self.prefix_hit_blocks += hit_blocks
        self.prefix_req_blocks += req_blocks

    def sample_kv(self, num_free: int, num_blocks: int) -> None:
        if num_blocks <= 0:
            return
        # 利用率 = 已用 / 总量
        self.kv_util_samples.append(1.0 - num_free / num_blocks)

    def note_finished(self, ttft_s: float, tpot_s: float | None) -> None:
        self.request_ttft_s.append(ttft_s)
        if tpot_s is not None:
            self.request_tpot_s.append(tpot_s)

    def note_ngram(self, draft_tokens: int, accepted_drafts: int) -> None:
        self.ngram_verify_steps += 1
        self.ngram_draft_tokens += draft_tokens
        self.ngram_accepted_drafts += accepted_drafts

    def summary(self) -> dict:
        hit_rate = (
            self.prefix_hit_blocks / self.prefix_req_blocks
            if self.prefix_req_blocks > 0 else 0.0
        )
        kv = self.kv_util_samples
        accept_rate = (
            self.ngram_accepted_drafts / self.ngram_draft_tokens
            if self.ngram_draft_tokens > 0 else 0.0
        )
        return {
            "prefix_cache_hit_rate": round(hit_rate, 4),
            "prefix_hit_blocks": self.prefix_hit_blocks,
            "prefix_req_blocks": self.prefix_req_blocks,
            "kv_utilization": {
                "mean": round(sum(kv) / len(kv), 4) if kv else 0.0,
                "peak": round(max(kv), 4) if kv else 0.0,
                "num_samples": len(kv),
            },
            "request_ttft_ms": {k: round(v * 1e3, 2) for k, v in _summarize(self.request_ttft_s).items()},
            "request_tpot_ms": {k: round(v * 1e3, 2) for k, v in _summarize(self.request_tpot_s).items()},
            "num_finished_requests": len(self.request_ttft_s),
            "ngram_accept_rate": round(accept_rate, 4),
            "ngram_draft_tokens": self.ngram_draft_tokens,
            "ngram_accepted_drafts": self.ngram_accepted_drafts,
            "ngram_verify_steps": self.ngram_verify_steps,
        }
