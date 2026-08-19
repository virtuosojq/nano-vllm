"""2.2 n-gram draft / accept 单元测试 (不加载模型)."""

from nanovllm.engine.ngram_speculation import propose_ngram_drafts, accept_greedy_drafts
import torch


def test_propose_finds_continuation():
    # ... a b c X Y | a b c  → draft X Y
    ids = [0, 1, 2, 10, 11, 0, 1, 2]
    assert propose_ngram_drafts(ids, max_draft=2, min_ngram=1, max_ngram=3) == [10, 11]


def test_propose_empty_when_no_match():
    ids = [1, 2, 3, 4, 5]
    assert propose_ngram_drafts(ids, max_draft=4, min_ngram=1, max_ngram=3) == []


def test_accept_all_drafts_plus_bonus():
    # drafts match argmax at 0..1; bonus from pos 2
    logits = torch.zeros(3, 8)
    logits[0, 3] = 10
    logits[1, 4] = 10
    logits[2, 5] = 10
    tokens, acc = accept_greedy_drafts(logits, [3, 4])
    assert tokens == [3, 4, 5]
    assert acc == 2


def test_accept_reject_first():
    logits = torch.zeros(2, 8)
    logits[0, 7] = 10  # model wants 7, draft says 1
    logits[1, 2] = 10
    tokens, acc = accept_greedy_drafts(logits, [1])
    assert tokens == [7]
    assert acc == 0


def test_accept_empty_draft():
    logits = torch.zeros(1, 5)
    logits[0, 2] = 9
    tokens, acc = accept_greedy_drafts(logits, [])
    assert tokens == [2]
    assert acc == 0
