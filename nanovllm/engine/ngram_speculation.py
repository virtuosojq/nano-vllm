"""2.2 Prompt-lookup / n-gram 投机解码: 草稿匹配 + 贪心 verify 接受.

不引入草稿模型. 验证前向在 ModelRunner.run_ngram 中完成; 本模块只做:
  1) 从历史 token 里找 n-gram 续写作为 draft
  2) 对 [K+1, V] logits 做贪心前缀接受 + bonus token
"""

from __future__ import annotations


def propose_ngram_drafts(
    token_ids: list[int],
    max_draft: int,
    *,
    min_ngram: int = 1,
    max_ngram: int = 3,
) -> list[int]:
    """在 token_ids 中查找与后缀匹配的 n-gram, 取其后续最多 max_draft 个 token.

    从大到小试 n ∈ [max_ngram, min_ngram]; 无匹配则返回 [].
    """
    if max_draft <= 0 or len(token_ids) < min_ngram + 1:
        return []
    max_ngram = min(max_ngram, len(token_ids) - 1)
    for n in range(max_ngram, min_ngram - 1, -1):
        needle = token_ids[-n:]
        # 在不含当前后缀的前缀里找最后一次匹配, 取匹配后的续写
        limit = len(token_ids) - n
        for start in range(limit - 1, -1, -1):
            if token_ids[start : start + n] == needle:
                cont = token_ids[start + n : start + n + max_draft]
                if cont:
                    return list(cont)
                break
    return []


def accept_greedy_drafts(
    logits_row: "object",
    drafts: list[int],
) -> tuple[list[int], int]:
    """贪心投机接受.

    logits_row: Tensor [K+1, V] — 对应输入 [last]+drafts 各位置的 next-token logits.
    返回 (要 append 的 token 列表, 接受的 draft 个数).
    列表长度 = accepted_drafts + 1 (含拒绝位重采样或全接受后的 bonus).
    """
    import torch

    assert isinstance(logits_row, torch.Tensor)
    k = len(drafts)
    assert logits_row.size(0) == k + 1
    preds = logits_row.argmax(dim=-1).tolist()
    accepted = 0
    for i, d in enumerate(drafts):
        if preds[i] == d:
            accepted += 1
        else:
            # 拒绝: 前 accepted 个 draft + 当前位置的 argmax
            return drafts[:accepted] + [preds[i]], accepted
    # 全部接受: drafts + bonus
    return drafts + [preds[k]], accepted
