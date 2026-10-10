"""生成与对账校验。

extractive：零 LLM 的抽取式作答——从命中块里挑证据句拼装，
每句挂引用。演示永不翻车的底气所在。
verify：引用覆盖率校验——答案句与被引块的 token 重叠率低于阈值
即降级/拒答。「答案必须能对账」的最后一道闸。
"""

from __future__ import annotations

import re

from duizhang.schemas import GraphState, Retrieved

from duizhang.index import word_tokens

_SENT_SPLIT = re.compile(r"(?<=[。；！？])\s*")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= 6]


def extractive_answer(state: GraphState, hits: list[Retrieved],
                      max_sents: int = 4) -> tuple[str, list[str]]:
    """从 top 命中块抽证据句：句与 query 的重叠分 + 位置先验。"""
    q = set(word_tokens(state["effective_query"]))
    # 问「原因/变动」时给归因句加权：财报的因果句式是「主要系…影响」「主要由于…所致」
    wants_reason = any(w in state["query"] for w in ("原因", "为什么", "为何", "变动"))
    _REASON_PAT = re.compile(r"主要系|主要因|主要由于|所致|影响")
    scored = []
    for h in hits[:4]:
        for s in _sentences(h.chunk.text):
            stoks = set(word_tokens(s)) | set(word_tokens(h.chunk.text[:40]))
            overlap = len(q & stoks)
            # 数字句优先（财报问答大多数时候在问数字）
            has_num = bool(re.search(r"\d", s)) or bool(h.chunk.num_notes)
            score = overlap * (1.4 if has_num else 1.0) + (0.2 if h.chunk.num_notes else 0.0)
            if wants_reason and _REASON_PAT.search(s):
                score += 1.5
            scored.append((score, s, h.chunk))
    scored.sort(key=lambda x: -x[0])
    picked, cited = [], []
    for score, s, chunk in scored:
        if score <= 0:
            continue
        picked.append(f"{s} {chunk.citation}")
        if chunk.citation not in cited:
            cited.append(chunk.citation)
        if len(picked) >= max_sents:
            break
    if not picked:
        return "", []
    notes = "；".join(hits[0].chunk.num_notes[:2]) if hits[0].chunk.num_notes else ""
    body = "\n".join(f"· {p}" for p in picked)
    if notes:
        body = f"{body}\n（数字参考：{notes}）"
    return body, cited


# ── 对账校验 ────────────────────────────────────────────


def citation_coverage(answer: str, hits: list[Retrieved]) -> float:
    """答案中被引用块实际支撑的比例：句 token 与被引块的重叠率均值。"""
    sents = [s for s in _SENT_SPLIT.split(re.sub(r"\[[^\]]*\]", "", answer))
             if len(s.strip()) >= 6]
    if not sents:
        return 0.0
    evidence = " ".join(h.chunk.text for h in hits[:4])
    etoks = set(word_tokens(evidence))
    cov = []
    for s in sents:
        stoks = set(word_tokens(s))
        if not stoks:
            continue
        cov.append(len(stoks & etoks) / len(stoks))
    return sum(cov) / len(cov) if cov else 0.0


def has_citation(answer: str) -> bool:
    return bool(re.search(r"\[[^\]]+第\d+页\]|\[[^\]]+_\d{4}[A-Z0-9]*\]", answer))


COVERAGE_THRESHOLD = 0.45   # 抽取式天然高；LLM 生成低于此值说明在编


def verify(answer: str, hits: list[Retrieved]) -> tuple[str, float]:
    """返回 (verdict, coverage)：pass / degraded（降级重写为抽取式由图负责）。"""
    if not answer.strip():
        return "refused", 0.0
    cov = citation_coverage(answer, hits)
    if cov >= COVERAGE_THRESHOLD and has_citation(answer):
        return "pass", round(cov, 3)
    return "degraded", round(cov, 3)
