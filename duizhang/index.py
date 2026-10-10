"""索引：双路 BM25（词级 + 字级 bigram）RRF 融合，sqlite 持久化。

为什么不上向量模型：v0.1 的第一原则是「一条 pip install 后可复现」——
中文词法双路 + RRF 在财报问答这种术语密集语料上已是强基线；
向量检索列入 roadmap（接口已按可替换设计）。
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from duizhang.schemas import Chunk, Doc, Retrieved

_CUT = jieba.lcut
_TOKEN_STRIP = re.compile(r"[\s，。；：、！？（）()《》【】\[\]{}\"'“”‘’,.'']")


def word_tokens(text: str) -> list[str]:
    return [t for t in _CUT(text) if t.strip() and not _TOKEN_STRIP.fullmatch(t)]


def char_bigrams(text: str) -> list[str]:
    s = re.sub(r"\s+", "", text)
    return [s[i:i + 2] for i in range(len(s) - 1)]


class Corpus:
    """chunk 库 + 双路 BM25 索引。"""

    def __init__(self) -> None:
        self.docs: dict[str, Doc] = {}
        self.chunks: list[Chunk] = []
        self._by_doc: dict[str, list[int]] = {}
        self._bm25_word: BM25Okapi | None = None
        self._bm25_char: BM25Okapi | None = None
        self._wtoks: list[list[str]] = []
        self._ctoks: list[list[str]] = []

    # ── 构建 ──
    def add(self, doc: Doc, chunks: list[Chunk]) -> None:
        self.docs[doc.doc_id] = doc
        base = len(self.chunks)
        self.chunks.extend(chunks)
        for c in chunks:
            self._by_doc.setdefault(c.doc_id, []).append(base)
            base += 1
        self._rebuild()

    def _rebuild(self) -> None:
        texts = [f"{c.text} {' '.join(c.num_notes)} {self.docs[c.doc_id].company}"
                 f" {self.docs[c.doc_id].kind}" for c in self.chunks]
        self._wtoks = [word_tokens(t) for t in texts]
        self._ctoks = [char_bigrams(t) for t in texts]
        self._bm25_word = BM25Okapi(self._wtoks) if self._wtoks else None
        self._bm25_char = BM25Okapi(self._ctoks) if self._ctoks else None

    # ── 检索：双路 RRF ──
    def search(self, query: str, k: int = 8,
               doc_filter: str = "") -> list[Retrieved]:
        if not self.chunks:
            return []
        q_word, q_char = word_tokens(query), char_bigrams(query)
        cand = range(len(self.chunks))
        if doc_filter:
            allow = {i for d, idx in self._by_doc.items()
                     if doc_filter in d or doc_filter in self.docs[d].company
                     for i in idx}
            if allow:
                cand = sorted(allow)

        scores_w = self._bm25_word.get_scores(q_word) if q_word and self._bm25_word else {}
        scores_c = self._bm25_char.get_scores(q_char) if self._bm25_char else {}

        def top(scores) -> list[tuple[int, float]]:
            pairs = [(i, scores[i]) for i in cand if scores[i] > 0]
            pairs.sort(key=lambda x: -x[1])
            return pairs[: k * 3]

        # Reciprocal Rank Fusion, k=60 惯例
        rrf: dict[int, float] = {}
        for ranking in (top(scores_w), top(scores_c)):
            for rank, (i, _) in enumerate(ranking, start=1):
                rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank)

        fused = sorted(rrf.items(), key=lambda x: -x[1])[:k]
        return [Retrieved(chunk=self.chunks[i], score=round(s, 4), rank=r)
                for r, (i, s) in enumerate(fused, start=1)]

    # ── 持久化（sqlite：结构与内容分离，重建索引靠重算） ──
    def save(self, db_path: str | Path) -> None:
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE IF NOT EXISTS docs "
                    "(doc_id TEXT PRIMARY KEY, company TEXT, period TEXT, kind TEXT,"
                    " source TEXT, n_pages INT)")
        con.execute("CREATE TABLE IF NOT EXISTS chunks "
                    "(chunk_id TEXT PRIMARY KEY, doc_id TEXT, page INT, seq INT,"
                    " text TEXT, is_table INT, num_notes TEXT)")
        for d in self.docs.values():
            con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)",
                        (d.doc_id, d.company, d.period, d.kind, d.source, d.n_pages))
        for c in self.chunks:
            con.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                        (c.chunk_id, c.doc_id, c.page, c.seq, c.text,
                         int(c.is_table), json.dumps(c.num_notes, ensure_ascii=False)))
        con.commit()
        con.close()

    @classmethod
    def load(cls, db_path: str | Path) -> "Corpus":
        corp = cls()
        con = sqlite3.connect(str(db_path))
        try:
            for row in con.execute("SELECT doc_id,company,period,kind,source,n_pages FROM docs"):
                d = Doc(*row)
                corp.docs[d.doc_id] = d
            for row in con.execute("SELECT chunk_id,doc_id,page,seq,text,is_table,num_notes FROM chunks"):
                cid, did, page, seq, text, is_table, notes = row
                corp.chunks.append(Chunk(cid, did, page, seq, text,
                                         bool(is_table), json.loads(notes)))
            for i, c in enumerate(corp.chunks):
                corp._by_doc.setdefault(c.doc_id, []).append(i)
            corp._rebuild()
        finally:
            con.close()
        return corp
