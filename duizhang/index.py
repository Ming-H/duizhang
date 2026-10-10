"""索引：双路 BM25（词级 + 字级 bigram）RRF 融合，sqlite 持久化。

规模实测（v0.1 用 rank_bm25 纯 Python 打分）：
· 59.6 万 chunk、5 文档样例时毫秒级；全库时单次检索 4.9s —— 不可用。
· v0.2 换自建倒排 + numpy 向量化 BM25（同公式 k1=1.5 b=0.75）：
  打分只触碰包含查询词的 posting，毫秒级。
· 为什么不上向量数据库：60 万 chunk 用 numpy 暴力余弦也才毫秒级，
  ANN 基础设施在这个规模是过度设计（见 README 选型表）。
"""

from __future__ import annotations

import array
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import jieba
import numpy as np

from duizhang.schemas import Chunk, Doc, Retrieved

_CUT = jieba.lcut
_TOKEN_STRIP = re.compile(r"[\s，。；：、！？（）()《》【】\[\]{}\"'“”‘’,.'']")


def word_tokens(text: str) -> list[str]:
    return [sys.intern(t) for t in _CUT(text)
            if t.strip() and not _TOKEN_STRIP.fullmatch(t)]


def char_bigrams(text: str, cap: int = 200) -> list[str]:
    """只取前 cap 字符：头部已足够判别，全量 bigram 内存会到 GB 级。"""
    s = re.sub(r"\s+", "", text)[:cap]
    return [sys.intern(s[i:i + 2]) for i in range(len(s) - 1)]


class SparseBM25:
    """倒排 + numpy 的 BM25（Okapi, k1=1.5, b=0.75）。

    打分只访问查询词的 posting list——与文档总数解耦；
    纯 Python 全量打分是 O(N·query_terms) 的 python 循环，60 万块上秒级。
    """

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        n = len(docs_tokens)
        self.k1, self.b, self.n = k1, b, n
        self.dl = np.array([len(t) for t in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if n else 1.0

        ids: dict[str, array.array] = {}
        tfs: dict[str, array.array] = {}
        df: dict[str, int] = {}
        for i, toks in enumerate(docs_tokens):
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, tf in counts.items():
                bucket = ids.get(t)
                if bucket is None:
                    ids[t] = array.array("i", [i])
                    tfs[t] = array.array("i", [tf])
                    df[t] = 1
                else:
                    bucket.append(i)
                    tfs[t].append(tf)
                    df[t] += 1
        self.post_ids = {t: np.frombuffer(v, dtype=np.int32) for t, v in ids.items()}
        self.post_tf = {t: np.frombuffer(v, dtype=np.int32).astype(np.float32)
                        for t, v in tfs.items()}
        # Okapi idf（恒正变体）
        self.idf = {t: math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    for t in df}

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        norm = self.k1 * (1.0 - self.b + self.b * self.dl / self.avgdl)
        for t in set(query_tokens):
            ids_ = self.post_ids.get(t)
            if ids_ is None or ids_.size == 0:
                continue
            tf = self.post_tf[t]
            # 对号入座：贡献只落在该词的 posting 文档上（postings 内 doc 唯一，+= 安全）
            scores[ids_] += self.idf[t] * tf * (self.k1 + 1.0) / (tf + norm[ids_])
        return scores


def _fingerprint(chunks: list[Chunk]) -> str:
    import hashlib
    h = hashlib.sha1()
    for c in chunks:
        h.update(c.chunk_id.encode())
    return h.hexdigest()[:16]


class Corpus:
    """chunk 库 + 双路 BM25 索引。"""

    def __init__(self) -> None:
        self.docs: dict[str, Doc] = {}
        self.chunks: list[Chunk] = []
        self._by_doc: dict[str, list[int]] = {}
        self._bm25_word: SparseBM25 | None = None
        self._bm25_char: SparseBM25 | None = None

    # ── 构建 ──
    def add(self, doc: Doc, chunks: list[Chunk], rebuild: bool = True) -> None:
        """rebuild=False 用于批量摄取，最后调 finalize() 一次建索引（O(N) 而非 O(N²)）。"""
        self.docs[doc.doc_id] = doc
        base = len(self.chunks)
        self.chunks.extend(chunks)
        for c in chunks:
            self._by_doc.setdefault(c.doc_id, []).append(base)
            base += 1
        if rebuild:
            self.finalize()

    def finalize(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        texts = [f"{c.text} {' '.join(c.num_notes)} {self.docs[c.doc_id].company}"
                 f" {self.docs[c.doc_id].kind}" for c in self.chunks]
        # 字级轨道把公司/类型放最前：截断（200字）不能把元数据截掉
        char_texts = [f"{self.docs[c.doc_id].company} {self.docs[c.doc_id].kind} {c.text}"
                      for c in self.chunks]
        self._bm25_word = SparseBM25([word_tokens(t) for t in texts]) if texts else None
        self._bm25_char = SparseBM25([char_bigrams(t) for t in char_texts]) if char_texts else None

    # ── 检索：双路 RRF ──
    def search(self, query: str, k: int = 8,
               doc_filter: str = "") -> list[Retrieved]:
        if not self.chunks or not self._bm25_word:
            return []
        q_word, q_char = word_tokens(query), char_bigrams(query)
        scores_w = self._bm25_word.get_scores(q_word)
        scores_c = self._bm25_char.get_scores(q_char)

        n = len(self.chunks)

        def top(scores: np.ndarray) -> np.ndarray:
            """返回得分降序的 chunk 下标数组（仅正值）。"""
            if doc_filter:
                allow = np.zeros(n, dtype=bool)
                for d, idx in self._by_doc.items():
                    if doc_filter in d or doc_filter in self.docs[d].company:
                        allow[idx] = True
                if not allow.any():
                    return np.empty(0, dtype=np.int64)
                cand = np.nonzero(allow & (scores > 0))[0]
            else:
                cand = np.nonzero(scores > 0)[0]
            if cand.size == 0:
                return cand.astype(np.int64)
            if cand.size > k * 3:
                part = np.argpartition(-scores[cand], k * 3)[: k * 3]
                cand = cand[part]
            return cand[np.argsort(-scores[cand])]

        # Reciprocal Rank Fusion, k=60 惯例
        rrf: dict[int, float] = {}
        for ranking in (top(scores_w), top(scores_c)):
            for rank, i in enumerate(ranking.tolist(), start=1):
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
        con.executemany("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)",
                        [(d.doc_id, d.company, d.period, d.kind, d.source, d.n_pages)
                         for d in self.docs.values()])
        con.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                        [(c.chunk_id, c.doc_id, c.page, c.seq, c.text,
                          int(c.is_table), json.dumps(c.num_notes, ensure_ascii=False))
                         for c in self.chunks])
        con.commit()
        con.close()

    # ── 索引缓存：pickle 只是缓存，删了也能跑（重建 117s@60万块） ──
    def save_index(self, idx_path: str | Path) -> None:
        import pickle
        with open(idx_path, "wb") as f:
            pickle.dump({"fp": _fingerprint(self.chunks),
                         "word": self._bm25_word, "char": self._bm25_char}, f)

    @classmethod
    def load(cls, db_path: str | Path, idx_path: str | Path | None = None) -> "Corpus":
        corp = cls()
        con = sqlite3.connect(str(db_path))
        try:
            for row in con.execute("SELECT doc_id,company,period,kind,source,n_pages FROM docs"):
                d = Doc(*row)
                corp.docs[d.doc_id] = d
            corp.chunks = [Chunk(cid, did, page, seq, text, bool(is_table),
                                 json.loads(notes))
                           for cid, did, page, seq, text, is_table, notes
                           in con.execute("SELECT chunk_id,doc_id,page,seq,text,"
                                          "is_table,num_notes FROM chunks")]
            for i, c in enumerate(corp.chunks):
                corp._by_doc.setdefault(c.doc_id, []).append(i)
            if idx_path and Path(idx_path).exists():
                try:
                    import pickle
                    with open(idx_path, "rb") as f:
                        cached = pickle.load(f)
                    if cached.get("fp") == _fingerprint(corp.chunks):
                        corp._bm25_word = cached["word"]
                        corp._bm25_char = cached["char"]
                        return corp
                except Exception:
                    pass          # 缓存损坏 → 重建，行为不变只是慢一次
            corp._rebuild()
            if idx_path:
                try:
                    corp.save_index(idx_path)
                except Exception:
                    pass
        finally:
            con.close()
        return corp
