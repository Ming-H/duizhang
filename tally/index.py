"""索引：分词原语 + 瘦身 Corpus（v0.4）。

v0.3 的自研内存倒排（SparseBM25）与 pickle 缓存层已随 FTS5 切换删除
（A/B 验收见 docs/DESIGN-v0.4.md §5，基线对比记录在 commit message）。
Corpus 不再驻留全部 chunk：docs 元数据全量，chunk 按需经 SQL 取。
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import jieba

from tally import fts
from tally.schemas import Chunk, Doc, Retrieved

_CUT = jieba.lcut
_TOKEN_STRIP = re.compile(r"[\s，。；：、！？（）()《》【】\[\]{}\"'“”‘’,.'']")


def word_tokens(text: str) -> list[str]:
    return [sys.intern(t) for t in _CUT(text)
            if t.strip() and not _TOKEN_STRIP.fullmatch(t)]


def char_text(text: str, cap: int = 200) -> str:
    """去空白取前 cap 字：字级 trigram 轨道的入库形态。"""
    return re.sub(r"\s+", "", text)[:cap]


class Corpus:
    """检索门面：docs 全量 + FTS5 按需查询。启动只读 docs 表（千级）。"""

    def __init__(self) -> None:
        self.docs: dict[str, Doc] = {}
        self.db_path: str = ""
        self._con: sqlite3.Connection | None = None

    # ── 加载 ──

    @classmethod
    def load(cls, db_path: str | Path) -> "Corpus":
        corp = cls()
        corp.db_path = str(db_path)   # v0.7：L1 台账读写要按路径开 Store
        corp._con = sqlite3.connect(str(db_path))
        for row in corp._con.execute(
                "SELECT doc_id, company, period, kind, source, n_pages FROM docs"):
            d = Doc(*row)
            corp.docs[d.doc_id] = d
        return corp

    # ── 检索（委托 fts 模块） ──

    def search(self, query: str, k: int = 8, doc_filter: str = "",
               doc_scope: set[str] | None = None) -> list[Retrieved]:
        if not self._con or not self.docs:
            return []
        # 向量第三路（P1.2 可选）：依赖在 + 库已嵌入才生效，否则 None
        # → fts.search 双轨行为不变
        from tally import embed
        extra = None
        if embed.enabled():
            extra = embed.ranking(self._con, self.docs, query,
                                  doc_filter, doc_scope)
        return fts.search(self._con, self.docs, query, k=k, doc_filter=doc_filter,
                          doc_scope=doc_scope, extra_ranking=extra)

    def chunks_of(self, doc_id: str, limit: int = 12) -> list[Chunk]:
        """summarize 用：按文档取前 N 块（替代 v0.3 的全内存扫描）。"""
        import json
        rows = self._con.execute(
            "SELECT chunk_id, doc_id, page, seq, text, is_table, num_notes "
            "FROM chunks WHERE doc_id=? ORDER BY rowid LIMIT ?",
            (doc_id, limit)).fetchall()
        return [Chunk(r[0], r[1], r[2], r[3], r[4], bool(r[5]), json.loads(r[6]))
                for r in rows]

    def chunks_page(self, doc_id: str, page: int) -> list[Chunk]:
        """页级证据组装用：某文档某页的全部块，按页内序号排。"""
        import json
        rows = self._con.execute(
            "SELECT chunk_id, doc_id, page, seq, text, is_table, num_notes "
            "FROM chunks WHERE doc_id=? AND page=? ORDER BY seq",
            (doc_id, page)).fetchall()
        return [Chunk(r[0], r[1], r[2], r[3], r[4], bool(r[5]), json.loads(r[6]))
                for r in rows]

    def counts(self) -> tuple[int, int]:
        n_docs = len(self.docs)
        n_chunks = self._con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return n_docs, n_chunks

    def close(self) -> None:
        if self._con:
            self._con.close()
