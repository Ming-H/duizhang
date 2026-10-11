"""FTS5 检索：词级（jieba 预分词 + unicode61）与字级（trigram）双路 RRF。

打分引擎从 v0.3 的自研内存倒排换为 SQLite FTS5 的 bm25()；
融合语义（RRF k=60）与 v0.3 完全一致。容量与内存解耦的兑现处。
"""

from __future__ import annotations

import re

from duizhang.schemas import Chunk, Retrieved

_CHUNK_COLS = ("ch.chunk_id, ch.doc_id, ch.page, ch.seq, ch.text, "
               "ch.is_table, ch.num_notes")

# 性能 PRAGMA：页缓存 200MB + mmap（OS 按需换页，冷查询的主要来源）。
# 注意 sqlite 的 PRAGMA 按连接生效——必须每次连接都设，不能做全局一次性。
_PRAGMAS = ("PRAGMA cache_size=-200000", "PRAGMA mmap_size=1073741824",
            "PRAGMA temp_store=MEMORY")


def _apply_pragmas(con):
    for p in _PRAGMAS:
        con.execute(p)


def _row_to_chunk(row) -> Chunk:
    import json
    return Chunk(chunk_id=row[0], doc_id=row[1], page=row[2], seq=row[3],
                 text=row[4], is_table=bool(row[5]), num_notes=json.loads(row[6]))


def _quote(term: str) -> str:
    """FTS5 MATCH 的安全词组：包裹引号并转义内嵌引号。"""
    return '"' + term.replace('"', '""') + '"'


def word_expr(query: str) -> str:
    """查询侧清洗：丢单字中文 token（的/年/亿等虚词与切割碎片，
    idf 极低却稀释 MATCH；数字与英文单字保留）。"""
    from duizhang.index import word_tokens
    toks = [t for t in dict.fromkeys(word_tokens(query))
            if not (len(t) == 1 and "\u4e00" <= t <= "\u9fff")]
    return " OR ".join(_quote(t) for t in toks)


def char_expr(query: str) -> str:
    """字级轨道：≥2 字词 token 做 trigram 短语 OR（子串命中，跨词边界）。

    平移 v0.3 字轨语义：bigram 集合匹配 ≈ 子串命中。不能用空格整段短语——
    「预计净利润多少」在原文里被年份隔开，整段短语永远不命中。
    """
    from duizhang.index import char_text, word_tokens
    segs = [char_text(t) for t in word_tokens(query)]
    segs = [s for s in segs if len(s) >= 2]
    return " OR ".join(_quote(s) for s in dict.fromkeys(segs))


def search(con, docs: dict, query: str, k: int = 8,
           doc_filter: str = "") -> list[Retrieved]:
    """双路 FTS5 → RRF 融合 → Retrieved 列表。

    docs: doc_id → Doc（公司过滤在 Python 侧展开为 doc_id 集合后 SQL 下推）。
    """
    allow_ids: list[str] | None = None
    if doc_filter:
        allow_ids = [d for d in docs
                     if doc_filter in d or doc_filter in docs[d].company]
        if not allow_ids:
            return []

    _apply_pragmas(con)

    def one_track(table: str, expr: str, limit: int):
        """两轨只取轻列（chunk_id + 关联 rowid）；正文延后到融合后取 top-k，
        避免两轨各拖 24 行长文本。"""
        if not expr:
            return []
        sql = (f"SELECT ch.chunk_id, {table}.rowid FROM {table} "
               f"JOIN chunks ch ON ch.rowid = {table}.rowid "
               f"WHERE {table} MATCH ? ")
        params: list = [expr]
        if allow_ids is not None:
            sql += f"AND ch.doc_id IN ({','.join('?' * len(allow_ids))}) "
            params += allow_ids
        sql += f"ORDER BY bm25({table}) LIMIT ?"
        params.append(limit)
        return con.execute(sql, params).fetchall()

    pool = k * 3
    rows_w = one_track("fts_w", word_expr(query), pool)
    rows_c = one_track("fts_c", char_expr(query), pool)

    # RRF k=60（与 v0.3 语义一致）
    rrf: dict[str, float] = {}
    rowid_of: dict[str, int] = {}
    for ranking in (rows_w, rows_c):
        for rank, row in enumerate(ranking, start=1):
            cid = row[0]
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
            if cid not in rowid_of:
                rowid_of[cid] = row[1]

    fused = sorted(rrf.items(), key=lambda x: -x[1])[:k]
    if not fused:
        return []
    # 融合后一次性取正文
    rowids = [rowid_of[cid] for cid, _ in fused]
    rows = con.execute(
        f"SELECT {_CHUNK_COLS.replace('ch.', 'chunks.')} FROM chunks WHERE rowid IN "
        f"({','.join('?' * len(rowids))}) ORDER BY rowid", rowids).fetchall()
    chunk_of = {r[0]: _row_to_chunk(r) for r in rows}
    return [Retrieved(chunk=chunk_of[cid], score=round(sc, 4), rank=r)
            for r, (cid, sc) in enumerate(fused, start=1)
            if cid in chunk_of]
