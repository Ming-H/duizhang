"""Store：docs / chunks / fts / manifest 四表的唯一写入口。

收敛动机（v0.4 设计 D-3/D-4）：此前 cli 直写 SQL 与 Corpus.save 双轨并存，
更新路径漏删旧行产生过「幽灵旧块」。所有写入经此模块，事务边界与
fts rowid 对齐只在one place成立。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from duizhang.index import char_text, word_tokens
from duizhang.schemas import Chunk, Doc

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
  doc_id TEXT PRIMARY KEY, company TEXT, period TEXT, kind TEXT,
  source TEXT, n_pages INT);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, page INT, seq INT,
  text TEXT, is_table INT, num_notes TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS manifest (
  path TEXT PRIMARY KEY, doc_id TEXT, fingerprint TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS vecs (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, vec BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_vecs_doc ON vecs(doc_id);
CREATE TABLE IF NOT EXISTS vecmeta (
  model TEXT PRIMARY KEY);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_w USING fts5(content_w, tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_c USING fts5(content_c, tokenize='trigram');
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.con = sqlite3.connect(self.db_path)
        # WAL 是库级持久属性；synchronous/cache 是连接级。
        # 写连接开 NORMAL + 大页缓存：增量写入与 rebuild_fts 受益。
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=NORMAL")
        self.con.execute("PRAGMA cache_size=-200000")
        self.con.executescript(SCHEMA)
        self.con.commit()

    def close(self) -> None:
        self.con.close()

    # ── 读 ──

    def doc_ids(self) -> set[str]:
        return {r[0] for r in self.con.execute("SELECT doc_id FROM docs")}

    def sources(self) -> dict[str, str]:
        """doc_id → source path（孤儿清理与 manifest 引导用）。"""
        return dict(self.con.execute("SELECT doc_id, source FROM docs"))

    def manifest(self) -> dict[str, str]:
        try:
            return dict(self.con.execute("SELECT path, fingerprint FROM manifest"))
        except sqlite3.OperationalError:
            return {}

    def counts(self) -> tuple[int, int]:
        n_docs = self.con.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        n_chunks = self.con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return n_docs, n_chunks

    # ── 写（事务） ──

    def add_doc(self, doc: Doc, chunks: list[Chunk]) -> None:
        """四表同事务写入；fts rowid 对齐 chunks.rowid（lastrowid）。

        幂等契约：同 doc_id 重复写入 = 完全替换（先清后写）。不能靠
        INSERT OR REPLACE——chunks 的 REPLACE 会换新 rowid，旧 rowid 的
        fts 行不被覆盖，留下幽灵命中。
        """
        con = self.con
        self.drop_doc(doc.doc_id)
        with con:
            con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)",
                        (doc.doc_id, doc.company, doc.period, doc.kind,
                         doc.source, doc.n_pages))
            for c in chunks:
                cur = con.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                    (c.chunk_id, c.doc_id, c.page, c.seq, c.text,
                     int(c.is_table), json.dumps(c.num_notes, ensure_ascii=False)))
                rowid = cur.lastrowid
                con.execute("INSERT INTO fts_w(rowid, content_w) VALUES (?,?)",
                            (rowid, " ".join(word_tokens(c.text))))
                con.execute("INSERT INTO fts_c(rowid, content_c) VALUES (?,?)",
                            (rowid, char_text(c.text)))

    def drop_doc(self, doc_id: str) -> None:
        """四表删除。fts 与 chunks 的 rowid 对齐在删除侧同样成立：
        先收 rowid，再删 fts，最后删 chunks。"""
        con = self.con
        with con:
            rowids = [r[0] for r in con.execute(
                "SELECT rowid FROM chunks WHERE doc_id=?", (doc_id,))]
            con.executemany("DELETE FROM fts_w WHERE rowid=?", [(r,) for r in rowids])
            con.executemany("DELETE FROM fts_c WHERE rowid=?", [(r,) for r in rowids])
            con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            con.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
            con.execute("DELETE FROM manifest WHERE doc_id=?", (doc_id,))
            con.execute("DELETE FROM vecs WHERE doc_id=?", (doc_id,))

    # ── 向量层（P1.2 可选路：bge-small-zh @ fastembed，DZ_VEC=off 关闭）──

    def vec_model(self) -> str:
        """当前向量库绑定的嵌入模型名（空 = 未嵌入）。"""
        row = self.con.execute("SELECT model FROM vecmeta LIMIT 1").fetchone()
        return row[0] if row else ""

    def set_vec_model(self, model: str) -> None:
        """绑定嵌入模型；换模型 → 旧向量全量作废（维度/语义空间不同，
        混用即错配）。"""
        with self.con:
            if self.vec_model() != model:
                self.con.execute("DELETE FROM vecs")
            self.con.execute("DELETE FROM vecmeta")
            self.con.execute("INSERT INTO vecmeta VALUES (?)", (model,))

    def add_vecs(self, rows: list[tuple[str, str, bytes]]) -> None:
        """批量写入向量（chunk_id, doc_id, float32 blob）。"""
        with self.con:
            self.con.executemany(
                "INSERT OR REPLACE INTO vecs VALUES (?,?,?)", rows)

    def vec_count(self) -> int:
        return self.con.execute("SELECT COUNT(*) FROM vecs").fetchone()[0]

    def vec_missing(self) -> list[tuple[str, str, str]]:
        """还没有向量的 (chunk_id, doc_id, text) 清单——增量嵌入的工作集。"""
        return self.con.execute(
            "SELECT chunk_id, doc_id, text FROM chunks WHERE chunk_id NOT IN "
            "(SELECT chunk_id FROM vecs)").fetchall()

    def set_manifest(self, path: str, doc_id: str, fingerprint: str,
                     ingested_at: str = "") -> None:
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO manifest VALUES (?,?,?,?)",
                (path, doc_id, fingerprint, ingested_at))

    def manifest_doc_of(self, path: str) -> str | None:
        row = self.con.execute(
            "SELECT doc_id FROM manifest WHERE path=?", (path,)).fetchone()
        return row[0] if row else None

    def analyze(self) -> None:
        """刷新查询计划器统计（sqlite_stat1）。全量/大增量摄取后调用一次，
        FTS JOIN 与 doc_id 过滤的计划选择受益。幂等，重复调用只是重算。"""
        with self.con:
            self.con.execute("ANALYZE")

    # ── 一次性迁移：为存量 chunks 补建 fts（幂等：先清空两表全量重建） ──

    def rebuild_fts(self, batch: int = 2000, progress=None) -> int:
        con = self.con
        with con:
            con.execute("DELETE FROM fts_w")
            con.execute("DELETE FROM fts_c")
        total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        done = 0
        cur = con.execute("SELECT rowid, text FROM chunks ORDER BY rowid")
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                break
            with con:
                con.executemany(
                    "INSERT INTO fts_w(rowid, content_w) VALUES (?,?)",
                    [(rid, " ".join(word_tokens(t))) for rid, t in rows])
                con.executemany(
                    "INSERT INTO fts_c(rowid, content_c) VALUES (?,?)",
                    [(rid, char_text(t)) for rid, t in rows])
            done += len(rows)
            if progress:
                progress(done, total)
        return total
