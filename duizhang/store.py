"""Store：docs / chunks / fts / manifest 四表的唯一写入口。

收敛动机（v0.4 设计 D-3/D-4）：此前 cli 直写 SQL 与 Corpus.save 双轨并存，
更新路径漏删旧行产生过「幽灵旧块」。所有写入经此模块，事务边界与
fts rowid 对齐只在one place成立。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from duizhang.index import word_tokens, char_text
from duizhang.schemas import Chunk, Doc

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
  doc_id TEXT PRIMARY KEY, company TEXT, period TEXT, kind TEXT,
  source TEXT, n_pages INT);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT, page INT, seq INT,
  text TEXT, is_table INT, num_notes TEXT);
CREATE TABLE IF NOT EXISTS manifest (
  path TEXT PRIMARY KEY, doc_id TEXT, fingerprint TEXT, ingested_at TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_w USING fts5(content_w, tokenize='unicode61');
CREATE VIRTUAL TABLE IF NOT EXISTS fts_c USING fts5(content_c, tokenize='trigram');
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self.con = sqlite3.connect(self.db_path)
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
        """四表同事务写入；fts rowid 对齐 chunks.rowid（lastrowid）。"""
        con = self.con
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

    def set_manifest(self, path: str, doc_id: str, fingerprint: str,
                     ingested_at: str = "") -> None:
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO manifest VALUES (?,?,?,?)",
                (path, doc_id, fingerprint, ingested_at))

    def drop_manifest_by_doc(self, doc_id: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM manifest WHERE doc_id=?", (doc_id,))

    def drop_manifest_by_path(self, path: str) -> None:
        with self.con:
            self.con.execute("DELETE FROM manifest WHERE path=?", (path,))

    def manifest_doc_of(self, path: str) -> str | None:
        row = self.con.execute(
            "SELECT doc_id FROM manifest WHERE path=?", (path,)).fetchone()
        return row[0] if row else None

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
