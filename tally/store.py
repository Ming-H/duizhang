"""Store：docs / chunks / fts / manifest 四表的唯一写入口。

收敛动机（v0.4 设计 D-3/D-4）：此前 cli 直写 SQL 与 Corpus.save 双轨并存，
更新路径漏删旧行产生过「幽灵旧块」。所有写入经此模块，事务边界与
fts rowid 对齐只在one place成立。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tally.index import char_text, word_tokens
from tally.schemas import Chunk, Doc

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
-- L1 事实台账（v0.7 Phase 3，DESIGN-v0.7 §7.4）：键寻址 (company,
-- period, metric, caliber)；append-only + supersede（永不物理删除）；
-- 部分索引只索引活行（superseded_by IS NULL）——死行留存审计轨迹。
CREATE TABLE IF NOT EXISTS verified_facts (
  id INTEGER PRIMARY KEY,
  company TEXT NOT NULL, period TEXT NOT NULL,
  metric TEXT NOT NULL, caliber TEXT NOT NULL,
  value_raw TEXT, value_std REAL, unit TEXT, yoy TEXT,
  citations TEXT NOT NULL,
  coverage REAL, verdict TEXT NOT NULL, run_id TEXT, created_at TEXT,
  source_doc_ids TEXT, doc_fps TEXT,
  superseded_by INTEGER REFERENCES verified_facts(id),
  supersede_reason TEXT);
CREATE INDEX IF NOT EXISTS vf_live ON verified_facts(company, period, metric, caliber)
  WHERE superseded_by IS NULL;
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
        fts 行不被覆盖，留下幽灵命中。**清与写同一事务**（v0.7 正确性
        批）：修前 drop_doc 先自提交，插入阶段崩溃后文档消失、manifest
        已删——检索静默缺文档直到下一次 ingest 才自愈。
        """
        con = self.con
        with con:
            self._drop_doc_tx(con, doc.doc_id)
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

    @staticmethod
    def _drop_doc_tx(con, doc_id: str) -> None:
        """四表删除语句（不开事务——由调用方的事务包住：
        drop_doc 独立删，add_doc 在写事务内清旧）。fts 与 chunks 的
        rowid 对齐在删除侧同样成立：先收 rowid，再删 fts，最后删
        chunks。"""
        rowids = [r[0] for r in con.execute(
            "SELECT rowid FROM chunks WHERE doc_id=?", (doc_id,))]
        con.executemany("DELETE FROM fts_w WHERE rowid=?", [(r,) for r in rowids])
        con.executemany("DELETE FROM fts_c WHERE rowid=?", [(r,) for r in rowids])
        con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM manifest WHERE doc_id=?", (doc_id,))
        con.execute("DELETE FROM vecs WHERE doc_id=?", (doc_id,))

    def drop_doc(self, doc_id: str) -> None:
        """四表删除（独立事务）。"""
        with self.con:
            self._drop_doc_tx(self.con, doc_id)

    # ── 向量层（P1.2 可选路：bge-small-zh @ fastembed，TALLY_VEC=off 关闭）──

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

    # ── L1 事实台账（v0.7 Phase 3）：读不豁免、写有闸，append-only ──

    def _fact_row(self, company: str, period: str, metric: str,
                  caliber: str) -> dict | None:
        """活行查询（部分索引命中）。返回 dict 或 None。"""
        row = self.con.execute(
            "SELECT id, value_raw, value_std, unit, yoy, citations, coverage,"
            " verdict, source_doc_ids, doc_fps, created_at FROM verified_facts"
            " WHERE company=? AND period=? AND metric=? AND caliber=?"
            " AND superseded_by IS NULL ORDER BY id DESC LIMIT 1",
            (company, period, metric, caliber)).fetchone()
        if not row:
            return None
        return {"id": row[0], "value_raw": row[1], "value_std": row[2],
                "unit": row[3], "yoy": row[4],
                "citations": json.loads(row[5] or "[]"),
                "coverage": row[6], "verdict": row[7],
                "source_doc_ids": json.loads(row[8] or "[]"),
                "doc_fps": json.loads(row[9] or "{}"),
                "created_at": row[10]}

    def current_fingerprints(self, doc_ids: list[str]) -> dict[str, str]:
        """doc_id → 当前 manifest 指纹（台账读时校验语料是否被重摄取）。"""
        if not doc_ids:
            return {}
        marks = ",".join("?" * len(doc_ids))
        return {d: fp for d, fp in self.con.execute(
            f"SELECT doc_id, fingerprint FROM manifest WHERE doc_id IN ({marks})",
            doc_ids)}

    def probe_fact(self, company: str, period: str, metric: str,
                   caliber: str) -> tuple[str, dict | None]:
        """探针：返回 (状态, 行)。状态 ∈ hit / stale / miss。
        stale = 源文档被重摄取（指纹漂移）或已不在库——读时失效
        （公理 6：失效=语料指纹），活行就地 supersede（reason=reingest），
        视同 miss。"""
        row = self._fact_row(company, period, metric, caliber)
        if row is None:
            return "miss", None
        fps = self.current_fingerprints(row["source_doc_ids"])
        ok = all(fps.get(d) == fp for d, fp in row["doc_fps"].items()) \
            and set(row["source_doc_ids"]) <= set(fps)
        if not ok:
            self.supersede_fact(row["id"], "reingest")
            return "stale", None
        return "hit", row

    def supersede_fact(self, fact_id: int, reason: str) -> None:
        with self.con:
            self.con.execute(
                "UPDATE verified_facts SET superseded_by=?, supersede_reason=?"
                " WHERE id=? AND superseded_by IS NULL", (fact_id, reason, fact_id))

    def write_fact(self, fact: dict) -> str:
        """合并写入（DESIGN-v0.7 §7.5）：返回算子名。
        noop        同值同指纹 → 不动（幂等重放）
        insert      无活行 → 新增
        conflict    异值且无期间推进 → 旧行 supersede(reason=conflict) 后
                    写新行——两值都有据时保守留新，旧行留痕待人工裁决
        restated    异值且新行引用了更晚的报告期文档（财报追溯调整的
                    轻量双时间语义）→ 旧行 supersede(reason=restated) 后写新行
        """
        key = (fact["company"], fact["period"], fact["metric"], fact["caliber"])
        old = self._fact_row(*key)
        if old is not None:
            same_val = (old["value_std"] is not None
                        and fact.get("value_std") is not None
                        and abs(old["value_std"] - fact["value_std"]) < 0.005)
            same_fps = old["doc_fps"] == (fact.get("doc_fps") or {})
            if same_val and same_fps:
                return "noop"
            reason = "reingest" if same_val else "conflict"
            if not same_val:
                old_max = max((self._doc_period(d) for d in old["source_doc_ids"]),
                              default="")
                new_max = max((self._doc_period(d) for d in fact["source_doc_ids"]),
                              default="")
                if new_max > old_max:
                    reason = "restated"
            self.supersede_fact(old["id"], reason)
        cols = ("company, period, metric, caliber, value_raw, value_std, unit,"
                " yoy, citations, coverage, verdict, run_id, created_at,"
                " source_doc_ids, doc_fps")
        vals = (fact["company"], fact["period"], fact["metric"], fact["caliber"],
                fact.get("value_raw"), fact.get("value_std"), fact.get("unit"),
                fact.get("yoy"), json.dumps(fact.get("citations") or [],
                                            ensure_ascii=False),
                fact.get("coverage"), fact.get("verdict", "pass"),
                fact.get("run_id", ""), fact.get("created_at", ""),
                json.dumps(fact.get("source_doc_ids") or [], ensure_ascii=False),
                json.dumps(fact.get("doc_fps") or {}, ensure_ascii=False))
        with self.con:
            self.con.execute(
                f"INSERT INTO verified_facts ({cols}) VALUES ({','.join('?' * 15)})",
                vals)
        op = "insert" if old is None else reason
        return op

    def _doc_period(self, doc_id: str) -> str:
        row = self.con.execute(
            "SELECT period FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return row[0] if row else ""

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
