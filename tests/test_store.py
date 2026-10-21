"""Store 写入层与 FTS 对齐的回归测试（幽灵块事故的永久防线）。"""

from pathlib import Path

import pytest

from duizhang.index import Corpus
from duizhang.ingest import ingest_file
from duizhang.store import Store

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
PDF = SAMPLES / "欧科亿_2026年半年度业绩预告.pdf"


@pytest.fixture()
def db(tmp_path):
    store = Store(tmp_path / "t.db")
    yield store
    store.close()


def _add(store, path=PDF):
    store.add_doc(*ingest_file(path))


def test_add_and_counts(db):
    _add(db)
    n_docs, n_chunks = db.counts()
    assert n_docs == 1 and n_chunks > 0


def test_search_hits_ingested_content(db):
    _add(db)
    corp = Corpus.load(db.db_path)
    hits = corp.search("预计 净利润", k=4, doc_filter="欧科亿")
    assert hits and "净利润" in hits[0].chunk.text


def test_drop_removes_fts_rows(db):
    """幽灵块回归：drop 后 fts 与 chunks 必须同删，搜旧内容零命中。"""
    _add(db)
    db.drop_doc("欧科亿_2026H1E")
    assert db.counts() == (0, 0)
    corp = Corpus.load(db.db_path)
    assert corp.search("预计 净利润", k=4) == []


def test_readd_after_drop_no_ghost(db):
    """更新路径：drop → re-add，块数一致且检索只命中新内容。"""
    _add(db)
    n1 = db.counts()[1]
    db.drop_doc("欧科亿_2026H1E")
    _add(db)
    assert db.counts()[1] == n1
    corp = Corpus.load(db.db_path)
    assert len({h.chunk.chunk_id for h in
                corp.search("净利润", k=8, doc_filter="欧科亿")}) <= n1


def test_add_doc_failure_is_atomic(db):
    """add_doc 原子性回归：drop+insert 必须同一事务——插入阶段抛错时
    旧文档完整保留。修前 drop_doc 先自提交，插入崩溃后文档消失：
    manifest 已删、检索静默缺文档，直到下一次 ingest 才自愈（--watch
    ≤60s，一次性用户可能永远不自愈——DB 与磁盘不一致）。"""
    _add(db)
    n_before = db.counts()
    doc, chunks = ingest_file(PDF)
    bad = chunks[0]
    bad.page = object()          # sqlite3 无法适配的类型 → INSERT 阶段抛错
    with pytest.raises(Exception):
        db.add_doc(doc, [bad])
    assert db.counts() == n_before            # 旧文档原封不动（含 manifest）
    corp = Corpus.load(db.db_path)
    assert corp.search("净利润", k=3, doc_filter="欧科亿")   # 检索仍命中


def test_rebuild_fts_idempotent(db):
    _add(db)
    n = db.rebuild_fts()
    n2 = db.rebuild_fts()
    assert n == n2
    corp = Corpus.load(db.db_path)
    assert corp.search("净利润", k=3, doc_filter="欧科亿")


def test_schema_has_doc_id_index(db):
    """chunks(doc_id) 是 drop_doc/chunks_of 的过滤键：无索引时全表扫
    60.7 万行（实测 31.8ms），索引后 <0.1ms；建索引 0.16s / ~15MB。
    存量库由 SCHEMA 的 IF NOT EXISTS 在下次 Store 打开时自动补建。"""
    names = {r[0] for r in db.con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='chunks'")}
    assert "idx_chunks_doc" in names


def test_analyze_populates_stats(db):
    """analyze 后查询计划器有统计（sqlite_stat1），全量摄取后应调用。"""
    _add(db)
    db.analyze()
    n = db.con.execute(
        "SELECT COUNT(*) FROM sqlite_stat1 WHERE tbl='chunks'").fetchone()[0]
    assert n >= 1


def test_vecs_lifecycle_model_stamp_and_ghost_cleanup(db):
    """P1.2 向量层（Store 仍是唯一写入口）：vecs 随 add_vecs 写入；
    drop_doc 级联清向量（幽灵向量防线，同 fts/chunks 对齐纪律）；
    换嵌入模型 → 旧向量全量作废（模型绑定 vecmodel 表）。"""
    _add(db)
    cid = next(iter(db.con.execute(
        "SELECT chunk_id FROM chunks LIMIT 1").fetchall()))[0]
    assert db.vec_model() == ""
    db.set_vec_model("bge-test")
    db.add_vecs([(cid, "欧科亿_2026H1E", b"\x01" * 16)])
    assert db.vec_count() == 1
    # 待嵌入清单：未入库向量的 chunks（chunk_id, doc_id, text）
    missing = {row[0] for row in db.vec_missing()}
    assert cid not in missing and len(missing) >= 1
    # 换模型 → 向量作废
    db.set_vec_model("another-model")
    assert db.vec_model() == "another-model"
    assert db.vec_count() == 0
    # 幽灵向量：drop_doc 后向量随文档消失
    db.set_vec_model("bge-test")
    db.add_vecs([(cid, "欧科亿_2026H1E", b"\x01" * 16)])
    db.drop_doc("欧科亿_2026H1E")
    assert db.vec_count() == 0
