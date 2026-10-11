"""Store 写入层与 FTS 对齐的回归测试（幽灵块事故的永久防线）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    before = db.counts()[1]
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


def test_rebuild_fts_idempotent(db):
    _add(db)
    n = db.rebuild_fts()
    n2 = db.rebuild_fts()
    assert n == n2
    corp = Corpus.load(db.db_path)
    assert corp.search("净利润", k=3, doc_filter="欧科亿")
