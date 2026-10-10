"""CLI：ingest / ask / eval / docs。

用法：
  python -m duizhang.cli ingest samples/            # 摄取目录（pdf/md）
  python -m duizhang.cli ask "欧科亿 2026 上半年预计净利润多少"
  DZ_LLM_API_KEY=... python -m duizhang.cli ask "..." --mode llm
  python -m duizhang.cli eval                       # 内置评估集
  python -m duizhang.cli docs                       # 已入库文档
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import os as _os
DB = Path(_os.environ.get("DZ_DB", "duizhang.db"))
IDX = Path(_os.environ.get("DZ_IDX", "duizhang.idx.pkl"))
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _corpus_from_db() -> "Corpus":
    from duizhang.index import Corpus
    if not DB.exists():
        sys.exit("未找到 duizhang.db，先运行: python -m duizhang.cli ingest samples/")
    return Corpus.load(DB, IDX)


def cmd_ingest(args):
    """增量摄取：manifest 指纹 diff → 增/改/删；--rebuild 全量重建；--watch 轮询。"""
    import sqlite3
    import time as _time
    from datetime import datetime
    from duizhang.index import Corpus
    from duizhang.ingest import ingest_file
    from duizhang import source

    def one_pass() -> None:
        if getattr(args, "rebuild", False):
            for f in (DB, IDX):
                Path(f).unlink(missing_ok=True)
        src = Path(args.path)
        corp = Corpus.load(DB, IDX) if DB.exists() else Corpus()
        con = sqlite3.connect(DB)
        con.execute("CREATE TABLE IF NOT EXISTS manifest "
                    "(path TEXT PRIMARY KEY, doc_id TEXT, fingerprint TEXT, ingested_at TEXT)")
        # 增量路径也可能面对全新 DB：docs/chunks 表必须在此建（此前只建 manifest，
        # 全新库 INSERT 全部静默失败，最终打出「无变化」的假象）
        con.execute("CREATE TABLE IF NOT EXISTS docs "
                    "(doc_id TEXT PRIMARY KEY, company TEXT, period TEXT, kind TEXT,"
                    " source TEXT, n_pages INT)")
        con.execute("CREATE TABLE IF NOT EXISTS chunks "
                    "(chunk_id TEXT PRIMARY KEY, doc_id TEXT, page INT, seq INT,"
                    " text TEXT, is_table INT, num_notes TEXT)")
        con.commit()

        scanned = source.scan(src)
        manifest = source.load_manifest(DB)
        # 引导：旧库无 manifest 时按 docs.source 一次性建账，避免全量重灌
        if not manifest and corp.docs:
            for d in corp.docs.values():
                if d.source in scanned:
                    con.execute("INSERT OR REPLACE INTO manifest VALUES (?,?,?,?)",
                                (d.source, d.doc_id, scanned[d.source], "bootstrap"))
            con.commit()
            manifest = source.load_manifest(DB)
        to_add, to_update, to_delete = source.diff(manifest, scanned)

        # 孤儿清理：docs.source 不在扫描范围（目录被排除/文件移动）→ 与 diff 删除同等处理。
        # 没有这步，旧库中被排除目录的文档会永久残留。
        dropped: set[str] = set()
        in_scope = set(scanned) | set(to_add)
        orphan_docs = [doc_id for doc_id, d in corp.docs.items()
                       if d.source not in in_scope]
        for doc_id in orphan_docs:
            con.execute("DELETE FROM manifest WHERE doc_id=?", (doc_id,))
        dropped.update(orphan_docs)

        # 删除：磁盘上已不存在（含新入排除目录的）→ 三表清除
        for path in to_delete:
            row = con.execute("SELECT doc_id FROM manifest WHERE path=?", (path,)).fetchone()
            if row and row[0]:
                dropped.add(row[0])
            con.execute("DELETE FROM manifest WHERE path=?", (path,))
        for doc_id in dropped:
            corp.docs.pop(doc_id, None)
            con.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))
            con.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        if dropped:
            corp.chunks = [c for c in corp.chunks if c.doc_id not in dropped]

        # 新增 + 更新（更新 = 先清该 doc 旧块）
        failed, n_new, n_upd = [], 0, 0
        for path in to_add + to_update:
            try:
                doc, chunks = ingest_file(path)
                if doc.doc_id in corp.docs:          # 撞 doc_id：按更新处理
                    corp.chunks = [c for c in corp.chunks if c.doc_id != doc.doc_id]
                corp.docs[doc.doc_id] = doc
                corp.chunks.extend(chunks)
                for c in chunks:
                    con.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?)",
                                (c.chunk_id, c.doc_id, c.page, c.seq, c.text,
                                 int(c.is_table), __import__("json").dumps(
                                     c.num_notes, ensure_ascii=False)))
                con.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?)",
                            (doc.doc_id, doc.company, doc.period, doc.kind,
                             doc.source, doc.n_pages))
                con.execute("INSERT OR REPLACE INTO manifest VALUES (?,?,?,?)",
                            (path, doc.doc_id, scanned[path],
                             datetime.now().isoformat(timespec="seconds")))
                n_upd += 1 if path in to_update else 0
                n_new += 1 if path in to_add else 0
            except Exception as e:
                failed.append((path, str(e)[:60]))
        con.commit()
        con.close()

        if failed:
            print(f"  ⚠ 失败 {len(failed)}：")
            for p_, e in failed[:3]:
                print(f"  ✗ {p_}: {e}")
        # 零变更：不动索引（否则每次幂等运行都白付 117s 重建 + 1GB pickle）
        if not (n_new or n_upd or to_delete or dropped):
            print(f"= 无变化  {len(corp.docs)} docs / {len(corp.chunks)} chunks")
            return
        corp._by_doc = {}
        for i, c in enumerate(corp.chunks):
            corp._by_doc.setdefault(c.doc_id, []).append(i)
        corp.finalize()
        corp.save_index(IDX)
        if getattr(args, "rebuild", False):
            corp.save(DB)
        print(f"+{n_new} 新增  ~{n_upd} 更新  -{len(to_delete)} 删除  "
              f"→ {len(corp.docs)} docs / {len(corp.chunks)} chunks")

    if getattr(args, "watch", False):
        print(f"watch 模式：每 60s 增量同步 {args.path}（Ctrl-C 退出）")
        while True:
            one_pass()
            _time.sleep(60)
    one_pass()


def cmd_ask(args):
    from duizhang.graph import ask
    corp = _corpus_from_db()
    st = ask(corp, args.query, mode=args.mode)
    print(f"\nQ: {args.query}")
    print(f"[route={st.get('route')} | relevance={st.get('relevance')} "
          f"| verdict={st.get('verdict')}]")
    print(f"A: {st.get('answer')}")
    if st.get("citations"):
        print(f"引用: {', '.join(st['citations'])}")
    print(f"trace: {' → '.join(st.get('trace', []))}")


def cmd_docs(args):
    corp = _corpus_from_db()
    for d in corp.docs.values():
        n = len(corp._by_doc.get(d.doc_id, []))
        print(f"{d.doc_id:<26} {d.kind:<8} {d.n_pages:>3}页  {n:>4}块  {d.source}")


def cmd_eval(args):
    from duizhang.evalx import run_eval
    corp = _corpus_from_db()
    run_eval(corp, mode=args.mode)


def main():
    ap = argparse.ArgumentParser(prog="duizhang", description="对账 — 带证据链的财报 RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="增量摄取（指纹 diff，自适应新增/修改/删除）")
    p.add_argument("path", nargs="?", default=str(SAMPLES))
    p.add_argument("--rebuild", action="store_true", help="全量重建（先清库）")
    p.add_argument("--watch", action="store_true", help="轮询模式：每 60s 增量同步")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="提问")
    p.add_argument("query")
    p.add_argument("--mode", default="heuristic", choices=["heuristic", "llm"])
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("docs", help="列出已入库文档")
    p.set_defaults(fn=cmd_docs)

    p = sub.add_parser("eval", help="跑内置评估集")
    p.add_argument("--mode", default="heuristic", choices=["heuristic", "llm"])
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
