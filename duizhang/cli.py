"""CLI：ingest / ask / eval / docs / index。

用法：
  python -m duizhang.cli ingest samples/            # 增量摄取（指纹 diff）
  python -m duizhang.cli ingest data/corpus --watch # 轮询模式
  python -m duizhang.cli index                      # 为存量 chunks 建/重建 FTS
  python -m duizhang.cli ask "欧科亿 2026 半年度预计净利润多少"
  python -m duizhang.cli eval                       # 内置评估集
  python -m duizhang.cli docs                       # 已入库文档
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

DB = Path(os.environ.get("DZ_DB", "duizhang.db"))
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _corpus_from_db():
    from duizhang.index import Corpus
    if not DB.exists():
        sys.exit("未找到 duizhang.db，先运行: python -m duizhang.cli ingest samples/")
    return Corpus.load(DB)


def cmd_ingest(args):
    """增量摄取：扫描 → manifest diff → Store 写入（纯 SQL 增量，无索引重建）。"""
    from duizhang import source
    from duizhang.ingest import ingest_file
    from duizhang.store import Store

    def one_pass() -> None:
        if getattr(args, "rebuild", False):
            Path(DB).unlink(missing_ok=True)
        store = Store(DB)
        root = Path(args.path)
        scanned = source.scan(args.path)
        manifest = store.manifest()
        sources = store.sources()

        def under_root(p: str) -> bool:
            """路径是否在本次扫描根之下。混合绝对/相对形式时保守判
            False（不删）——多根摄取下，删除只对本根负责。"""
            try:
                Path(p).relative_to(root)
                return True
            except ValueError:
                return False

        # 引导：旧库无 manifest 时按 docs.source 一次性建账
        if not manifest and sources:
            for doc_id, src in sources.items():
                if src in scanned:
                    store.set_manifest(src, doc_id, scanned[src], "bootstrap")
            manifest = store.manifest()
        to_add, to_update, to_delete = source.diff(manifest, scanned)
        # manifest 里其它根（如 samples/）的路径不该被本次扫描判死
        to_delete = [p for p in to_delete if under_root(p)]

        # 删除（本根内磁盘消失）+ 孤儿（source 曾在本根下但已不在扫描范围：
        # 目录被排除/移动）。其它根的文档不受影响。
        in_scope = set(scanned) | set(to_add)
        dropped = {store.manifest_doc_of(p) for p in to_delete} - {None}
        dropped |= {doc_id for doc_id, src in sources.items()
                    if src not in in_scope and under_root(src)}
        for doc_id in dropped:
            store.drop_doc(doc_id)

        failed, n_new, n_upd = [], 0, 0
        existing = store.doc_ids()
        for path in to_add + to_update:
            try:
                doc, chunks = ingest_file(path)
                if doc.doc_id in existing:       # 撞 doc_id：按更新处理
                    store.drop_doc(doc.doc_id)
                store.add_doc(doc, chunks)
                store.set_manifest(path, doc.doc_id, scanned[path],
                                   datetime.now().isoformat(timespec="seconds"))
                existing.add(doc.doc_id)
                n_upd += 1 if path in to_update else 0
                n_new += 1 if path in to_add else 0
            except Exception as e:
                failed.append((path, str(e)[:60]))

        if failed:
            print(f"  ⚠ 失败 {len(failed)}：")
            for p_, e in failed[:3]:
                print(f"  ✗ {p_}: {e}")

        n_docs, n_chunks = store.counts()
        if not (n_new or n_upd or to_delete or dropped):
            print(f"= 无变化  {n_docs} docs / {n_chunks} chunks")
            store.close()
            return
        store.analyze()   # 查询计划器统计：有写入才值得重算
        print(f"+{n_new} 新增  ~{n_upd} 更新  -{len(to_delete)} 删除  "
              f"→ {n_docs} docs / {n_chunks} chunks")
        store.close()

    if getattr(args, "watch", False):
        print(f"watch 模式：每 60s 增量同步 {args.path}（Ctrl-C 退出）")
        while True:
            one_pass()
            time.sleep(60)
    else:
        one_pass()


def cmd_index(args):
    """为存量 chunks（重）建 FTS 索引。v0.3 → v0.4 迁移或索引损坏时使用。"""
    from duizhang.store import Store
    store = Store(DB)

    def progress(done, total):
        if done % 100000 == 0 or done == total:
            print(f"  … {done}/{total}")

    n = store.rebuild_fts(progress=progress)
    print(f"FTS 重建完成：{n} chunks")


def cmd_embed(args):
    """为存量 chunks 嵌入向量（可选第三路 RRF；fastembed + bge-small-zh）。
    增量：只嵌没有向量的块；换模型自动作废旧向量。"""
    from duizhang import embed
    if not embed.installed():
        sys.exit("向量路不可用：pip install -r requirements-vec.txt")
    from duizhang.store import Store
    store = Store(DB)
    if store.vec_model() != embed.MODEL:
        store.set_vec_model(embed.MODEL)   # 换模型 → 旧向量作废
    todo = store.vec_missing()
    n_docs, _ = store.counts()
    print(f"待嵌入 {len(todo)} chunks")
    done = 0
    B = 64
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        blobs = embed.embed_blobs([t for _, _, t in batch])
        store.add_vecs([(cid, doc, b) for (cid, doc, _), b in zip(batch, blobs)])
        done += len(batch)
        if done % 1024 < B or done == len(todo):
            print(f"  … {done}/{len(todo)}")
    print(f"向量嵌入完成：{store.vec_count()} chunks（模型 {embed.MODEL}，"
          f"{n_docs} docs）")
    store.close()


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
    n_docs, n_chunks = corp.counts()
    for d in corp.docs.values():
        print(f"{d.doc_id:<26} {d.kind:<8} {d.n_pages:>3}页  {d.source}")
    print(f"\n共 {n_docs} docs / {n_chunks} chunks")


def cmd_eval(args):
    from duizhang.evalx import run_eval
    corp = _corpus_from_db()
    run_eval(corp, mode=args.mode, dataset=args.dataset)


def main():
    ap = argparse.ArgumentParser(prog="duizhang", description="对账 — 带证据链的财报 RAG")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest", help="增量摄取（指纹 diff，自适应新增/修改/删除）")
    p.add_argument("path", nargs="?", default=str(SAMPLES))
    p.add_argument("--rebuild", action="store_true", help="全量重建（先清库）")
    p.add_argument("--watch", action="store_true", help="轮询模式：每 60s 增量同步")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("index", help="为存量 chunks（重）建 FTS 索引")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("embed", help="为存量 chunks 嵌入向量（可选第三路 RRF，"
                                     "需 pip install -r requirements-vec.txt）")
    p.set_defaults(fn=cmd_embed)

    p = sub.add_parser("ask", help="提问")
    p.add_argument("query")
    p.add_argument("--mode", default="auto", choices=["auto", "llm", "heuristic"],
                   help="auto=有 key 走 llm 无 key 走 heuristic（默认）")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("docs", help="列出已入库文档")
    p.set_defaults(fn=cmd_docs)

    p = sub.add_parser("eval", help="跑内置评估集（agent 集含多跳对比题）")
    p.add_argument("--mode", default="auto", choices=["auto", "llm", "heuristic"])
    p.add_argument("--dataset", default="default",
                   choices=["default", "agent", "innolight", "maotai"],
                   help="default=单跳 10 题（CI 回归）；agent=多跳对比 8 题；"
                        "innolight=中际旭创真值 31 题（事实/趋势/陷阱/鲁棒）；"
                        "maotai=贵州茅台真值 9 题（简称/口径/断词泛化）")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
