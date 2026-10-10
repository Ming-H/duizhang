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

DB = Path("duizhang.db")
SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _corpus_from_db() -> "Corpus":
    from duizhang.index import Corpus
    if not DB.exists():
        sys.exit("未找到 duizhang.db，先运行: python -m duizhang.cli ingest samples/")
    return Corpus.load(DB)


def cmd_ingest(args):
    from duizhang.index import Corpus
    from duizhang.ingest import ingest_file
    corp = Corpus.load(DB) if DB.exists() else Corpus()
    src = Path(args.path)
    files = sorted(list(src.rglob("*.pdf")) + list(src.rglob("*.md"))) \
        if src.is_dir() else [src]
    failed = []
    for i, f in enumerate(files, 1):
        try:
            doc, chunks = ingest_file(f)
            if doc.doc_id in corp.docs:      # 同公司同期重复文件：跳过
                continue
            corp.add(doc, chunks, rebuild=False)
        except Exception as e:
            failed.append((str(f), str(e)[:60]))
        if i % 50 == 0:
            print(f"  … {i}/{len(files)}  ({len(corp.docs)} docs, {len(corp.chunks)} chunks)")
    corp.finalize()
    corp.save(DB)
    print(f"\n共 {len(corp.docs)} 文档 / {len(corp.chunks)} 块 → {DB}")
    if failed:
        print(f"失败 {len(failed)} 个文件（已跳过）：")
        for f, e in failed[:5]:
            print(f"  ✗ {f}: {e}")


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

    p = sub.add_parser("ingest", help="摄取 pdf/md 文件或目录")
    p.add_argument("path", nargs="?", default=str(SAMPLES))
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
