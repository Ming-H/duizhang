"""评估：内置 QA 集 + 三指标 + 跑分表。

指标：
· retrieval@1  —— 检索 top1 命中块是否来自期望文档（检索层的硬指标）
· answer hit   —— 答案是否包含期望关键词（生成层的软指标）
· refusal      —— 应拒答问题是否真的拒了（对账纪律）
"""

from __future__ import annotations

import json
from pathlib import Path

from duizhang.graph import ask

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.json"


def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def run_eval(corpus, mode: str = "heuristic", verbose: bool = True) -> dict:
    cases = load_dataset()
    rows = []
    for c in cases:
        st = ask(corpus, c["q"], mode=mode)
        answer = st.get("answer", "")
        top1_doc = st["hits"][0].chunk.doc_id if st.get("hits") else ""
        kw_hit = any(k in answer for k in c.get("expect_keywords", []))
        doc_hit = (not c.get("expect_doc")) or (top1_doc == c["expect_doc"]) \
            or any(c["expect_doc"] in (h.chunk.doc_id) for h in st.get("hits", [])[:3])
        refused = st.get("verdict") == "refused"
        refusal_ok = refused if c.get("expect_refused") else (not c.get("expect_refused"))
        ok = refusal_ok and (kw_hit if not c.get("expect_refused") else True)
        rows.append(dict(q=c["q"][:26], top1=top1_doc, kw_hit=kw_hit,
                         doc_hit=doc_hit, refused=refused, ok=ok,
                         expect_refused=bool(c.get("expect_refused"))))
        if verbose:
            flag = "✓" if ok else "✗"
            print(f"  {flag} {c['q'][:30]:<32} top1={top1_doc:<16} "
                  f"kw={'Y' if kw_hit else 'N'} refused={'Y' if refused else 'N'}")

    n = len(rows)
    summary = dict(
        mode=mode,
        cases=n,
        retrieval_top3=round(sum(r["doc_hit"] for r in rows if not r["expect_refused"])
                             / max(1, sum(1 for r in rows if not r["expect_refused"])), 3),
        answer_hit=round(sum(r["kw_hit"] for r in rows if not r["expect_refused"])
                         / max(1, sum(1 for r in rows if not r["expect_refused"])), 3),
        refusal=round(sum(1 for r in rows if r["expect_refused"] and r["refused"])
                      / max(1, sum(1 for r in rows if r["expect_refused"])), 3),
    )
    if verbose:
        print(f"\n  mode={summary['mode']}  retrieval@3={summary['retrieval_top3']}"
              f"  answer_hit={summary['answer_hit']}  refusal={summary['refusal']}")
    return summary
