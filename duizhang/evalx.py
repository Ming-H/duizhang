"""评估：内置 QA 集 + 三指标 + 跑分表。

指标：
· retrieval@3 —— 检索 top3 命中块是否来自期望文档（检索层的硬指标）
· answer hit   —— 答案是否包含期望关键词（生成层的软指标）
· refusal      —— 应拒答问题是否真的拒了（对账纪律）

逐题 ✓ 口径与汇总一致：非拒答题必须「关键词命中 且 检索命中」
（judge()），否则逐题打勾数与 retrieval@3 / answer_hit 对不上。
"""

from __future__ import annotations

import json
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.json"


def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text(encoding="utf-8"))


def judge(case: dict, st: dict) -> dict:
    """单题判定（纯函数，便于测试与自定义评估集复用）。

    检索命中的两条通道都要认：metric 路由填 hits，summary 路由不检索
    但填 citations——只看 hits 会把摘要题系统性判死。
    """
    answer = st.get("answer", "")
    hits = st.get("hits", [])
    top1_doc = hits[0].chunk.doc_id if hits else ""
    kw_hit = any(k in answer for k in case.get("expect_keywords", []))
    doc_hit = (not case.get("expect_doc")) \
        or (top1_doc == case["expect_doc"]) \
        or any(case["expect_doc"] in h.chunk.doc_id for h in hits[:3]) \
        or any(case["expect_doc"] in c for c in st.get("citations", []))
    refused = st.get("verdict") == "refused"
    refusal_ok = refused if case.get("expect_refused") else not case.get("expect_refused")
    ok = refusal_ok and (kw_hit and doc_hit if not case.get("expect_refused") else True)
    return dict(q=case["q"][:26], top1=top1_doc, kw_hit=kw_hit,
                doc_hit=doc_hit, refused=refused, ok=ok,
                expect_refused=bool(case.get("expect_refused")))


def run_eval(corpus, mode: str = "auto", verbose: bool = True) -> dict:
    from duizhang.graph import build_graph
    from duizhang.schemas import new_state

    app = build_graph(corpus)      # 图只编译一次，逐题 invoke
    rows = []
    for c in load_dataset():
        st = app.invoke(new_state(c["q"], mode))
        row = judge(c, st)
        rows.append(row)
        if verbose:
            flag = "✓" if row["ok"] else "✗"
            print(f"  {flag} {c['q'][:30]:<32} top1={row['top1']:<16} "
                  f"kw={'Y' if row['kw_hit'] else 'N'} "
                  f"doc={'Y' if row['doc_hit'] else 'N'} "
                  f"refused={'Y' if row['refused'] else 'N'}")

    summary = dict(
        mode=mode,
        cases=len(rows),
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
