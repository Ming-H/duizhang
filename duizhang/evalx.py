"""评估：多数据集 + 三指标 + 跑分表。

指标：
· retrieval@3 —— 检索 top3 命中块是否来自期望文档（检索层的硬指标；
  对比题用 expect_docs 列表：每家都要在场）
· answer hit   —— 答案是否包含期望关键词（生成层的软指标）
· refusal      —— 应拒答问题是否真的拒了（对账纪律）

v0.7 新增：
· steps_summary —— ReAct 步级观汇总（calls/new/dup/empty/waste_rate）：
  dup/empty 占比 = 步级浪费率，eₖ（步错率）的可测代理（DESIGN-v0.7
  §3 度量先行；gₖ 闸门逃生率的出口 = verify verdict 分布，已由
  faithfulness_pass 承担）
· multiturn 数据集 —— 轮间带 L0 会话记忆的连续问答（每 case 是一串
  turns，判定按轮）
· eval 默认隔离 L1 台账（DZ_MEMORY=off）——防止把检索能力测成记忆
  命中；DZ_EVAL_MEMORY=on 显式放开（记忆专项评估用）

逐题 ✓ 口径与汇总一致：非拒答题必须「关键词命中 且 检索命中」
（judge()），否则逐题打勾数与 retrieval@3 / answer_hit 对不上。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parents[1] / "evals"
DATASETS = {"default": DATASET_DIR / "dataset.json",
            "agent": DATASET_DIR / "dataset-agent.json",
            "innolight": DATASET_DIR / "dataset-innolight.json",
            "maotai": DATASET_DIR / "dataset-maotai.json",
            "multihop": DATASET_DIR / "dataset-multihop.json",
            "multiturn": DATASET_DIR / "dataset-multiturn.json",
            "quarters": DATASET_DIR / "dataset-quarters.json"}


def load_dataset(name: str = "default") -> list[dict]:
    path = DATASETS.get(name, Path(name))
    return json.loads(path.read_text(encoding="utf-8"))


def _docs_in_state(expect: str, st: dict) -> bool:
    """期望 doc 是否在检索（hits[:8]）或引用里。"""
    return (any(expect in h.chunk.doc_id for h in st.get("hits", [])[:8])
            or any(expect in c for c in st.get("citations", [])))


def judge(case: dict, st: dict) -> dict:
    """单题判定（纯函数，便于测试与自定义评估集复用）。

    检索命中的两条通道都要认：metric 路由填 hits，summary 路由不检索
    但填 citations——只看 hits 会把摘要题系统性判死。对比题
    expect_docs 列表：每个期望文档都要在场，只中一家不算过。
    """
    answer = st.get("answer", "")
    hits = st.get("hits", [])
    top1_doc = hits[0].chunk.doc_id if hits else ""
    kw_hit = any(k in answer for k in case.get("expect_keywords", []))
    expect_docs = case.get("expect_docs") or ([case["expect_doc"]] if case.get("expect_doc") else [])
    doc_hit = (not expect_docs) or all(_docs_in_state(d, st) for d in expect_docs)
    refused = st.get("verdict") == "refused"
    refusal_ok = refused if case.get("expect_refused") else not case.get("expect_refused")
    ok = refusal_ok and (kw_hit and doc_hit if not case.get("expect_refused") else True)
    return dict(q=case["q"][:26], top1=top1_doc, kw_hit=kw_hit,
                doc_hit=doc_hit, refused=refused, ok=ok,
                expect_refused=bool(case.get("expect_refused")))


def chunk_pr(case: dict, st: dict, k: int = 8) -> dict | None:
    """chunk 级 P/R（操作性定义，P1.3）。

    真值承载块 = doc ∈ expect_docs 且 text 含任一 expect_keywords
    （关键词是人工核对过的原文数字/词）。relevant 集未做全量标注
    （同一数字常在叙述/表格多处出现），故：
    · P@k = top-k 中真值承载块占比——检索噪声的下界度量；
    · R@k = top-k 内出现 ≥1 真值承载块记 1——命中率下界近似。
    拒答题/无关键词题返回 None，不计入统计。
    """
    if case.get("expect_refused") or not case.get("expect_keywords"):
        return None
    hits = st.get("hits", [])[:k]
    if not hits:
        return {"p": 0.0, "r": 0.0}
    expect_docs = case.get("expect_docs") or [case["expect_doc"]]

    def _rel(h):
        c = h.chunk if hasattr(h, "chunk") else h
        return (c.doc_id in expect_docs
                and any(kw in c.text for kw in case["expect_keywords"]))

    n_rel = sum(1 for h in hits if _rel(h))
    return {"p": round(n_rel / len(hits), 3), "r": 1.0 if n_rel else 0.0}


def summarize(rows: list[dict], states: list[dict], mode: str, dataset: str,
              prs: list[dict] | None = None) -> dict:
    """汇总三指标 + faithfulness + chunk 级 P/R + 步级浪费率（纯函数）。"""
    answered = [(r, s) for r, s in zip(rows, states)
                if not r["expect_refused"]]
    n_ans = max(1, len(answered))
    prs = prs or []
    # 无陷阱题的数据集（如 multiturn 接力链）refusal 口径不适用 →
    # None（打印 N/A、JSON null）——打 0.0 会被未来 CI 门槛误判
    n_ref = sum(1 for r in rows if r["expect_refused"])
    covs = [s.get("coverage") for _, s in answered
            if s.get("coverage") is not None]
    # ReAct 步级观测汇总（v0.7 Phase 1）：waste_rate = (dup+empty)/calls
    # ——dup=幂等缓存拦下的重复动作，empty=零观察动作。eₖ 的可测代理。
    calls = sum(len(s.get("agent_steps") or []) for _, s in answered)
    gates = [stp.get("gate") for _, s in answered
             for stp in (s.get("agent_steps") or [])
             if stp.get("gate") in ("new", "dup", "empty")]
    n_dup = gates.count("dup")
    n_empty = gates.count("empty")
    steps_summary = {"calls": calls, "new": gates.count("new"),
                     "dup": n_dup, "empty": n_empty,
                     "waste_rate": round((n_dup + n_empty) / len(gates), 3)
                     if gates else 0.0}
    return dict(
        mode=mode,
        dataset=dataset,
        cases=len(rows),
        retrieval_top3=round(sum(r["doc_hit"] for r, _ in answered) / n_ans, 3),
        answer_hit=round(sum(r["kw_hit"] for r, _ in answered) / n_ans, 3),
        refusal=(round(sum(1 for r in rows
                           if r["expect_refused"] and r["refused"]) / n_ref, 3)
                 if n_ref else None),
        faithfulness_pass=round(
            sum(1 for _, s in answered if s.get("verdict") == "pass") / n_ans, 3),
        mean_coverage=round(sum(covs) / len(covs), 3) if covs else 0.0,
        steps=steps_summary,
        **{"chunk_p@8": round(sum(p["p"] for p in prs) / len(prs), 3) if prs else 0.0,
           "chunk_r@8": round(sum(p["r"] for p in prs) / len(prs), 3) if prs else 0.0},
    )


def run_eval(corpus, mode: str = "auto", verbose: bool = True,
             dataset: str = "default") -> dict:
    from duizhang.graph import build_graph
    from duizhang.memory import append_turn, summarize_turn
    from duizhang.schemas import new_state

    # eval 默认隔离 L1 台账（v0.7 Phase 3）：DZ_EVAL_MEMORY=on 才放开
    # ——记忆命中的答案检索痕迹为空，混跑会把检索指标测虚高
    prev_env = os.environ.get("DZ_MEMORY")
    if os.environ.get("DZ_EVAL_MEMORY", "") != "on":
        os.environ["DZ_MEMORY"] = "off"

    try:
        app = build_graph(corpus)      # 图只编译一次，逐题 invoke
        rows, states, prs = [], [], []
        cases = load_dataset(dataset)
        for c in cases:
            if "turns" in c:           # multiturn：轮间带 L0 会话记忆
                history: list[dict] = []
                for t in c["turns"]:
                    st = app.invoke(new_state(t["q"], mode, history))
                    row = judge(t, st)
                    rows.append(row)
                    states.append(st)
                    pr = chunk_pr(t, st)
                    if pr:
                        prs.append(pr)
                    if verbose:
                        flag = "✓" if row["ok"] else "✗"
                        print(f"  {flag} {t['q'][:30]:<32} top1={row['top1']:<16} "
                              f"kw={'Y' if row['kw_hit'] else 'N'} "
                              f"doc={'Y' if row['doc_hit'] else 'N'} "
                              f"refused={'Y' if row['refused'] else 'N'}")
                    append_turn(history, summarize_turn(t["q"], st))
                continue
            st = app.invoke(new_state(c["q"], mode))
            row = judge(c, st)
            rows.append(row)
            states.append(st)
            pr = chunk_pr(c, st)
            if pr:
                prs.append(pr)
            if verbose:
                flag = "✓" if row["ok"] else "✗"
                print(f"  {flag} {c['q'][:30]:<32} top1={row['top1']:<16} "
                      f"kw={'Y' if row['kw_hit'] else 'N'} "
                      f"doc={'Y' if row['doc_hit'] else 'N'} "
                      f"refused={'Y' if row['refused'] else 'N'}")

        summary = summarize(rows, states, mode, dataset, prs)
        if verbose:
            print(f"\n  mode={summary['mode']} dataset={dataset}"
                  f"  retrieval@3={summary['retrieval_top3']}"
                  f"  answer_hit={summary['answer_hit']}"
                  f"  refusal={summary['refusal'] if summary['refusal'] is not None else 'N/A'}"
                  f"  faithfulness(pass/cov)={summary['faithfulness_pass']}"
                  f"/{summary['mean_coverage']}"
                  f"  chunk P/R@8={summary['chunk_p@8']}/{summary['chunk_r@8']}"
                  f"  steps={summary['steps']}")
        # rows 随行（P1 观测性）：--json 脚本化修前/修后四格表的数据面
        return {**summary, "rows": rows}
    finally:
        if prev_env is None:
            os.environ.pop("DZ_MEMORY", None)
        else:
            os.environ["DZ_MEMORY"] = prev_env
