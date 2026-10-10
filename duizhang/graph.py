"""LangGraph 图：对账的编排心脏。

            ┌─────────┐
            │  route  │ 指标/摘要/其它
            └────┬────┘
      ┌──────────┼──────────────┐
   metric     summary         other
      │          │              │
      ▼          ▼              ▼
  retrieve   summarize_doc    refuse
      │                        （无据/闲聊，礼貌拒答）
      ▼
  grade_docs ──insufficient & retries<1──▶ rewrite_query ─┐
      │ sufficient                          （改写后重检索）│
      ▼                                     ◀─────────────┘
  generate（heuristic 抽取式 / llm grounded 生成）
      │
      ▼
  verify_citations ──fail──▶ extractive 降级 ─┐
      │ pass                                  │
      ▼                                       ▼
     END                                     END

设计要点（为什么是这张图）：
· Corrective-RAG 纠错环：grade→rewrite→retrieve 的有界重试（≤1），
  用条件边实现；无界重试是状态机的反模式。
· 每个节点窄接口：只读需要的字段、只写负责的字段（见 GraphState 注释）。
· heuristic 与 llm 双模式走同一张图：把「换模型」从架构变更
  降级为节点实现的替换。
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from duizhang import answer as ans
from duizhang import llm as llmx
from duizhang.index import Corpus
from duizhang.ingest import expand_query
from duizhang.schemas import GraphState, Retrieved

MAX_RETRIES = 1
TOP_K = 8
GRADE_THRESHOLD = 1.5   # 命中块与 query 的平均 token 重叠下限


# ── 节点实现 ────────────────────────────────────────────


def make_nodes(corpus: Corpus):
    from duizhang.index import word_tokens

    def route(state: GraphState) -> GraphState:
        q = state["query"]
        if state["mode"] == "llm" and llmx.llm_enabled():
            try:
                r = llmx.chat_json(
                    "你是财报问答系统的路由器。分类 query 并返回 JSON："
                    '{"route":"metric|summary|other","reason":"一句话"}。'
                    "metric=问具体数字/指标/事实；summary=要概览或分析；other=闲聊或超纲。",
                    q)
                return {**state, "route": r.get("route", "metric"),
                        "route_reason": r.get("reason", "")[:80], "trace": ["route:llm"]}
            except Exception:
                pass  # LLM 失败静默降级到规则
        # 摘要意图优先判定：否则「总结业绩预告」会被「预告」劫持到 metric
        if any(w in q for w in ("总结", "摘要", "概览", "要点", "怎么样", "如何看", "怎么看", "观点")):
            return {**state, "route": "summary", "route_reason": "命中摘要词",
                    "trace": ["route:summary"]}
        # 期间守卫：query 里的年份若库中全无对应报告，先对账再回答——拒。
        import re as _re
        q_years = {y for y in _re.findall(r"(20\d{2})", q)}
        have_years = {d.period[:4] for d in corpus.docs.values() if d.period[:4].isdigit()}
        if q_years and not (q_years & have_years):
            periods = "、".join(sorted({d.period for d in corpus.docs.values()}))
            return {**state, "answer": f"库里没有 {('/'.join(sorted(q_years)))} 年期间的报告"
                    f"（现有期间：{periods}）。对不上账，不硬答。",
                    "citations": [], "verdict": "refused",
                    "route": "other", "route_reason": "期间不在库",
                    "trace": ["route:period-guard→refuse"]}
        if any(w in q for w in ("多少", "几个点", "占比", "增长率", "率", "每股",
                                "营收", "收入", "利润", "净利", "毛利", "现金流", "订单", "在手", "手持",
                                "亿元", "万元", "同比", "环比", "预告")):
            return {**state, "route": "metric", "route_reason": "命中指标词",
                    "trace": ["route:metric"]}
        return {**state, "route": "other", "route_reason": "未命中意图词",
                "trace": ["route:other"]}

    def _doc_filter(query: str) -> str:
        for doc_id in corpus.docs:
            company = corpus.docs[doc_id].company
            if company and company in query:
                return company
        return ""

    def _period_hints(query: str) -> list[str]:
        hints = []
        if "半年度" in query or "半年报" in query or "中报" in query:
            hints += ["H1"]
        if "第一季" in query or "一季报" in query or "Q1" in query:
            hints += ["Q1"]
        if "第三季" in query or "三季报" in query:
            hints += ["Q3"]
        if "年度" in query or "年报" in query:
            hints += ["FY"]
        return hints

    def retrieve(state: GraphState) -> GraphState:
        q = state["effective_query"]
        eff = expand_query(q)
        hints = _period_hints(state["query"])
        filt = _doc_filter(q)
        hits = corpus.search(eff, k=TOP_K, doc_filter=filt)
        # 期间硬过滤：问「半年度」且该公司有半年度文档时，年报/季报全是噪声——
        # 检索质量（海量年报块）会淹没 1.5× 加权，过滤比加权诚实。
        if hints:
            exact = [h for h in hits
                     if any(ht in corpus.docs[h.chunk.doc_id].period for ht in hints)]
            if exact:
                hits = exact[:TOP_K]
        if not hits and hints and filt:
            # 该公司该期间没有：放宽数字，但明确标注
            hits = corpus.search(eff, k=TOP_K, doc_filter=filt)
        return {**state, "hits": hits, "trace": [f"retrieve:{len(hits)} hits"]}

    def grade_docs(state: GraphState) -> GraphState:
        if not state.get("hits"):
            return {**state, "sufficient": False, "relevance": 0.0,
                    "trace": ["grade:0 hits"]}
        q = set(word_tokens(state["effective_query"]))
        rel = []
        for h in state["hits"][:4]:
            ht = set(word_tokens(f"{h.chunk.text} {' '.join(h.chunk.num_notes)}"))
            rel.append(len(q & ht))
        score = max(rel) if rel else 0.0
        sufficient = score >= GRADE_THRESHOLD
        if state["mode"] == "llm" and llmx.llm_enabled():
            try:
                r = llmx.chat_json(
                    "你是检索质量评审。根据 query 与命中片段判断是否足够回答。返回 JSON："
                    '{"sufficient":true/false,"reason":"一句话"}',
                    f"query: {state['query']}\n\n片段：\n"
                    + "\n---\n".join(h.chunk.text[:200] for h in state["hits"][:3]))
                sufficient = bool(r.get("sufficient", sufficient))
            except Exception:
                pass
        return {**state, "relevance": round(score, 2), "sufficient": sufficient,
                "trace": [f"grade:{score:.1f} {'ok' if sufficient else 'insufficient'}"]}

    def rewrite_query(state: GraphState) -> GraphState:
        base = state["query"]
        if state["mode"] == "llm" and llmx.llm_enabled():
            try:
                r = llmx.chat_json(
                    "改写检索 query 使其更贴近财报术语，保留原意。返回 JSON："
                    '{"query":"改写后的 query"}', base)
                base = r.get("query", base) or base
            except Exception:
                base = expand_query(base) + " 预计 营业收入 净利润"
        else:
            base = f"{expand_query(base)} 预计 营业收入 净利润 同比"
        return {**state, "effective_query": base, "rewritten": True,
                "retries": state.get("retries", 0) + 1,
                "trace": ["rewrite"]}

    def generate(state: GraphState) -> GraphState:
        hits = state["hits"]
        if state["mode"] == "llm" and llmx.llm_enabled():
            evidence = "\n\n".join(
                f"[{h.chunk.doc_id} 第{h.chunk.page}页] {h.chunk.text[:400]}"
                for h in hits[:4])
            try:
                out = llmx.chat_text(
                    "你是财报问答助手。只依据给定证据回答；每个事实句末尾必须"
                    "标注来源，格式如 [公司_报告期 第N页]。证据不足则回答"
                    "「原文中没有找到依据」。禁止编造数字。",
                    f"问题：{state['query']}\n\n证据：\n{evidence}")
                if "没有找到依据" not in out:
                    return {**state, "answer": out, "citations":
                            [h.chunk.citation for h in hits[:4]],
                            "trace": ["generate:llm"]}
                return {**state, "answer": "", "verdict": "refused",
                        "trace": ["generate:llm-no-evidence"]}
            except Exception:  # LLM 故障 → 走抽取式，绝不空手而归
                pass
        body, cited = ans.extractive_answer(state, hits)
        return {**state, "answer": body, "citations": cited,
                "trace": ["generate:extractive"]}

    MIN_SUMMARY_SCORE = 0.028  # 单路弱匹配 <0.026（60万块上垃圾词也能凑出），双路真匹配 >0.03

    def summarize_doc(state: GraphState) -> GraphState:
        filt = _doc_filter(state["query"])
        target = None
        from duizhang.index import word_tokens as _wt
        q_toks = set(_wt(state["query"])) - {"怎么样", "怎么看", "如何", "观点", "总结", "摘要"}
        # 域词守卫：遍历前6命中，找到「分数达标 且 ≥2 词元重叠」的第一个；
        # 第一名可能是无关键词的偶然块（字级 bigram 撞出来的分数）。
        for h in corpus.search(expand_query(state["query"]), k=6, doc_filter=filt):
            overlap = len(q_toks & set(_wt(h.chunk.text[:300])))
            if h.score >= MIN_SUMMARY_SCORE and overlap >= 2:
                target = h.chunk.doc_id
                break
        if target is None:
            return refuse(state)
        doc_chunks = [c for c in corpus.chunks if c.doc_id == target][:12]
        head = corpus.docs[target]
        key_lines = []
        for c in doc_chunks:
            for s in ans._sentences(c.text):
                if any(w in s for w in ("营业收入", "净利润", "同比", "增长", "下降", "预计")):
                    key_lines.append(f"{s} {c.citation}")
                if len(key_lines) >= 5:
                    break
            if len(key_lines) >= 5:
                break
        body = (f"{head.company}（{head.period} {head.kind}）要点：\n"
                + "\n".join(f"· {l}" for l in key_lines))
        return {**state, "answer": body, "citations": [target], "verdict": "pass",
                "trace": [f"summarize:{target}"]}

    def refuse(state: GraphState) -> GraphState:
        return {**state,
                "answer": "原文中没有找到依据。换个问法，或先 ingest 对应报告。",
                "citations": [], "verdict": "refused", "trace": ["refuse"]}

    def verify_citations(state: GraphState) -> GraphState:
        a = state.get("answer", "")
        verdict, cov = ans.verify(a, state.get("hits", []))
        if verdict == "pass":
            return {**state, "verdict": "pass", "trace": [f"verify:pass({cov})"]}
        if verdict == "refused":
            return refuse({**state, "trace": [f"verify:empty→refuse"]})
        # degraded：LLM 答案对不上账 → 降级为抽取式（永远有真凭实据）
        body, cited = ans.extractive_answer(state, state["hits"])
        if body:
            return {**state, "answer": body, "citations": cited, "verdict": "degraded",
                    "trace": [f"verify:degraded({cov})→extractive"]}
        return refuse({**state, "trace": [f"verify:degraded({cov})→refuse"]})

    return route, retrieve, grade_docs, rewrite_query, generate, summarize_doc, refuse, verify_citations


# ── 条件边 ──────────────────────────────────────────────


def build_graph(corpus: Corpus):
    route, retrieve, grade, rewrite, generate, summarize, refuse, verify = make_nodes(corpus)

    g = StateGraph(GraphState)

    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_docs", grade)
    g.add_node("rewrite_query", rewrite)
    g.add_node("generate", generate)
    g.add_node("summarize_doc", summarize)
    g.add_node("refuse", refuse)
    g.add_node("verify_citations", verify)

    g.add_edge(START, "route")

    def after_route(state: GraphState) -> str:
        r = state.get("route", "metric")
        return {"metric": "retrieve", "summary": "summarize_doc",
                "other": "refuse"}[r]

    g.add_conditional_edges("route", after_route,
                            {"retrieve": "retrieve",
                             "summarize_doc": "summarize_doc",
                             "refuse": "refuse"})

    g.add_edge("retrieve", "grade_docs")

    def after_grade(state: GraphState) -> str:
        if state.get("sufficient"):
            return "generate"
        if state.get("retries", 0) < MAX_RETRIES:
            return "rewrite_query"
        return "refuse"

    g.add_conditional_edges("grade_docs", after_grade,
                            {"generate": "generate",
                             "rewrite_query": "rewrite_query",
                             "refuse": "refuse"})

    g.add_edge("rewrite_query", "retrieve")     # 纠错环：改写后重检索
    g.add_edge("generate", "verify_citations")
    g.add_edge("summarize_doc", END)
    g.add_edge("refuse", END)

    g.add_edge("verify_citations", END)

    return g.compile()


def ask(corpus: Corpus, query: str, mode: str = "heuristic") -> GraphState:
    """一条命令问答：建图 → invoke。"""
    app = build_graph(corpus)
    from duizhang.schemas import new_state
    return app.invoke(new_state(query, mode))
