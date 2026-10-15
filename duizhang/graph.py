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

import re

from langgraph.graph import END, START, StateGraph

from duizhang import answer as ans
from duizhang import llm as llmx
from duizhang.index import Corpus, word_tokens
from duizhang.ingest import expand_query
from duizhang.schemas import GraphState

MAX_RETRIES = 1
TOP_K = 8
GRADE_THRESHOLD = 1.5   # 命中块与 query 的平均 token 重叠下限

# 意图关键词（route 与 rule_understand 共用；摘要词优先，防止
# 「总结业绩预告」被「预告」劫持到 metric）
_SUMMARY_WORDS = ("总结", "摘要", "概览", "要点", "怎么样", "如何看", "怎么看", "观点")
_METRIC_WORDS = ("多少", "几个点", "占比", "增长率", "率", "每股",
                 "营收", "收入", "利润", "净利", "毛利", "现金流", "订单", "在手", "手持",
                 "亿元", "万元", "同比", "环比", "预告")
_COMPARE_WORDS = ("对比", "比较", "vs", "VS", "哪家", "谁强", "差异")


# ── 查询理解 ────────────────────────────────────────────


def rule_understand(query: str, corpus) -> dict:
    """规则查询理解（heuristic 主路径 / LLM 故障兜底）。

    相比 v0.4 的 _doc_filter 只回首个公司，这里全量命中——多公司是
    compare 分解的前提。period 复用 period_hints 的互斥匹配。
    """
    companies = list(dict.fromkeys(
        d.company for d in corpus.docs.values()
        if d.company and d.company in query))
    hints = period_hints(query)
    if len(companies) >= 2 and any(w in query for w in _COMPARE_WORDS):
        intent = "compare"
    elif any(w in query for w in _SUMMARY_WORDS):
        intent = "summary"
    elif any(w in query for w in _METRIC_WORDS):
        intent = "metric"
    else:
        intent = "other"
    return {"intent": intent, "companies": companies,
            "period": hints[0] if hints else "", "reason": "rule"}


# ── 节点实现 ────────────────────────────────────────────


def period_hints(query: str) -> list[str]:
    """query → 期间提示（period 字段后缀：H1/Q1/Q3/FY）。

    互斥匹配，从最具体的期间开始判定：「半年度」包含「年度」，
    顺序 if 会让 FY 混进半年度问题的 hints，期间硬过滤随之失效
    （2026-10-14 全量库实测：问 2026 半年度，命中 2025FY:5 / 2026H1E:2）。
    季度口语映射按财报日历：二季度→H1（半年报）、四季度→FY（年报）。
    """
    if any(w in query for w in ("半年度", "半年报", "中报", "二季度", "上半年")):
        return ["H1"]
    if "一季" in query or "Q1" in query:
        return ["Q1"]
    if "三季" in query or "Q3" in query:
        return ["Q3"]
    if "年度" in query or "年报" in query or "四季度" in query or "全年" in query:
        return ["FY"]
    return []


def make_nodes(corpus: Corpus):

    def understand(state: GraphState) -> GraphState:
        """查询理解（v0.5：LLM 是大脑的入口）。一次结构化抽取
        intent/companies/period，驱动路由与 doc_scope；heuristic 与
        LLM 故障时回退 rule_understand（双模式纪律不变）。"""
        q = state["query"]
        u: dict | None = None
        trace: list[str] = []
        if state["mode"] == "llm" and llmx.llm_enabled():
            try:
                r = llmx.chat_json(
                    "你是财报问答系统的查询理解器。从用户问题抽取 JSON：\n"
                    '{"intent":"metric|summary|compare|other",'
                    '"companies":["公司名"],"period":"H1|Q1|Q3|FY|",'
                    '"reason":"一句话"}\n'
                    "规则：companies 原样取自问题中的公司名（没有则空数组）；"
                    "period 按财报日历（一季度/一季报/Q1→Q1，半年度/中报/二季度→H1，"
                    "三季→Q3，年度/年报/四季度→FY，未提及则空串）；"
                    "intent：metric=问具体数字/指标/事实，summary=要概览/总结/分析，"
                    "compare=对比多家公司，other=闲聊或超纲。", q)
                intent = r.get("intent", "metric")
                if intent not in ("metric", "summary", "compare", "other"):
                    intent = "metric"
                raw_period = str(r.get("period", ""))
                ph = period_hints(raw_period)      # LLM 可能回「2026一季报」等口语
                u = {"intent": intent,
                     "companies": [c for c in r.get("companies", [])
                                   if isinstance(c, str)][:4],
                     "period": ph[0] if ph else "",
                     "reason": str(r.get("reason", ""))[:80]}
                trace = ["understand:llm"]
            except Exception:
                trace = ["understand:llm→fallback"]
        else:
            trace = ["understand:rule"]
        if u is None:
            u = rule_understand(q, corpus)
        else:
            # 公司名纠偏：LLM 幻觉出语料外的公司时，用规则命中的兜底
            known = {d.company for d in corpus.docs.values() if d.company}
            if not set(u["companies"]) <= known:
                rule_hits = rule_understand(q, corpus)["companies"]
                if rule_hits:
                    u["companies"] = rule_hits
        return {**state, "understanding": u, "trace": trace}

    def route(state: GraphState) -> GraphState:
        """纯函数分发（v0.5）：消费 understand 的 intent——理解已在
        understand 节点完成（LLM 或规则），这里只做分发与期间守卫，
        不再调 LLM/关键词。"""
        q = state["query"]
        u = state.get("understanding") or rule_understand(q, corpus)
        intent = u.get("intent", "metric")
        # 期间守卫：query 里的年份若库中全无对应报告，先对账再回答——拒。
        q_years = {y for y in re.findall(r"(20\d{2})", q)}
        have_years = {d.period[:4] for d in corpus.docs.values() if d.period[:4].isdigit()}
        if q_years and not (q_years & have_years):
            periods = "、".join(sorted({d.period for d in corpus.docs.values()}))
            return {**state, "answer": f"库里没有 {('/'.join(sorted(q_years)))} 年期间的报告"
                    f"（现有期间：{periods}）。对不上账，不硬答。",
                    "citations": [], "verdict": "refused",
                    "route": "other", "route_reason": "期间不在库",
                    "trace": ["route:period-guard→refuse"]}
        return {**state, "route": intent,
                "route_reason": (u.get("reason") or intent)[:80],
                "trace": [f"route:{intent}"]}

    def retrieve(state: GraphState) -> GraphState:
        """检索：doc_scope 由 understanding 的公司+期间构建——LLM 理解
        直接驱动检索限定（v0.4 的字符串子串猜测退役，仅规则兜底时等价）。"""
        q = state["effective_query"]
        eff = expand_query(q)
        u = state.get("understanding") or rule_understand(state["query"], corpus)
        companies = u.get("companies") or []
        period = u.get("period") or ""
        filt = companies[0] if len(companies) == 1 else ""
        raw = corpus.search(eff, k=TOP_K, doc_filter=filt)
        hits = raw
        # 期间硬过滤（文档级）：该公司有对应期间文档时，其它期间全是噪声
        # ——海量年报块会淹没目标期间（命中级后过滤会把目标块挤出 top-8，
        # exact 落空回退未过滤结果 = 拿年报答一季报，v0.4 实测），限定
        # doc 集合重查一次。
        if period:
            allowed = {d for d, doc in corpus.docs.items()
                       if (not companies or doc.company in companies)
                       and period in doc.period}
            if allowed:
                scoped = corpus.search(eff, k=TOP_K, doc_filter=filt, doc_scope=allowed)
                if scoped:
                    hits = scoped
        # 该公司该期间没有对应文档（allowed 为空或 scoped 空）→ 用未过滤结果
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
        fallback: list[str] = []
        if state["mode"] == "llm" and llmx.llm_enabled():
            try:
                r = llmx.chat_json(
                    "你是检索质量评审。根据 query 与命中片段判断是否足够回答。返回 JSON："
                    '{"sufficient":true/false,"reason":"一句话"}',
                    f"query: {state['query']}\n\n片段：\n"
                    + "\n---\n".join(h.chunk.text[:200] for h in state["hits"][:3]))
                sufficient = bool(r.get("sufficient", sufficient))
            except Exception:
                fallback = ["grade:llm→fallback"]
        return {**state, "relevance": round(score, 2), "sufficient": sufficient,
                "trace": fallback + [f"grade:{score:.1f} {'ok' if sufficient else 'insufficient'}"]}

    def rewrite_query(state: GraphState) -> GraphState:
        base = state["query"]
        fallback: list[str] = []
        if state["mode"] == "llm" and llmx.llm_enabled():
            try:
                r = llmx.chat_json(
                    "改写检索 query 使其更贴近财报术语，保留原意。返回 JSON："
                    '{"query":"改写后的 query"}', base)
                base = r.get("query", base) or base
            except Exception:
                fallback = ["rewrite:llm→fallback"]
                base = expand_query(base) + " 预计 营业收入 净利润"
        else:
            base = f"{expand_query(base)} 预计 营业收入 净利润 同比"
        return {**state, "effective_query": base, "rewritten": True,
                "retries": state.get("retries", 0) + 1,
                "trace": fallback + ["rewrite"]}

    def decompose(state: GraphState) -> GraphState:
        """compare 多跳分解（v0.5）：按公司拆子查询、各自公司+期间
        doc_scope 检索、交错合并——两家证据都在场且不被单方淹没，
        后续 grade/generate/verify 走既有路径（对账闸门不豁免）。"""
        u = state.get("understanding") or rule_understand(state["query"], corpus)
        companies = [c for c in (u.get("companies") or []) if c] or [""]
        period = u.get("period") or ""
        per: dict[str, list] = {}
        k_each = max(4, TOP_K // len(companies))
        for comp in companies:
            scope = {d for d, doc in corpus.docs.items()
                     if (not comp or doc.company == comp)
                     and (not period or period in doc.period)}
            sub = f"{comp} {state['query']}" if comp else state["query"]
            hits = corpus.search(expand_query(sub), k=k_each,
                                 doc_filter=comp, doc_scope=scope or None)
            if hits:
                per[comp] = hits
        # 交错合并：公司间轮流取——单方大文档不能挤掉另一方的证据
        merged: list = []
        for r in range(max((len(v) for v in per.values()), default=0)):
            for hits in per.values():
                if r < len(hits):
                    merged.append(hits[r])
        return {**state, "hits": merged[:TOP_K],
                "trace": [f"decompose:{'|'.join(per)} → {len(merged)} hits"]}

    def generate(state: GraphState) -> GraphState:
        hits = state["hits"]
        fallback: list[str] = []
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
            except Exception:  # LLM 故障 → 走抽取式，绝不空手而归（留痕）
                fallback = ["generate:llm→fallback"]
        body, cited = ans.extractive_answer(state, hits)
        return {**state, "answer": body, "citations": cited,
                "trace": fallback + ["generate:extractive"]}

    # 摘要守卫（双条件或）：RRF 分 ≥0.028 = 词字双轨一致命中；或强域词重叠
    # ≥3。仅分数阈值在 FTS5 分数分布下单轨真匹配会被误杀（实测 0.0164），
    # 仅重叠则垃圾词偶合可过——两者互补。

    def summarize_doc(state: GraphState) -> GraphState:
        u = state.get("understanding") or rule_understand(state["query"], corpus)
        companies = u.get("companies") or []
        filt = companies[0] if len(companies) == 1 else ""
        target = None
        q_toks = set(word_tokens(state["query"])) - {"怎么样", "怎么看", "如何", "观点", "总结", "摘要"}
        # 域词守卫：遍历前6命中，找到「分数达标 且 ≥2 词元重叠」的第一个；
        # 第一名可能是无关键词的偶然块（字级 trigram 撞出来的分数）。
        for h in corpus.search(expand_query(state["query"]), k=6, doc_filter=filt):
            overlap = len(q_toks & set(word_tokens(h.chunk.text[:300])))
            if (h.score >= 0.028 and overlap >= 2) or overlap >= 3:
                target = h.chunk.doc_id
                break
        if target is None:
            return refuse(state)
        doc_chunks = corpus.chunks_of(target, limit=12)
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
                + "\n".join(f"· {line}" for line in key_lines))
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
            return refuse({**state, "trace": ["verify:empty→refuse"]})
        # degraded：LLM 答案对不上账 → 降级为抽取式（永远有真凭实据）
        body, cited = ans.extractive_answer(state, state["hits"])
        if body:
            return {**state, "answer": body, "citations": cited, "verdict": "degraded",
                    "trace": [f"verify:degraded({cov})→extractive"]}
        return refuse({**state, "trace": [f"verify:degraded({cov})→refuse"]})

    return (understand, route, retrieve, grade_docs, rewrite_query, generate,
            decompose, summarize_doc, refuse, verify_citations)


# ── 条件边 ──────────────────────────────────────────────


def build_graph(corpus: Corpus):
    understand, route, retrieve, grade, rewrite, generate, decompose, summarize, refuse, verify = make_nodes(corpus)

    g = StateGraph(GraphState)

    g.add_node("understand", understand)
    g.add_node("route", route)
    g.add_node("retrieve", retrieve)
    g.add_node("decompose", decompose)
    g.add_node("grade_docs", grade)
    g.add_node("rewrite_query", rewrite)
    g.add_node("generate", generate)
    g.add_node("summarize_doc", summarize)
    g.add_node("refuse", refuse)
    g.add_node("verify_citations", verify)

    g.add_edge(START, "understand")
    g.add_edge("understand", "route")

    def after_route(state: GraphState) -> str:
        r = state.get("route", "metric")
        return {"metric": "retrieve", "summary": "summarize_doc",
                "compare": "decompose", "other": "refuse"}[r]

    g.add_conditional_edges("route", after_route,
                            {"retrieve": "retrieve",
                             "summarize_doc": "summarize_doc",
                             "decompose": "decompose",
                             "refuse": "refuse"})

    g.add_edge("retrieve", "grade_docs")
    g.add_edge("decompose", "grade_docs")    # 多跳合并后走同一条对账路径

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


def ask(corpus: Corpus, query: str, mode: str = "auto") -> GraphState:
    """一条命令问答：建图 → invoke。"""
    app = build_graph(corpus)
    from duizhang.schemas import new_state
    return app.invoke(new_state(query, mode))
