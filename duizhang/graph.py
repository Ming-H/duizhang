"""LangGraph 图：对账的编排心脏。

            ┌─────────┐
            │understand│ LLM 结构化理解（意图/公司/期间），规则兜底
            └────┬────┘
            ┌────▼────┐
            │  route  │ 纯函数分发 + 期间守卫
            └────┬────┘
      ┌──────────┼────────────────┐
   metric/compare│             summary/other
   （llm）       │（heuristic）      │
      ▼          ▼                  ▼
 agent_research  retrieve/decompose summarize_doc / refuse
 （工具循环，模型 ──▶ grade_docs ─┐
  自主检索，步数≤6）  ─insufficient & retries<1──▶ rewrite_query ─┐
      │             │ sufficient                   （改写后重检索）│
      │             ▼                              ◀─────────────┘
      └──────▶ generate（llm grounded / heuristic 抽取式）
                 │
                 ▼
      verify_citations ──fail──▶ 抽取式降级/拒答 ─┐
                 │ pass                            │
                 ▼                                 ▼
                END                               END

设计要点（为什么是这张图）：
· v0.6 真 Agent 环：llm 模式下 metric/compare 的检索交给模型（单一
  search 工具 + 循环，模型决定查什么/何时收手）；heuristic 固定管线
  （含 grade→rewrite 有界纠错环 ≤1）保留为零 key 兜底与回归基线。
· 每个节点窄接口：只读需要的字段、只写负责的字段（见 GraphState 注释）。
· 两条路径殊途同归于 generate→verify_citations：自主检索不豁免对账闸门。
"""

from __future__ import annotations

import json
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
MAX_AGENT_STEPS = 6     # Agent 环步数硬上限（防失控；业界惯例）

# 唯一工具：检索（对比类问题模型自己逐家调）——刻意不建工具注册表，
# 一个 search 已覆盖 metric/compare 的真实需要（防过度设计）。
_SEARCH_TOOL = {"type": "function", "function": {
    "name": "search",
    "description": "在财报语料中检索证据块，返回 [{doc,page,text}] 列表。"
                   "对比多家公司时逐家调用（company 参数）；期间用 H1/Q1/Q3/FY。",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "检索词（指标术语，如：营业收入 同比）"},
        "company": {"type": "string", "description": "公司名，可选"},
        "period": {"type": "string", "description": "期间 H1/Q1/Q3/FY，可选"}},
        "required": ["query"]}}}

# 意图关键词（route 与 rule_understand 共用；摘要词优先，防止
# 「总结业绩预告」被「预告」劫持到 metric）
_SUMMARY_WORDS = ("总结", "摘要", "概览", "要点", "怎么样", "如何看", "怎么看", "观点")
_METRIC_WORDS = ("多少", "几个点", "占比", "增长率", "率", "每股",
                 "营收", "收入", "利润", "净利", "毛利", "现金流", "订单", "在手", "手持",
                 "亿元", "万元", "同比", "环比", "预告")
_COMPARE_WORDS = ("对比", "比较", "vs", "VS", "哪家", "谁强", "差异")
# 跨期趋势：财报分析最高频的真实问法（「最近几个季度…趋势」）；
# 投资建议类问题也归 trend——用数据回答，不拒答不瞎答
_TREND_WORDS = ("趋势", "走势", "最近几", "近几", "几个季度", "几个报告期",
                "逐季", "逐年", "季度营收", "季度利润")
_PERIOD_MONTH = {"Q1": 3, "H1": 6, "H1E": 7, "Q3": 9, "Q3E": 10, "FY": 12}
_re_digits = re.compile(r"\d")


def _period_key(period: str) -> tuple[int, int]:
    """报告期 → (年, 月) 时间序键。字典序会把 H1 排在 Q1 前
    （H<Q），时间上是错的（Q1=3月，H1=6月）——后缀必须映射月份。"""
    year = int(period[:4]) if period[:4].isdigit() else 0
    suffix = period[4:]
    return (year, _PERIOD_MONTH.get(suffix, 0))


def query_years(query: str) -> set[str]:
    """query 中出现的年份（20xx）。doc_scope 的年份维度：期间过滤此前
    只有类型（Q1/H1/FY）没有年份——三年语料实测问 2025Q1 时 2024Q1/
    2026Q1 的块混排进答案。"""
    return {y for y in re.findall(r"(20\d{2})", query)}


def page_evidence(hits, corpus, top_n: int = 4, window: int = 2,
                  cap: int = 20) -> list:
    """页级证据组装（small-to-big 最小形态，DESIGN-v0.6 §5.3）：
    top 命中去重页 → 同页块按 seq → 命中块为中心的 ±2 窗口 → 总量
    硬顶。半句的后续与同页表格行由此进入生成上下文（top-8 每页最多
    1 块，兄弟块从不自动在场——实测证伪「相邻块共召回」）。"""
    out, seen_pages = [], set()
    chunks_page = getattr(corpus, "chunks_page", None)
    for h in hits[:top_n]:
        key = (h.chunk.doc_id, h.chunk.page)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        if chunks_page is None:          # 语料对象无页查询能力 → 命中块自身
            out.append(h.chunk)
            continue
        siblings = chunks_page(*key)
        idx = next((i for i, c in enumerate(siblings)
                    if c.chunk_id == h.chunk.chunk_id), 0)
        for c in siblings[max(0, idx - window): idx + window + 1]:
            if len(out) >= cap:
                return out
            out.append(c)
    return out


def trend_skeleton(hits, corpus=None) -> str:
    """趋势答案的确定性骨架：每期营收句+净利句 + 该期引用，按时间
    升序。LLM 直生成多期长答案实测自相矛盾（0.8b 编造报告期/修正
    推断螺旋）——引用必须 100% 真实，骨架交给代码。候选池 = 该期
    全部命中块 + 首命中所在页的全页块（检索常被节标题块占位，钱在
    同页兄弟块里——实测 2026H1「主要会计数据」标题块挤掉营收行）。"""
    import re as _re
    by_doc: dict = {}
    for h in hits:
        ch = h.chunk if hasattr(h, "chunk") else h
        by_doc.setdefault(ch.doc_id, []).append(ch)
    lines = []
    for doc_id, chs in by_doc.items():            # 插入序 = 时间升序
        pool = list(chs)
        get_page = getattr(corpus, "chunks_page", None) if corpus else None
        if get_page and chs[0].page:
            pool += [c for c in get_page(doc_id, chs[0].page)
                     if c.chunk_id not in {x.chunk_id for x in pool}]
        def pick(require: str) -> str:
            best_s, best_sc = "", -99
            for c in pool:
                for sent in ans._sentences(c.text):
                    if not _re.search(r"\d", sent) or require not in sent:
                        continue
                    score = 2
                    if any(w in sent for w in ("同比", "增减", "较上年", "%")):
                        score += 2
                    if any(w in sent for w in ("终止经营", "少数股东", "按所有权",
                                                "净亏损以", "综合收益")):
                        score -= 5               # 利润表表头/权益变动残渣
                    if score > best_sc:
                        best_s, best_sc = sent, score
            return best_s if best_sc > 0 else ""

        picked = [p for p in (pick("营业收入"), pick("净利润")) if p]
        if picked:
            # 折叠相邻重复数字：主要会计数据表带「调整前/调整后」双列，
            # 无追溯调整时两列相同、原文重复（2025H1 实测）——折叠不改事实
            picked = [_re.sub(r"(\d[\d,.]*%?)\s+\1(?=\s|$)", r"\1", p)
                      for p in picked]
            lines.append(f"· {doc_id}：" +
                         "；".join(picked) + f" {pool[0].citation}")
    if not lines:
        return ""
    return ("（各期数字依次为：本期、上年同期、同比增减；含追溯调整列时"
            "数值相同已折叠）\n" + "\n".join(lines))


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
    elif any(w in query for w in _TREND_WORDS):
        intent = "trend"
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
    if any(w in query for w in ("半年度", "半年报", "中报", "二季度", "上半年", "H1")):
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
                    "intent：metric=问具体数字/指标/事实/原因/归因（需要检索证据句），"
                    "trend=跨期趋势/多期对比（最近几个季度/近几年的指标变化；"
                    "问投资建议也归 trend，用数据回答），"
                    "summary=要整体概览/总结全篇，compare=对比多家公司，"
                    "other=闲聊或超纲。注意「XX变动的原因」是 metric 不是 summary，"
                    "「最近几个季度的趋势」是 trend 不是 summary。", q)
                intent = r.get("intent", "metric")
                if intent not in ("metric", "summary", "compare", "trend", "other"):
                    # 宽松归一：常思模型偶尔回中文意图（glm-5.3 实测
                    # 「对比营收」）——按语义归类，不能一律掉到 metric
                    raw_intent = str(intent)
                    if (len(r.get("companies", [])) >= 2
                            and any(w in raw_intent for w in ("对比", "比较", "compare", "vs", "VS"))):
                        intent = "compare"
                    elif any(w in raw_intent for w in ("总结", "摘要", "概览")):
                        intent = "summary"
                    else:
                        intent = "metric"   # 「变动的原因」等归因题走 metric（检索证据句）
                raw_period = str(r.get("period", ""))
                if raw_period in ("H1", "Q1", "Q3", "FY"):
                    period = raw_period
                else:                     # LLM 可能回「2026一季报」等口语
                    ph = period_hints(raw_period)
                    period = ph[0] if ph else ""
                u = {"intent": intent,
                     "companies": [c for c in r.get("companies", [])
                                   if isinstance(c, str)][:4],
                     "period": period,
                     "reason": str(r.get("reason", ""))[:80]}
                trace = ["understand:llm"]
            except Exception:
                trace = ["understand:llm→fallback"]
        else:
            trace = ["understand:rule"]
        if u is None:
            u = rule_understand(q, corpus)
        else:
            # 意图守卫：小模型会把「哪家赚得多」判成合法枚举 metric——
            # 公司≥2 且 query 含对比词时强制 compare（多跳分解的前提）
            if (u.get("intent") == "metric" and len(u.get("companies") or []) >= 2
                    and any(w in q for w in _COMPARE_WORDS)):
                u["intent"] = "compare"
            # 趋势守卫：0.8b 实测把趋势问句判成 summary（单文档残渣）
            if u.get("intent") == "summary" and any(w in q for w in _TREND_WORDS):
                u["intent"] = "trend"
            # 公司名纠偏：LLM 幻觉出语料外的公司时，用规则命中的兜底
            known = {d.company for d in corpus.docs.values() if d.company}
            rule_u = rule_understand(q, corpus)
            if not set(u["companies"]) <= known:
                if rule_u["companies"]:
                    u["companies"] = rule_u["companies"]
            # period 覆盖：query 有明确期间词时规则是确定性子串匹配，
            # LLM 抽取实测会稳定抽错（「第一季度」→H1，v0.6.2 提示词
            # 加长后劣化）——规则非空即覆盖；query 无期间词时规则为空，
            # 保留 LLM 对隐式期间的判断
            if rule_u["period"]:
                u["period"] = rule_u["period"]
            # trend 必要条件：query 含趋势词（0.8b 会把「预计…多少」
            # 判成 trend 并在 reason 里幻觉数字，实测）
            if u.get("intent") == "trend" and not any(w in q for w in _TREND_WORDS):
                u["intent"] = "metric"
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
            years = query_years(state["query"])
            allowed = {d for d, doc in corpus.docs.items()
                       if (not companies or doc.company in companies)
                       and period in doc.period
                       and (not years or doc.period[:4] in years)}
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
        # v0.6 起 llm 模式经 agent_research 直连 generate，本节点仅
        # heuristic 固定管线可达——检索质量的 LLM 评审由 Agent 环的
        # 「自主决定何时收手」承担，此处的 LLM 分支已删（死代码清理）
        return {**state, "relevance": round(score, 2), "sufficient": sufficient,
                "trace": [f"grade:{score:.1f} {'ok' if sufficient else 'insufficient'}"]}

    def rewrite_query(state: GraphState) -> GraphState:
        # 同 grade_docs：v0.6 后仅 heuristic 可达，改写的 LLM 分支已删
        base = f"{expand_query(state['query'])} 预计 营业收入 净利润 同比"
        return {**state, "effective_query": base, "rewritten": True,
                "retries": state.get("retries", 0) + 1,
                "trace": ["rewrite"]}

    def decompose(state: GraphState) -> GraphState:
        """compare 多跳分解（v0.5）：按公司拆子查询、各自公司+期间
        doc_scope 检索、交错合并——两家证据都在场且不被单方淹没，
        后续 grade/generate/verify 走既有路径（对账闸门不豁免）。"""
        u = state.get("understanding") or rule_understand(state["query"], corpus)
        companies = [c for c in (u.get("companies") or []) if c] or [""]
        period = u.get("period") or ""
        # 子查询只留指标词：公司+期间的限定已由 doc_scope 完成，文本里
        # 重复公司名/期间词只会让页眉块（公司名+报告标题）完美匹配，
        # 把数字内容块挤出 top-k（实测海伦哲 top-4 全是页眉）
        sub_text = state["query"]
        for c in companies:
            sub_text = sub_text.replace(c, " ")
        sub_text = re.sub(r"对比|比较|哪家|谁强|差异|vs|VS|和|与|及", " ", sub_text)
        sub_text = re.sub(r"\s+", " ", sub_text).strip() or state["query"]
        per: dict[str, list] = {}
        k_each = max(4, TOP_K // len(companies))
        years = query_years(state["query"])
        for comp in companies:
            scope = {d for d, doc in corpus.docs.items()
                     if (not comp or doc.company == comp)
                     and (not period or period in doc.period)
                     and (not years or doc.period[:4] in years)}
            hits = corpus.search(expand_query(sub_text), k=k_each,
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
        # 页级证据（small-to-big）：命中块 ±2 窗口的兄弟块一并进入
        # 生成与校验的视野——检索精度不动，上下文在生成侧补全
        page_chunks = page_evidence(hits, corpus)
        fallback: list[str] = []
        if state["mode"] == "llm" and llmx.llm_enabled():
            # 证据按页聚合（页首标注），块序拼接；数字归一化注释随块
            by_page: dict[tuple, list] = {}
            for c in page_chunks:
                by_page.setdefault((c.doc_id, c.page), []).append(c)
            parts = []
            for (doc_id, page), cs in by_page.items():
                body = "\n".join(
                    c.text[:500] + (f"（数字参考：{'；'.join(c.num_notes[:3])}）"
                                    if c.num_notes else "") for c in cs)
                parts.append(f"[{doc_id} 第{page}页]\n{body}")
            evidence = "\n\n".join(parts)
            # 趋势题：确定性骨架 + LLM 仅做归纳（0.8b 直生成多期长答案
            # 实测自相矛盾——见 trend_skeleton 注释）
            if (state.get("understanding") or {}).get("intent") == "trend":
                skeleton = trend_skeleton(hits, corpus)
                if not skeleton:
                    return {**state, "answer": "", "verdict": "refused",
                            "trace": ["generate:trend-no-facts→refuse"]}
                summary = ""
                try:
                    summary = llmx.chat_text(
                        "只依据给定的逐期财务数据，用两三句话归纳营收与"
                        "净利润的趋势方向（上升/放缓/波动）。不要出现任何"
                        "数字——包括换算后的量级。",
                        skeleton, max_tokens=512).strip()
                except Exception:
                    pass
                if _re_digits.search(summary):   # 数字禁令由代码执行
                    summary = ""
                body = skeleton
                if summary:
                    body = f"{body}\n\n趋势归纳：{summary.strip()}"
                body += "\n（以上为财报事实梳理，不构成投资建议）"
                cited = list(dict.fromkeys(
                    h.chunk.citation for h in hits if h.chunk.citation))
                # verify 证据池与骨架候选池同源（每期首命中页的全页块）：
                # 骨架数字常来自页内兄弟块，而 hits 首块可能只是节标题——
                # 引用串相同（同页）但正文没数字，文本级重叠会崩（实测 0.119）
                ev, seen_docs = [], set()
                for h in hits:
                    if h.chunk.doc_id in seen_docs:
                        continue
                    seen_docs.add(h.chunk.doc_id)
                    ev += corpus.chunks_page(h.chunk.doc_id, h.chunk.page)
                return {**state, "answer": body, "citations": cited,
                        "evidence": ev or hits, "trace": ["generate:trend"]}
            try:
                out = llmx.chat_text(
                    "你是财报问答助手。只依据给定证据回答；每个事实句末尾必须"
                    "标注来源，格式如 [公司_报告期 第N页]。证据来自 PDF 表格"
                    "抽取，数字可能分行或断裂——可把相关数字碎片组织成答案"
                    "并如实标注来源，但禁止编造不在证据里的数字。仔细区分"
                    "指标本身的数值区间与「同比增加/将增加」的变动幅度，"
                    "引用问题所问的那个。问「净利润」而未提扣非时，默认指"
                    "归属于上市公司股东的净利润（非扣非口径）。确实无任何"
                    "相关数字才回答「原文中没有找到依据」。",
                    f"问题：{state['query']}\n\n证据：\n{evidence}")
                no_evidence = any(p in out for p in (
                    "没有找到依据", "未找到依据", "未找到相关", "无法从",
                    "无法确定", "无法回答")) or len(out.strip()) < 15
                if not no_evidence:
                    cited = list(dict.fromkeys(
                        h.chunk.citation for h in hits[:8]
                        if h.chunk.citation in out))
                    return {**state, "answer": out, "citations": cited,
                            "evidence": page_chunks, "trace": ["generate:llm"]}
                # LLM 判无据 → 降级抽取式（永远有真凭实据），而非拒答——
                # 与 verify 的「降级而非失败」同一纪律
                fallback = ["generate:llm-no-evidence→extractive"]
            except Exception:  # LLM 故障 → 走抽取式，绝不空手而归（留痕）
                fallback = ["generate:llm→fallback"]
        body, cited = ans.extractive_answer(state, hits)
        return {**state, "answer": body, "citations": cited,
                "evidence": page_chunks,
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
        # 目标检索带期间+年份范围：三年语料下无范围检索会被旧年报块
        # 抢占（实测「2025 年报」选中 2024FY）
        years = query_years(state["query"])
        scope = {d for d, doc in corpus.docs.items()
                 if (not companies or doc.company in companies)
                 and (not u.get("period") or u["period"] in doc.period)
                 and (not years or doc.period[:4] in years)} or None
        hits = corpus.search(expand_query(state["query"]), k=6,
                             doc_filter=filt, doc_scope=scope)
        for h in hits:
            overlap = len(q_toks & set(word_tokens(h.chunk.text[:300])))
            if (h.score >= 0.028 and overlap >= 2) or overlap >= 3:
                target = h.chunk.doc_id
                break
        if target is None:
            return refuse(state)
        # 要点种子双检索：目标选择的命中块 + 内容词检索（「总结/年报」
        # 词元天然锚不中经营分析——种子查询用财务关键词，范围锁定目标
        # 文档）。长报告（184-236 页）的经营分析在第 10 页后，文档头部
        # 12 块是封面/董事名录（茅台 102 页实测要点为空）
        company0 = (companies or [""])[0]
        probe = expand_query(f"{company0} 主要会计数据 营业收入 净利润 同比")
        seed_hits = corpus.search(probe, k=6, doc_filter=filt,
                                  doc_scope={target})
        seed = [h.chunk for h in seed_hits][:6] \
            + [h.chunk for h in hits if h.chunk.doc_id == target][:6]
        seen_ids = set()
        seed = [c for c in seed if not (c.chunk_id in seen_ids or seen_ids.add(c.chunk_id))]
        doc_chunks = seed + [c for c in corpus.chunks_of(target, limit=12)
                             if c.chunk_id not in seen_ids]
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

    def agent_research(state: GraphState) -> GraphState:
        """真 Agent 环（v0.6）：模型在循环里自主决定检索什么、何时收手——
        单一 search 工具（公司/期间参数），步数硬上限。工具层故障或
        零证据时回退固定检索管线（不空手）。出口仍走 generate→verify，
        **自主检索不豁免对账闸门**。heuristic 模式不进此节点。"""
        pool: dict[str, object] = {}
        u = state.get("understanding") or {}

        def run_search(args: dict) -> str:
            q = str(args.get("query", "")).strip()
            # 工具默认值从 understanding 继承（廉价结构化抽取做先验，
            # 模型显式传参可覆盖）——小模型单跳题常不传 company，
            # 全库检索会被千家同类内容淹没（实测 top1 全是无关公司）
            u_comps = [c for c in (u.get("companies") or []) if c]
            comp = str(args.get("company", "") or "") or \
                (u_comps[0] if len(u_comps) == 1 else "")
            # 期间以理解为准、参数只补空白：理解是模型对原问题的结构化
            # 决定（含规则回填），工具参数是检索战术——0.8b 实测会把
            # 「半年度」幻觉成 FY，显式错误参数覆盖先验后年报块洪水
            raw_p = str(args.get("period", "") or "")
            ph = period_hints(raw_p)
            period = u.get("period") or (ph[0] if ph else "")
            if not q:
                return "[]"
            if q not in agent_queries:
                agent_queries.append(expand_query(q))   # 扩展后的词：
                # 模型常用口语词（营收/revenue），抽取式打分要能对上
                # 证据里的财报术语（营业收入）——实测裸词零重叠致空手拒答
            # 多公司 understanding 且模型没指定公司 → 逐家检索交错合并
            # （对比题的正确语义；单公司才继承为默认过滤）
            targets = [comp] if comp else (u_comps or [""])
            years = query_years(state["query"])
            scope_common = {d for d, doc in corpus.docs.items()
                            if (not period or period in doc.period)
                            and (not years or doc.period[:4] in years)} or None
            per: dict[str, list] = {}
            for c in targets:
                scope = ({d for d in scope_common
                          if corpus.docs[d].company == c}
                         if c and scope_common is not None else scope_common)
                for h in corpus.search(expand_query(q), k=8, doc_filter=c,
                                       doc_scope=scope):
                    per.setdefault(c, [])
                    if h.chunk.chunk_id not in pool:
                        pool[h.chunk.chunk_id] = h
                        per[c].append(h)
            merged = [h for i in range(8) for lst in per.values()
                      if i < len(lst) for h in [lst[i]]]
            return json.dumps([{"doc": h.chunk.doc_id, "page": h.chunk.page,
                                "text": h.chunk.text[:160]} for h in merged],
                              ensure_ascii=False)

        n_steps = n_calls = 0
        agent_queries: list[str] = []  # Agent 实际用的检索词（写进
        # effective_query：证据是靠这些词找到的，下游抽取式/verify 打分
        # 必须与之一致——原始口语 query 与证据零重叠会误拒，实测）
        seen: dict[str, str] = {}      # 幂等缓存：小模型会把同一调用原样重复 6 步（实测）
        stopped = "budget"
        try:
            msgs = [
                {"role": "system", "content":
                    "你是财报研究 agent。用 search 工具检索证据；对比类问题"
                    "逐家公司检索；证据足够回答后停止调用工具并简述结论。"
                    "每次检索换不同角度（换指标词/换公司/换期间），不重复。"
                    "对比结论必须与所引数字的大小关系一致。"},
                {"role": "user", "content": state["query"]},
            ]
            for _ in range(MAX_AGENT_STEPS):
                resp = llmx.chat_with_tools(msgs, [_SEARCH_TOOL])
                n_steps += 1
                if not resp["tool_calls"]:
                    stopped = "self"
                    break
                msgs.append({"role": "assistant", "content": resp["content"],
                             "tool_calls": [{"function": {
                                 "name": tc["name"], "arguments": tc["arguments"]}}
                                 for tc in resp["tool_calls"]]})
                fresh = False
                for tc in resp["tool_calls"]:
                    key = json.dumps(tc["arguments"], sort_keys=True,
                                     ensure_ascii=False)
                    if key in seen:      # 重复调用：回缓存结果并提示换角度
                        msgs.append({"role": "tool",
                                     "content": seen[key] + "（重复检索：结果同上。"
                                                "请换检索角度或停止调用工具。）"})
                        continue
                    fresh = True
                    n_calls += 1
                    obs = run_search(tc["arguments"]) if tc["name"] == "search" \
                        else "[]"
                    seen[key] = obs
                    msgs.append({"role": "tool", "content": obs})
                if not fresh and pool:   # 整批全是重复 = 模型没有新问题要问
                    stopped = "重复收手"
                    break
        except Exception:
            n_steps = -1
        if not pool:   # 工具层故障/模型没找到证据 → 固定管线兜底
            u = state.get("understanding") or rule_understand(state["query"], corpus)
            base = decompose(state) if u.get("intent") == "compare" else retrieve(state)
            return {**base, "trace": ["agent:fallback"] + base["trace"]}
        # 证据按文档轮转输出：单方信息密度高不能挤掉另一方的证据
        # （decompose 交错合并的同款教训，pool 插入序会让先查的一方包揽前排）
        by_doc: dict[str, list] = {}
        for h in pool.values():
            by_doc.setdefault(h.chunk.doc_id, []).append(h)
        hits_out: list = []
        r = 0
        while len(hits_out) < TOP_K and any(len(v) > r for v in by_doc.values()):
            for chunks in by_doc.values():
                if r < len(chunks) and len(hits_out) < TOP_K:
                    hits_out.append(chunks[r])
            r += 1
        eff = state.get("effective_query", state["query"])
        for q in agent_queries:
            if q not in eff:
                eff = f"{eff} {q}"
        return {**state, "hits": hits_out, "effective_query": eff,
                "trace": [f"agent:{n_steps}步/{n_calls}次search/{len(pool)}块证据/{stopped}"]}

    def trend_research(state: GraphState) -> GraphState:
        """跨期研究（v0.6.2）：遍历该公司报告期（时间序，最近 6 期）、
        每期 doc_scope 检索关键指标、按时间升序合并——确定性遍历，
        双模式共用；比让 Agent 自己逐期调工具可靠（0.8b 会漏期/死循环）。
        出口照常 generate→verify，多期不豁免闸门。"""
        u = state.get("understanding") or rule_understand(state["query"], corpus)
        companies = [c for c in (u.get("companies") or []) if c] or [""]
        quarterly = "季" in state["query"]          # 季度问法 → 季度粒度报告期
        periods = sorted(
            (d for d, doc in corpus.docs.items()
             if doc.company in companies
             and (not quarterly or any(doc.period.endswith(sfx)
                                       for sfx in ("Q1", "H1", "Q3", "H1E")))),
            key=lambda d: _period_key(corpus.docs[d].period))[-6:]
        hits: list = []
        for d in periods:                           # 时间升序：趋势叙事方向
            # 锚定「主要会计数据」节——一行式营收/净利润在那里；
            # 泛查会捞到利润表深处的碎片（实测终止经营表头残渣）
            hits += corpus.search("主要会计数据 " + expand_query("营业收入 净利润"),
                                  k=5, doc_filter=companies[0], doc_scope={d})
        return {**state, "hits": hits,
                "trace": [f"trend:{len(periods)}期"]}

    def verify_citations(state: GraphState) -> GraphState:
        a = state.get("answer", "")
        # 证据池用页级集合（同页窗口块 citation 串相同，被引池自动扩到
        # 整页窗口）——跨块半句不再误降级
        verdict, cov = ans.verify(a, state.get("evidence") or state.get("hits", []))
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
            decompose, summarize_doc, refuse, verify_citations, agent_research,
            trend_research)


# ── 条件边 ──────────────────────────────────────────────


def build_graph(corpus: Corpus):
    understand, route, retrieve, grade, rewrite, generate, decompose, summarize, refuse, verify, agent, trend = make_nodes(corpus)

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
    g.add_node("agent_research", agent)
    g.add_node("trend_research", trend)

    g.add_edge(START, "understand")
    g.add_edge("understand", "route")

    def after_route(state: GraphState) -> str:
        # 期间守卫已在 route 内写明具体原因（缺哪年、现有哪些期间），
        # 直达 END——此前 other→refuse 会用通用文案覆盖掉它
        if state.get("verdict") == "refused":
            return END
        r = state.get("route", "metric")
        # v0.6：llm 模式下 metric/compare 进 Agent 环（模型自主检索）；
        # heuristic 模式走固定管线（兜底 + 回归基线）
        if state.get("mode") == "llm" and r in ("metric", "compare"):
            return "agent_research"
        if r == "trend":
            return "trend_research"
        return {"metric": "retrieve", "summary": "summarize_doc",
                "compare": "decompose", "other": "refuse"}[r]

    g.add_conditional_edges("route", after_route,
                            {"retrieve": "retrieve",
                             "summarize_doc": "summarize_doc",
                             "decompose": "decompose",
                             "agent_research": "agent_research",
                             "trend_research": "trend_research",
                             "refuse": "refuse",
                             END: END})

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
    g.add_edge("agent_research", "generate")    # Agent 环收集证据后直接生成
    g.add_edge("trend_research", "generate")    # 跨期证据汇合后同一生成路径
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
