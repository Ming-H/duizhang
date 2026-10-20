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
import os
import re

from langgraph.graph import END, START, StateGraph

from duizhang import answer as ans
from duizhang import llm as llmx
from duizhang import memory as mem
from duizhang.index import Corpus, word_tokens
from duizhang.ingest import expand_query
from duizhang.schemas import GraphState

MAX_RETRIES = 1
TOP_K = 8
GRADE_THRESHOLD = 1.5   # 命中块与 query 的平均 token 重叠下限
MAX_AGENT_STEPS = 6     # Agent 环步数硬上限（防失控；业界惯例）
MAX_SUBGOALS = 5        # Planner 子目标数上限（DESIGN-v0.7 §4 Phase 2）

# 工具注册表（v0.7 Phase 4，白名单=封闭世界契约第 2 条）：search（词法
# 检索）/ metrics_table（单文档主要会计数据结构化行）/ derive（差分
# 算子，Q2=H1−Q1、Q4=FY−Q3——原文不披露的单季值，带「派生值」标注）。
# 注册表之外的任何 action 一律返回 "[]"（模型无权自造工具）。
_SEARCH_TOOL = {"type": "function", "function": {
    "name": "search",
    "description": "在财报语料中检索证据块，返回 [{doc,page,text}] 列表。"
                   "对比多家公司时逐家调用（company 参数）；期间用 H1/Q1/Q3/FY。",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "检索词（指标术语，如：营业收入 同比）"},
        "company": {"type": "string", "description": "公司名，可选"},
        "period": {"type": "string", "description": "期间 H1/Q1/Q3/FY，可选"}},
        "required": ["query"]}}}
_METRICS_TOOL = {"type": "function", "function": {
    "name": "metrics_table",
    "description": "取某公司某报告期「主要会计数据」表的结构化行"
                   "（营收/归母净利，含同比与引用）。单期单指标数值优先用"
                   "本工具而非 search——直接给对齐好的行，免去从碎片拼数。",
    "parameters": {"type": "object", "properties": {
        "company": {"type": "string", "description": "公司名"},
        "period": {"type": "string", "description": "报告期，如 2025FY / 2026H1 / Q1"}},
        "required": ["company", "period"]}}}
_DERIVE_TOOL = {"type": "function", "function": {
    "name": "derive",
    "description": "差分算子：由累计值推单季值（Q2=H1−Q1，Q4=FY−Q3），"
                   "返回带「派生值」标注与双引用的结果行。财报原文不直接"
                   "披露单季 Q2/Q4 时用它，禁止自己心算。",
    "parameters": {"type": "object", "properties": {
        "company": {"type": "string", "description": "公司名"},
        "period": {"type": "string", "description": "目标单季：Q2 或 Q4"},
        "year": {"type": "string", "description": "年份，如 2025"},
        "metric": {"type": "string", "description": "营收 或 净利润，默认营收"}},
        "required": ["company", "period"]}}}
AGENT_TOOLS = [_SEARCH_TOOL, _METRICS_TOOL, _DERIVE_TOOL]

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
                "逐季", "逐年", "季度营收", "季度利润",
                # 多季度聚合问法（v0.7）：「各季度/每季度/分季度」天然跨期
                "各季度", "每季度", "分季度")
# 单季问法（v0.7 Phase 4）：第二/第四季度单靠 as-reported 答不了
# （Q2/Q4 未披露），必须走 trend 跨期遍历 + 差分——trend 语义，不是
# metric。第一/第三季度在 Q1/Q3 报告直接披露，保持 metric 单跳。
_QUARTER_Q_RE = re.compile(r"第?[一二三四1-4]季度|单季度|单季")  # 单季粒度信号
_DERIVE_Q_RE = re.compile(r"第?[二两2四4]季度|单季度|单季")      # 需差分 → trend


def _wants_trend(query: str) -> bool:
    """trend 语义判定（_TREND_WORDS ∪ 需差分的单季问法）——
    understand/route 守卫三处共用，口径必须一致。"""
    return any(w in query for w in _TREND_WORDS) or bool(_DERIVE_Q_RE.search(query))
_PERIOD_MONTH = {"Q1": 3, "Q2": 6, "H1": 6, "H1E": 7, "Q3": 9, "Q3E": 10,
                 "Q4": 12, "FY": 12}   # Q2/Q4（v0.6.7）：单季派生行的排序键，
                                      # 缺席时 key=0 排最前，Q1 环比对着 Q4 算
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
    years = {y for y in re.findall(r"(20\d{2})", query)}
    # 口语二位年份：「15年中报」= 2015——要求紧跟「年」字，金额/编号
    # 不误判（实测 15 年中报静默答成 2025）
    years |= {"20" + m for m in
              re.findall(r"(?<!\d)(1[5-9]|2\d)(?=\s*年)", query)}
    return years


def page_evidence(hits, corpus, top_n: int = 4, window: int = 2,
                  cap: int = 20) -> list:
    """页级证据组装（small-to-big 最小形态，DESIGN-v0.6 §5.3）：
    top 命中去重页 → 同页块按 seq → 命中块为中心的 ±2 窗口 → 总量
    硬顶。半句的后续与同页表格行由此进入生成上下文（top-8 每页最多
    1 块，兄弟块从不自动在场——实测证伪「相邻块共召回」）。"""
    out, seen_pages = [], set()
    chunks_page = getattr(corpus, "chunks_page", None)
    # 按页分层取样：前置（D14）把主要会计数据表页 8+ 块顶到 hits 队头，
    # 切 hits[:top_n] 会全落同一页——利润表/MD&A 页进不了 LLM 证据
    # （总收入题 小模型只见 p5 营业收入行，实测）。扫到凑满 top_n 个
    # 不同页为止，总量仍受 cap 硬顶。窗仍锚各页首个命中 ±2——同页
    # 全命中并窗实测净负（小模型上下文稀释，0.778 vs 0.852，D14 cap
    # 哲学同源：淹没比缺席更伤小模型）。
    for h in hits:
        if len(seen_pages) >= top_n or len(out) >= cap:
            break
        key = (h.chunk.doc_id, h.chunk.page)
        if key in seen_pages:
            continue
        seen_pages.add(key)
        if chunks_page is None:          # 语料对象无页查询能力 → 命中块自身
            out.append(h.chunk)
            continue
        siblings = chunks_page(*key)
        idx = next((i for i, c in enumerate(siblings)
                    if c.chunk_id == h.chunk.chunk_id), None)
        if idx is None:                  # 合成块（拼接行）不在语料：整块进上下文
            out.append(h.chunk)
            continue
        for c in siblings[max(0, idx - window): idx + window + 1]:
            if len(out) >= cap:
                return out
            out.append(c)
    return out


def stitch_adjacent(hits, corpus=None) -> list:
    """同页相邻表块拼接（D16+B2）：PDF 主要会计数据表会把「口径标签
    块」与「光数值块」切成相邻块（2025Q1 扣非：seq4 标签 / seq5 数值
    1,568,385,287.19）。光数值块零词面、检索召不回；标签块无数字、句级
    打分强降权——问扣非时被页 10 的「标签+数字齐全」归母行抢答
    （1,582,876,128.67 顶掉 1,568,385,287.19）。命中块与同页相邻块
    在页内 seq 序上互补（一侧纯标签、一侧纯数值）时拼成整行再进生成。
    完整行不受影响；无 chunks_page 能力时原样返回。"""
    from duizhang.schemas import Retrieved as _R
    get_page = getattr(corpus, "chunks_page", None) if corpus else None
    if not get_page:
        return list(hits)

    def _cjk_len(t: str) -> int:
        return len(re.findall(r"[一-鿿]", t))

    def _is_label(t: str) -> bool:      # 无数字且指标词足够长
        return not re.search(r"\d", t) and _cjk_len(t) >= 6

    def _is_bare_nums(t: str) -> bool:  # 光数值：有 3 位以上数字且零汉字
        return bool(re.search(r"\d{3}", t)) and _cjk_len(t) == 0

    out, used = [], set()
    for h in hits:
        ch = h.chunk if hasattr(h, "chunk") else h
        if ch.chunk_id in used:
            continue
        new_chunk = None
        if ch.page and (_is_label(ch.text) or _is_bare_nums(ch.text)):
            sibs = sorted(get_page(ch.doc_id, ch.page), key=lambda c: c.seq)
            idx = next((i for i, c in enumerate(sibs)
                        if c.chunk_id == ch.chunk_id), None)
            for j in ((idx - 1) if idx else None,
                      (idx + 1) if idx is not None and idx + 1 < len(sibs) else None):
                if j is None:
                    continue
                nb = sibs[j]
                if nb.chunk_id in used or nb.chunk_id == ch.chunk_id:
                    continue
                if (_is_label(ch.text) and _is_bare_nums(nb.text)) or \
                        (_is_bare_nums(ch.text) and _is_label(nb.text)):
                    label, nums = (ch, nb) if _is_label(ch.text) else (nb, ch)
                    from duizhang.schemas import Chunk as _C
                    new_chunk = _C(
                        chunk_id=f"{label.chunk_id}~{nums.chunk_id}",
                        doc_id=label.doc_id, page=label.page, seq=label.seq,
                        text=f"{label.text} {nums.text}",
                        num_notes=nums.num_notes)
                    used.add(nb.chunk_id)
                    break
        target = new_chunk or ch
        out.append(_R(chunk=target, score=getattr(h, "score", 0.0),
                      rank=getattr(h, "rank", 0))
                   if hasattr(h, "chunk") else target)
        if new_chunk:
            used.add(ch.chunk_id)
    return out


_FRONT_CAP = 8   # 前置封顶：主要会计数据行集中在页首（前 8 块覆盖营收/归母/扣非/每股）


def front_key_table(hits, u, query, corpus, cap: int = _FRONT_CAP) -> list:
    """metric 意图：主要会计数据表页前置（D14 封顶版）——真值行常被
    无数字表头块挤出前排（词面匹配更强，实测 5,171,485,967 行排 #6 而
    表头占 #1#2），小模型从分季度表抓单季数当全年。前置后 LLM 证据头部
    即真值行（trend 同款哲学）。D14 两条收口：①前置块数封顶 cap——
    整页前置（长页 12+ 块）会淹没 Agent 自己收集的证据，真值行集中在
    页首；②归因题（原因/为什么/为何/变动）跳过——要叙事证据句，
    表格行前置只会挤掉归因句。"""
    if u.get("intent") != "metric" or not hits:
        return hits
    if any(w in query for w in ("原因", "为什么", "为何", "变动")):
        return hits
    comps = [c for c in (u.get("companies") or []) if c]
    comp0 = comps[0] if len(comps) == 1 else ""
    # 年份：query 现抽优先；裸追问（「它的营收呢」）不带年份时用
    # understanding 里 L0 继承的年份——否则前置探针会跨年漂移
    # （实测继承 2026 却前置了 2024H1 的表页）
    years = query_years(query) or set(u.get("years") or [])
    scope = {d for d, doc in corpus.docs.items()
             if (not comp0 or doc.company == comp0)
             and (not u.get("period") or u["period"] in doc.period)
             and (not years or doc.period[:4] in years)} or None
    probe = corpus.search("主要会计数据", k=2, doc_filter=comp0,
                          doc_scope=scope)
    from duizhang.schemas import Retrieved
    key_page_chunks: list = []
    get_page = getattr(corpus, "chunks_page", None)
    if probe and get_page:
        kdoc, kpage = probe[0].chunk.doc_id, probe[0].chunk.page
        page_chunks = get_page(kdoc, kpage)
        key_page_chunks = [
            Retrieved(chunk=c, score=0.3, rank=i)
            for i, c in enumerate(page_chunks[:cap])]
        # #6 窗口扩展：长页净资产行可落在 cap 之外（innolight Q29 类），
        # 页内与查询指标词有词面命中的表内块也纳入前置，总量封顶 cap*2
        qtoks = set(word_tokens(expand_query(query)))
        for c in page_chunks[cap:cap * 2]:
            if qtoks & set(word_tokens(c.text)):
                key_page_chunks.append(
                    Retrieved(chunk=c, score=0.3, rank=len(key_page_chunks)))
    probe_ids = {c.chunk.chunk_id for c in key_page_chunks}
    return key_page_chunks + [h for h in hits
                              if h.chunk.chunk_id not in probe_ids]


def _line_nums(text: str) -> set[str]:
    """文本中的大数集合（≥6 位含逗号分隔，如 44,464,207,646.01）。"""
    return set(re.findall(r"\d[\d,]{5,}(?:\.\d+)?", text))


def _caliber_nums(lines: list[str], pred) -> set[str]:
    """口径标签行的数字集：标签/数值分块行（D16 场景，标签行无数字）
    取标签行的下一行数字并入。"""
    out: set[str] = set()
    for i, ln in enumerate(lines):
        if pred(ln):
            got = _line_nums(ln)
            out |= got
            if not got and i + 1 < len(lines):
                out |= _line_nums(lines[i + 1])
    return out


def caliber_mismatch(query: str, answer: str, chunks) -> bool:
    """口径数值闸（D22）：问句点名口径（扣非/总收入/净资产/总资产）
    时，答案的大数必须出现在该口径标签行的数字集里——小模型实测拿归母
    44,516,880,421.86 答扣非题、营业收入 90,703,260,964.48 答总收入
    题：数字都在证据页，verify 引用级照样 pass（gate≠correctness，
    §9 边界的第一处收口——引用级对账之上加口径级数值对账）。证据里
    没有该口径行 → False（检索缺口不归闸门背锅，交给降级抽取式）。
    """
    if ("扣非" not in query and "扣除非" not in query and "总收入" not in query
            and "净资产" not in query and "总资产" not in query
            and "营业收入" not in query):
        return False
    ans_nums = _line_nums(answer)
    if not ans_nums:
        return False
    # 逐行修复断词残片（块级修复只看块首，多行块的后续行残片会漏）。
    # D24：chunks 混装 Chunk 与 Retrieved（调用点是页级 page_chunks ∪
    # hits，后者是 Retrieved 包装）——裸取 .text 即 AttributeError，
    # 被 generate 的宽 except 吞成 llm→fallback，闸门从未真正执行
    lines = [ans.split_fix(ln) for c in chunks
             for ln in getattr(c, "chunk", c).text.splitlines()]
    good_sets: list[set[str]] = []
    if "扣非" in query or "扣除非" in query:
        good_sets.append(_caliber_nums(
            lines, lambda ln: ("扣除" in ln or "扣非" in ln) and "净利润" in ln))
    if "净资产" in query and "总资产" not in query:
        good_sets.append(_caliber_nums(
            lines, lambda ln: "净资产" in ln and "总资产" not in ln))
    if "总资产" in query:
        good_sets.append(_caliber_nums(lines, lambda ln: "总资产" in ln))
    if "总收入" in query:
        # 段级截取：利润表头行总/子口径数字混居（「营业总收入 92,278
        # … 其中：营业收入 90,703 …」），行级数字集分不开——取总收入
        # 标签之后、子口径标签（其中/营业收入）之前的段
        seg = set()
        for ln in lines:
            if "总收入" not in ln:
                continue
            rest = ln[ln.find("总收入"):]
            for stop in ("其中", "营业收入"):
                j = rest.find(stop)
                if j > 0:
                    rest = rest[:j]
            seg |= _line_nums(rest)
        good_sets.append(seg)
    if "营业收入" in query and "总收入" not in query:
        # D23：营业收入（子口径）原不在闸名单——更大档位模型实测拿营业总收入
        # 174,144,069,958.25 答营业收入题（真值 170,899,152,276.34，
        # 茅台 2024FY p63 两行并存）。营业收入⊄营业总收入（中间隔
        # 「总」），行匹配天然分开，只需段级截取（同总收入）+ 标签/
        # 数值分块行取下一行（同 _caliber_nums）
        seg = set()
        for i, ln in enumerate(lines):
            j = ln.find("营业收入")
            if j < 0:
                continue
            rest = ln[j:]
            for stop in ("其中", "利息", "利润总额", "净利润"):
                k = rest.find(stop)
                if k > 0:
                    rest = rest[:k]
            got = _line_nums(rest)
            if not got and i + 1 < len(lines):
                got = _line_nums(lines[i + 1])
            seg |= got
        good_sets.append(seg)
    for good in good_sets:
        if good and not (ans_nums & good):
            return True
    return False


def _parse_metric_row_raw(text: str, cumulative: bool = False) -> tuple[float, str, str] | None:
    """主要会计数据行 → (本期原币值, 亿值, 同比%)。v0.7 拆出原币精度：
    derive 差分（Q2=H1−Q1）与 L1 台账（value_std）都不能用四舍五入
    的亿值做算术——误差会到百万级。
    本期取第一个非百分数、≥1e6 的大数（防「4.842 546」这类残缺数字
    误入）；同比取最后的百分比。解析不出返回 None（该指标省略）。
    D26 双列版式：Q3 主要会计数据是「本报告期 单季同比 年初至报告
    期末 累计同比」四列——大数/小数交错签名「V P V P」（同比常不带
    % 号，签名不能依赖它）。cumulative=True 取第二个大数与第二个
    同比（累计列），同一 bullet 各指标同列取数——修复前营收可取利润
    表累计、归母取本报告期单季的混装。非双列行保持旧律（首大数+末
    同比），1e3–1e6 的中间量级（万元残片）两不用。
    负号捕获：同比负增长（-1.21）的负号此前被正则吞掉，方向反转。"""
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?%?", text)
    vals: list[str] = []
    pcts: list[str] = []
    seq: list[str] = []            # V/P 交错序（双列签名判定用）
    for n in nums:
        try:
            v = float(n.rstrip("%").replace(",", ""))
        except ValueError:
            continue
        if v >= 1e6:
            vals.append(n)
            seq.append("v")
        elif abs(v) < 1000:
            pcts.append(n)
            seq.append("p")
    if not vals:
        return None
    dual = len(vals) >= 2 and len(pcts) >= 2 and seq[:4] == ["v", "p", "v", "p"]
    idx = 1 if (cumulative and dual) else 0
    raw = float(vals[idx].rstrip("%").replace(",", ""))
    yi = raw / 1e8
    # 同比取本指标三元组内的（首列同比），非整行末位——茅台「多行一块」
    # 版式把多指标拼一行，末位同比属于下一个指标（2026H1 营收被安上
    # 归母的 -1.95%，真值 +1.47）
    pct = pcts[idx] if len(pcts) > idx else (pcts[-1] if pcts else "")
    return raw, f"{yi:,.1f} 亿", (pct.rstrip("%") + "%") if pct else ""


def _parse_metric_row(text: str, cumulative: bool = False) -> tuple[str, str] | None:
    """旧签名包装（亿值, 同比%）——trend 骨架显示层与单测沿用。"""
    parsed = _parse_metric_row_raw(text, cumulative)
    return (parsed[1], parsed[2]) if parsed else None


def _pick_metric_line(pool, require: str, cumulative: bool = False) -> str | None:
    """行级选行（D21/D26）：池内所有块的所有行里，选指标词所在、
    口径匹配、可解析的最佳行，返回从指标词起截的行段。trend 骨架、
    metrics_table 工具、L1 事实抽取共用。"""
    cands: list[tuple[int, str]] = []
    for c in pool:
        for raw in c.text.splitlines():
            line = raw.strip()
            if not line:
                continue
            for frag, full in (("润（元）", "净利润"), ("入（元）", "营业收入"),
                               ("者权益（元）", "归属于上市公司股东的净资产"),
                               ("常性损益的净利润", "扣除非经常性损益的净利润"),
                               ("利润 ", "归属于上市公司股东的净利润")):
                if line.startswith(frag):
                    line = full + " " + line
                    break
            if require not in line:
                continue
            # 口径守卫（D16，D21 行级精化）：含「扣除/扣非」的是
            # 另一口径，词面同样含「净利润」，先到先得会拿扣非数
            # 当归母（实测 15.7 亿顶掉 15.8 亿）。但表块行内拼接
            # 时归母行与下一行「扣非」标签尾巴同处一行，整行含
            # 「扣除」就跳过会把归母陪葬——只看指标词前 20 字的
            # 标签上下文：归母自己的标签不含扣除，放行；扣非行
            # 的标签含扣除，照旧跳过。要求扣非口径时 require 自身
            # 含「扣非」。
            idx = line.find(require)
            ctx = line[max(0, idx - 20):idx]
            if ("扣除" in ctx or "扣非" in ctx) and "扣非" not in require:
                continue
            # 分季度横排守卫（v0.7 Phase 4）：FY 年报 §九「分季度主要
            # 财务数据」的行是四个单季值并排、无同比列——首大数是 Q1
            # 不是本报告期值（实测 457.8 亿顶掉全年 1,709 亿）。特征：
            # ≥4 个大数且 0 同比；真 as-reported 行必带同比列（本期/
            # 上年同期/增减三元组），不会误伤。宁可丢行（保守失败）
            # 不可取错行。
            if (len(re.findall(r"\d[\d,]{5,}(?:\.\d+)?", line)) >= 4
                    and "%" not in line
                    and not re.search(r"\s-?\d{1,3}(?:\.\d+)?\s", " " + line + " ")):
                continue
            if not re.search(r"\d", line):
                continue
            # D21 行级取数：表块无标点，句级切分整块一句，
            # pick(净利润) 与 pick(营业收入) 命中同一句 → 归母
            # 取到营收值（茅台 2025Q1 实测 506.0/506.0 同值、
            # verify 满分放行）。行内从指标词起截，取其后首个
            # 大数——同行多口径混排也各取各的。
            seg = line[idx:]
            score = 2
            # 带同比的行优先（D26）：Q3 双列锚定行带两个同比，
            # 利润表裸值行一个没有——同分时插入序碰运气会让裸值
            # 行胜出（营收缺同比、数字取自他页而引用标锚定页）。
            # 同比常不带 % 号，签名用「独立小数」（15.29 形态）
            if "%" in seg or re.search(r"\s-?\d{1,3}(?:\.\d+)?\s",
                                       " " + seg + " "):
                score += 2
            if any(w in line for w in ("适用", "不适用", "差异", "终止经营",
                                        "少数股东", "会计准则", "年均",
                                        "现金分红", "未分配利润")):
                score -= 6          # 段落噪声（2026H1 会计准则段实测；
                # 年均/现金分红/未分配利润＝分红与留存段引用本期值
                # 的行，2024FY 实测「三年年均 745.6 亿」顶掉归母 862.3 亿）
            cands.append((score, seg))
    # 名次回退（D21 补面）：打分最高的行解析不过时（MD&A 散文
    # 「454.03 亿元」带 % 赢表行、却过不了 ≥1e6 原币门槛），按
    # 分数降序取第一个可解析行——只取最佳会让该指标整行丢失
    # （茅台 2025H1/2026H1 归母缺失实测根因）。
    for score, seg in sorted(cands, key=lambda x: -x[0]):
        if score <= 0:
            break
        if _parse_metric_row_raw(seg, cumulative=cumulative):
            return seg
    return None


_SKELETON_REQUIRES = (("营收", "营业收入"), ("归母净利", "净利润"))


def metric_rows(pool, cumulative: bool = False,
                requires=_SKELETON_REQUIRES) -> list[dict]:
    """结构化指标行（v0.7）：[{label, key, raw(原币), yi(亿), yoy, seg}]
    ——trend 骨架 / metrics_table 工具 / derive 算子 / L1 事实抽取
    四方共用的单一解析原语。"""
    rows = []
    for label, key in requires:
        seg = _pick_metric_line(pool, key, cumulative)
        if not seg:
            continue
        parsed = _parse_metric_row_raw(seg, cumulative)
        if parsed:
            raw, yi, pct = parsed
            rows.append({"label": label, "key": key, "raw": raw,
                         "yi": yi, "yoy": pct, "seg": seg})
    return rows


def trend_rows(hits, corpus=None) -> list[dict]:
    """跨期结构化（v0.7）：每份命中文档 → {doc_id, period, tag,
    cumulative, rows, citation}。趋势骨架与 L1 事实写入的数据源。"""
    by_doc: dict = {}
    for h in hits:
        ch = h.chunk if hasattr(h, "chunk") else h
        by_doc.setdefault(ch.doc_id, []).append(ch)
    entries = []
    for doc_id, chs in by_doc.items():            # 插入序 = 时间升序
        period_full = doc_id.rsplit("_", 1)[-1]
        period_tag = period_full[-2:]
        # D26：Q3 双列版式锚累计列（年初至报告期末）——basis 标注
        # 「前三季累计」才名副其实，且营收/归母同列（行级口径一致）
        cumulative = period_tag == "Q3"
        pool = list(chs)
        # 页扩展：命中页 ±1 窗口（去重命中页 cap 3）——FY 年报常首中
        # 分季度表页而主要会计数据在邻页（2024FY 实测命中 p6/p35，
        # 主要会计数据在 p5），只扩命中页本身会把全年行漏掉/取错
        # （457.8 亿＝Q1 顶掉 1,709 亿）
        get_page = getattr(corpus, "chunks_page", None) if corpus else None
        if get_page:
            seen = {x.chunk_id for x in pool}
            for p in list(dict.fromkeys(c.page for c in chs if c.page))[:3]:
                for pp in (p - 1, p, p + 1):
                    for c in get_page(doc_id, pp):
                        if c.chunk_id not in seen:
                            pool.append(c)
                            seen.add(c.chunk_id)
        rows = metric_rows(pool, cumulative)
        if rows:
            entries.append({"doc_id": doc_id, "period": period_full,
                            "tag": period_tag, "cumulative": cumulative,
                            "rows": rows, "citation": pool[0].citation,
                            "page": pool[0].page})
    return entries


def _derive_quarters(entries: list[dict]) -> tuple[list[tuple], list[dict]]:
    """差分算子（v0.7 Phase 4）：as-reported 累计值推单季。
    Q2 = H1（半年累计）− Q1（单季=首季累计）；Q4 = FY（全年累计）−
    Q3（前三季累计）——两组操作数的口径天然对齐（都是「年初至该期末
    累计」），差分合法。原币精度计算（亿值舍入误差百万级）。
    返回 ((时间键, 行文本), ...) 列表与 derived 登记（verify 出口重算）。
    负值/缺操作数 → 跳过该行（选错行的保守失败），绝不打印错数。"""
    items: list[tuple] = []
    derived: list[dict] = []
    by_comp: dict[str, list[dict]] = {}
    for e in entries:
        by_comp.setdefault(e["doc_id"].rsplit("_", 1)[0], []).append(e)

    def find(comp: str, year: str, tag: str, label: str) -> dict | None:
        """按 (公司, 年, 后缀, 指标) 精确取行——只按 tag 匹配会把
        2024Q2 派生到 2025 的 H1/Q1 行上（三年语料实测：三个年份的
        Q2 全是同一个 387.9 亿，年份只是标签）。"""
        for e in by_comp.get(comp, []):
            if e["tag"] == tag and e["period"][:4] == year:
                for r in e["rows"]:
                    if r["label"] == label:
                        return {**r, "citation": e["citation"],
                                "period": e["period"]}
        return None

    for comp, ents in by_comp.items():
        years = sorted({e["period"][:4] for e in ents if e["period"][:4].isdigit()})
        for label in ("营收", "归母净利"):
            for year in years:
                for tgt, hi_tag, lo_tag in (("Q2", "H1", "Q1"), ("Q4", "FY", "Q3")):
                    hi = find(comp, year, hi_tag, label)
                    lo = find(comp, year, lo_tag, label)
                    if not (hi and lo and hi["raw"] and lo["raw"]):
                        continue
                    val = hi["raw"] - lo["raw"]
                    if val <= 0:
                        continue
                    line = (f"· {comp}_{year}{tgt}（单季·派生）：{label} "
                            f"{val / 1e8:,.1f} 亿（= {year}{hi_tag} {hi['yi']} − "
                            f"{year}{lo_tag} {lo['yi']}，原文未直接披露）"
                            f" {hi['citation']} {lo['citation']}")
                    items.append((_period_key(f"{year}{tgt}"), line))
                    derived.append({"label": f"{comp}_{year}{tgt}{label}",
                                    "value": val, "a": hi["raw"], "b": lo["raw"]})
    return items, derived


_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7,
           "八": 8, "九": 9}


def _quarter_window(query: str) -> dict | None:
    """问句的季度窗口语义 → {"year": "2025"|None, "n": 4}|None。
    两种形态：年份切片（「25年4个季度/2025年四个季度」→ 该日历年
    Q1–Q4）与相对窗口（「近/最近/过去 N 个季度」→ 最近 N 季）。
    作为 understand LLM 抽取的规则兜底（双脑纪律）——正则永远打地鼠，
    语义理解的主责在模型，这里只保零 key 模式可用。"""
    m = re.search(r"(20\d{2}|[12]\d)\s*年\s*的?\s*([1-9一二三四五六七八九])"
                  r"\s*个?\s*季度", query)
    if m:
        y = m.group(1)
        year = y if len(y) == 4 else "20" + y
        return {"year": year, "n": int(_CN_NUM.get(m.group(2), m.group(2)))}
    m = re.search(r"(?:近|最近|过去)\s*([1-9一二三四五六七八九]|\d+)"
                  r"\s*个?\s*季", query)
    if m:
        return {"year": None, "n": int(_CN_NUM.get(m.group(1), m.group(1)))}
    return None


def _quarter_n(query: str) -> int | None:
    """兼容旧签名：窗口存在则返回 n。"""
    w = _quarter_window(query)
    return w["n"] if w else None


def _quarter_view(entries: list[dict], n: int = 4, year: str | None = None
                  ) -> tuple[list[tuple], list[dict]]:
    """统一单季视图（v0.6.6）：把报告期口径折算成日历季度序列——
    Q1 直取（一季报即首季累计）、Q2=H1−Q1、Q3=三季报累计−H1、
    Q4=FY−三季报累计（三组差分操作数同为「年初至期末累计」，口径对齐
    合法；原币精度、负值跳过=保守失败）。每季一行（营收+归母同列），
    按时间序取最近 n 季——「近4个季度」要的是这个序列，as-reported
    混排视图（单季/半年累计/前三季累计）跨基期不可比。派生行照常登记
    derived（verify 出口重算）。"""
    items: list[tuple] = []
    derived: list[dict] = []
    by_comp: dict[str, list[dict]] = {}
    for e in entries:
        by_comp.setdefault(e["doc_id"].rsplit("_", 1)[0], []).append(e)

    def find(comp: str, year: str, tag: str, label: str) -> dict | None:
        for e in by_comp.get(comp, []):
            if e["tag"] == tag and e["period"][:4] == year:
                for r in e["rows"]:
                    if r["label"] == label:
                        return {**r, "citation": e["citation"],
                                "period": e["period"]}
        return None

    for comp, ents in by_comp.items():
        years = sorted({e["period"][:4] for e in ents
                        if e["period"][:4].isdigit()
                        and (not year or e["period"][:4] == year)})
        for y_ in years:
            for q in ("Q1", "Q2", "Q3", "Q4"):
                parts: list[str] = []
                cites: list[str] = []
                if q == "Q1":
                    e = next((x for x in ents
                              if x["tag"] == "Q1" and x["period"][:4] == y_),
                             None)
                    if not e:
                        continue
                    for r in e["rows"]:
                        parts.append(f"{r['label']} {r['yi']}"
                                     + (f"（同比 {r['yoy']}）" if r["yoy"] else ""))
                    cites.append(e["citation"])
                    mark = "单季"
                else:
                    hi_tag, lo_tag = {"Q2": ("H1", "Q1"),
                                      "Q3": ("Q3", "H1"),
                                      "Q4": ("FY", "Q3")}[q]
                    for label in ("营收", "归母净利"):
                        hi = find(comp, y_, hi_tag, label)
                        lo = find(comp, y_, lo_tag, label)
                        if not (hi and lo and hi["raw"] and lo["raw"]):
                            continue
                        val = hi["raw"] - lo["raw"]
                        if val <= 0:
                            continue
                        parts.append(
                            f"{label} {val / 1e8:,.1f} 亿"
                            f"（= {y_}{hi_tag} {hi['yi']} − {y_}{lo_tag} "
                            f"{lo['yi']}，原文未直接披露）")
                        cites += [hi["citation"], lo["citation"]]
                        derived.append({"label": f"{comp}_{y_}{q}{label}",
                                        "value": val, "a": hi["raw"],
                                        "b": lo["raw"]})
                    if not parts:
                        continue
                    mark = "单季·派生"
                items.append((_period_key(f"{y_}{q}"),
                              f"· {comp}_{y_}{q}（{mark}）：" +
                              "；".join(parts) + " " +
                              " ".join(dict.fromkeys(cites))))
    items.sort(key=lambda x: x[0])
    if not year:                      # 相对窗口取最近 n 季；年份切片天然全年
        items = items[-n:]
    # 环比（确定性算子，v0.6.7）：相邻单季的趋势量，叙述答「上升/下降」
    # 直接有据；数字进骨架即进 verify 证据池（合成块机制）
    prev: dict[str, float] = {}
    out_items = []
    for key, line in items:
        m = re.match(r"· (\S+?)_\d{4}Q\d（", line)
        comp_q = m.group(1) if m else ""
        vm = re.search(r"营收 ([\d,.]+) 亿", line)
        if comp_q and vm:
            cur = float(vm.group(1).replace(",", ""))
            p_ = prev.get(comp_q)
            if p_ and p_ > 0:
                qoq = (cur - p_) / p_ * 100
                line = line.replace("）：", f"，环比 {qoq:+.1f}%）：", 1)
            prev[comp_q] = cur
        out_items.append((key, line))
    return out_items, derived


def trend_skeleton_ex(hits, corpus=None, quarter_n: int | None = None,
                      quarter_year: str | None = None
                      ) -> tuple[str, list[dict], list[dict]]:
    """趋势数据层全量版（v0.7）：返回 (骨架文本, 事实行, 派生登记)。
    骨架文本 = as-reported 行 + 派生单季行按时间序交错；事实行进
    state['fact_rows']（L1 写入源），派生登记进 state['derived']
    （verify 出口重算 ±0.01）。quarter_n（「近N个季度」）→ 单季统一
    视图；派生值非 as-reported 事实，L1 不写（fact_rows 空）。"""
    entries = trend_rows(hits, corpus)
    if not entries:
        return "", [], []
    if quarter_n:
        q_items, q_derived = _quarter_view(entries, quarter_n, quarter_year)
        if q_items:
            return "\n".join(t for _, t in q_items), [], q_derived
    items: list[tuple] = []
    fact_rows: list[dict] = []
    for e in entries:
        parts = [f"{r['label']} {r['yi']}（同比 {r['yoy']}）" if r["yoy"]
                 else f"{r['label']} {r['yi']}" for r in e["rows"]]
        basis = {"H1": "半年累计", "Q1": "单季", "Q3": "前三季累计",
                 "FY": "全年"}.get(e["tag"], "")
        basis_note = f"（{basis}）" if basis else ""
        items.append((_period_key(e["period"]),
                      f"· {e['doc_id']}{basis_note}：" + "；".join(parts) +
                      f" {e['citation']}"))
        fact_rows.append(e)
    d_items, derived = _derive_quarters(entries)
    items += d_items
    items.sort(key=lambda x: x[0])
    return "\n".join(t for _, t in items), fact_rows, derived


def trend_skeleton(hits, corpus=None) -> str:
    """趋势数据层（确定性，旧签名）：每期解析为结构化行
    「· 期间：营收 X 亿（同比 Y）；归母净利 Z 亿（同比 W）[引用]」。
    原始表格行（本期/同期/同比混排）可读性差（用户实测「很乱」），
    且 小模型需要干净输入才能做分析——数据层干净了分析层才放得开。
    断词工件（「润（元）…」=净利润行）在此同样注入。"""
    return trend_skeleton_ex(hits, corpus)[0]


def _analysis_numbers_ok(analysis: str, rows: str) -> bool:
    """投资分析数字闸门：分析中每个数字（年份除外）必须出现在数据行里
    ——表外数字=幻觉，整个分析段丢弃（宁缺毋滥，但不再一刀切禁数字：
    引用表内数字的分析是允许且需要的）。"""
    allowed = set(re.findall(r"[\d,]*\d(?:\.\d+)?%?", rows))
    allowed |= {a.rstrip("%") for a in allowed}
    # 骨架同比带负号（-1.21%），叙述侧数字正则不捕负号 → 无符号变体
    # 也放行，否则合法的「下降 1.21%」叙述被自家闸门误杀
    allowed |= {a.lstrip("-").rstrip("%") for a in allowed}
    for n in re.findall(r"[\d,]*\d(?:\.\d+)?%?", analysis):
        if re.fullmatch(r"20\d{2}", n):
            continue
        if n not in allowed and n.rstrip("%") not in allowed:
            return False
    return True


# 方向闸（trend 双闸之二，D26 家族）：一阶方向词——二阶词（放缓/收窄/
# 改善/扭亏）需前值方向上下文，v1 不判（§9 边界账本）
_DIR_POS = ("增长", "上升", "提升", "上涨", "增加", "回升", "走高")
_DIR_NEG = ("下降", "回落", "下滑", "减少", "下跌", "暴跌", "萎缩", "走低")
_GATE_METRICS = ("营收", "归母", "净利", "利润", "收入")


def _skeleton_signs(skeleton: str) -> dict[str, list[int]]:
    """骨架行 → 各指标同比符号序列（方向闸的判定基准）。
    行式：「营收 1,207.8 亿（同比 16.95%）；归母净利 608.3 亿…」。"""
    signs: dict[str, list[int]] = {}
    for m in re.finditer(r"(营收|归母净利)\s*[\d,.]+\s*亿（同比\s*(-?[\d.]+)%）",
                         skeleton):
        key = "营收" if m.group(1) == "营收" else "归母"
        v = float(m.group(2))
        signs.setdefault(key, []).append((v > 0) - (v < 0))
    return signs


def direction_gate(narration: str, skeleton: str) -> str | None:
    """方向闸：叙述中的方向词必须与骨架同比符号相容——+90.05% 被说成
    「大幅回落」这类数字对/引用对/方向错的叙述，数字闸与 verify_citations
    都看不见（§9 gate≠correctness 的方向维度收口）。规则机械可判：
    正词要求该指标至少一期同比为正，负词要求至少一期为负；全期同号而
    叙述反向 → 违例，返回违例片段（供 trace），合规返回 None。
    指标绑定取方向词前 14 字内最靠近的指标词（中文语序「指标+方向」；
    绑错指标会把合法叙述误杀）。边界（v1 不判，§9 记账）：①二阶词
    （放缓/收窄/改善/扭亏）需前值方向上下文；②「环比/较上期」是定基
    比较，骨架只有同比数据，词前上下文含环比标记时跳过。"""
    signs = _skeleton_signs(skeleton)
    if not signs:
        return None
    for m in re.finditer("|".join(_DIR_POS + _DIR_NEG), narration):
        word = m.group(0)
        positive = word in _DIR_POS
        ctx = narration[max(0, m.start() - 14):m.start()]
        if "环比" in ctx or "较上期" in ctx:
            continue
        # 二阶（速率）方向 v1 不判（v0.6.6 收口）：「增速回落/增幅收窄」
        # 是速率事实——同比 146%→37% 时指标仍在涨，与同比正号相容；
        # 此前被当一阶词整段误杀（glm-5.3 正确方向叙述退骨架，trace:
        # direction-gate(回落…)）。裸「回落」（指标回落）仍一阶拦截
        if any(rn in ctx for rn in ("增速", "增幅", "涨幅", "增长率")):
            continue
        key, best = None, -1
        for w in _GATE_METRICS:
            i = ctx.rfind(w)
            if i > best:
                best = i
                key = "营收" if w in ("营收", "收入") else "归母"
        pool = signs.get(key) if key else [s for v in signs.values() for s in v]
        pool = pool or [s for v in signs.values() for s in v]
        ok_pos = any(s > 0 for s in pool)
        ok_neg = any(s < 0 for s in pool)
        if (positive and not ok_pos) or (not positive and not ok_neg):
            return f"{word}「…{ctx[-8:]}{word}…」"
    return None


# 裁判自测（v0.6.5）：敏感词表是裁判 veto(sensitive) 的**执行器**——
# 检测交给裁判（抓语义，如绕开词表的变体表述），执行交给词表（确定性
# 可测试）。重写敏感句=教模型换词绕词表，故只剥不重写
_SENSITIVE_WORDS = ("买入", "卖出", "目标价", "建议", "加仓", "减仓", "重仓",
                    "抄底", "逃顶", "看多", "看空", "增持", "回避")


def _strip_sensitive(narration: str) -> str:
    """剥掉含敏感词的句子；全剥空返回空串（调用方退骨架）。"""
    sents = re.split(r"(?<=[。；！？])\s*", narration)
    kept = [s for s in sents
            if s.strip() and not any(w in s for w in _SENSITIVE_WORDS)]
    return "".join(kept).strip()


def _self_check(query: str, skeleton: str, narration: str) -> dict | None:
    """裁判自测：结构化评判叙述 {faithful/on_topic/sensitive/issues}
    （RAGAS faithfulness/answer-relevancy 的本地化简化 + 敏感维度）。
    **非对称信任**：裁判只能否决/降级，永不解锁——确定性双闸先于裁判
    跑（裁判全绿 ≠ 豁免闸门），裁判说不行 → 剥句/有界修复/退骨架。
    拿不准判 true：误杀的代价是体验退回骨架（保守失败），误放的代价
    是错误出门（不可逆）。DZ_SELF_CHECK=off 关闭（延迟敏感场景）；
    裁判异常 → None 不拦叙述（裁判失灵不等于叙述有错，双闸仍在）。"""
    if os.environ.get("DZ_SELF_CHECK", "").strip().lower() == "off":
        return None
    try:
        return llmx.chat_json(
            "你是财报回答的裁判。只评判给定叙述，输出严格 JSON："
            '{"faithful": true/false, "on_topic": true/false, '
            '"sensitive": true/false, "issues": ["原因"]}。'
            "faithful=每个数字与方向（增/降）都能在给定数据行找到；"
            "on_topic=回答了所问问题；sensitive=含投资建议（买入/卖出/"
            "目标价/加减仓）或无依据的未来预测（将会/预计达到+具体幅度）。"
            "拿不准一律判 true——你只能高置信度否决。",
            f"问题：{query}\n\n数据行：\n{skeleton}\n\n叙述：\n{narration}",
            max_tokens=256)
    except Exception:
        return None


# ── 查询理解 ────────────────────────────────────────────

# 通用后缀 token 不做简称归一（「科技公司营收」不得命中「日联科技」）
_GENERIC_CO_TOKENS = ("科技", "股份", "集团", "控股", "实业", "公司")


def _resolve_companies(cands: list[str], corpus) -> list[str]:
    """D20 简称归一：c 与已知公司双向子串（c in k 或 k in c）且唯一
    命中时替换为全称。query 用简称（茅台）而语料是全称（贵州茅台）
    时，doc_scope 精确匹配 miss → 检索泄漏全库（实测「茅台 2026
    半年度扣非」答成欧科亿 4.19 亿、四家引用混池仍 verify:pass；
    「茅台…趋势」trend:0期 拒答）。无命中/多命中（歧义）保持原样，
    由下游精确匹配自然失效。"""
    known = sorted({d.company for d in corpus.docs.values() if d.company})
    out = []
    for c in cands:
        # L2 显式别名表优先（v0.7 Phase 3）：双向子串之外的别名
        # （如「欧科」→「欧科亿」，子串逻辑方向反了的情形）
        c = mem.alias_of(c) or c
        if (c in known or c in _GENERIC_CO_TOKENS or len(c) < 2
                or not "一" <= c[0] <= "鿿"):
            out.append(c)
            continue
        hits = [k for k in known if c in k or k in c]
        out.append(hits[0] if len(hits) == 1 else c)
    return list(dict.fromkeys(out))


def rule_understand(query: str, corpus) -> dict:
    """规则查询理解（heuristic 主路径 / LLM 故障兜底）。

    相比 v0.4 的 _doc_filter 只回首个公司，这里全量命中——多公司是
    compare 分解的前提。period 复用 period_hints 的互斥匹配。
    D20：全名子串 miss 时补简称面——jieba 分词 token 双向子串归一
    （「茅台近几年…」→ 贵州茅台），歧义/通用后缀不命中。
    """
    known = {d.company for d in corpus.docs.values() if d.company}
    companies = [k for k in dict.fromkeys(
        d.company for d in corpus.docs.values()
        if d.company and d.company in query)]
    import jieba
    toks = [t for t in jieba.cut(query)
            if len(t) >= 2 and "一" <= t[0] <= "鿿"]
    for k in _resolve_companies(toks, corpus):
        if k in known and k not in companies:
            companies.append(k)
    hints = period_hints(query)
    if len(companies) >= 2 and any(w in query for w in _COMPARE_WORDS):
        intent = "compare"
    elif _wants_trend(query):
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
    D18 裸年份→FY：只有年份没有期间词时默认年报口径（「2025 年营收」
    = 2025FY）。此前 hints 为空 → 期间不限定 → Q1/H1 的数顶掉年报数
    （2026-10-19 innolight 库 6 处失败共同根因）。年份判定复用
    query_years（含口语二位「15年」）；该公司无 FY 文档时消费端
    回退未过滤结果（retrieve 的 allowed 为空分支），不会误拒。
    """
    if any(w in query for w in ("半年度", "半年报", "中报", "二季度", "上半年", "H1")):
        return ["H1"]
    if "一季" in query or "Q1" in query:
        return ["Q1"]
    if "三季" in query or "Q3" in query:
        return ["Q3"]
    if "年度" in query or "年报" in query or "四季度" in query or "全年" in query:
        return ["FY"]
    if query_years(query):
        return ["FY"]
    return []



# ── v0.7 Planner/派生值的模块级助手（确定性，可单测） ────


def _wants_plan(query: str, u: dict) -> bool:
    """多跳信号（确定性路由闸，Phase 2）：≥3 家公司 / 跨年 / ≥2 指标族
    ——只有这些形态进 Planner，其余照旧 agent_research（不拿单跳题
    冒规划失败的险）。"""
    comps = u.get("companies") or []
    if len(comps) >= 3:
        return True
    if len(query_years(query)) >= 2:
        return True
    return len(mem.metric_families(query)) >= 2


def _validate_plan(subgoals, u, query, corpus) -> list[dict]:
    """Planner 校验器（确定性，Phase 2）：action 白名单（只收 search）；
    公司过 D20 归一；期间规约到 Q1/H1/Q3/FY；去重；上限 MAX_SUBGOALS。
    校验器只删不造——不替模型发明子目标（封闭世界：LLM 规划，代码裁决）。
    全删空 = 校验失败 → 调用方回退 agent_research。"""
    out: list[dict] = []
    seen: set[tuple] = set()
    comps = [c for c in (u.get("companies") or []) if c]
    for sg in subgoals:
        if not isinstance(sg, dict):
            continue
        if str(sg.get("action", "")).strip() != "search":
            continue
        comp = str(sg.get("company", "") or "").strip()
        if comp:
            comp = _resolve_companies([comp], corpus)[0]
        elif len(comps) == 1:
            comp = comps[0]
        period = str(sg.get("period", "") or "").strip().upper()
        if period not in ("Q1", "H1", "Q3", "FY"):
            ph = period_hints(period)
            period = ph[0] if ph else ""
        metric = str(sg.get("metric", "") or "").strip()[:12]
        if not comp and not comps:
            continue          # 无公司锚点且理解也没有 → 检索面太宽，弃
        k = (comp, period, metric)
        if k in seen:
            continue
        seen.add(k)
        out.append({"action": "search", "company": comp, "period": period,
                    "metric": metric})
        if len(out) >= MAX_SUBGOALS:
            break
    return out


def _ckpt_ok(hits, metric: str) -> bool:
    """子目标检查点（Phase 2，确定性）：证据须「财报级数字在场 + 指标词
    命中」——不过检查点的子目标不进答案的证据池（verify-as-you-go，
    切传播边，DESIGN-v0.7 §3 杠杆 2）。"""
    mkey = (metric or "").strip()
    for h in hits:
        ch = h.chunk if hasattr(h, "chunk") else h
        t = ch.text
        if not re.search(r"\d{4}", t.replace(",", "").replace(".", "")):
            continue
        if not mkey or mkey in t or any(
                w in t for w in expand_query(mkey).split()[:6]):
            return True
    return False


def _derived_ok(derived, pool) -> bool:
    """派生值出口对账（Phase 4）：登记的 {a, b, value} 按证据池重算——
    操作数必须真在场于证据池文本（信任登记不如重算），且
    |a−b−value| ≤ 0.01（原币分位容差）。空登记恒过。"""
    if not derived:
        return True
    texts = re.sub(r"[^\d.]", "", " ".join(
        (h.text if hasattr(h, "text") else h.chunk.text) for h in pool))
    for d in derived:
        a, b = d.get("a"), d.get("b")
        if a is None or b is None or d.get("value") is None:
            return False
        for v in (a, b):
            s = f"{v:.2f}"
            if s not in texts and s.rstrip("0").rstrip(".") not in texts:
                return False
        if abs((a - b) - d["value"]) > 0.01:
            return False
    return True


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
                    "三季→Q3，年度/年报/四季度→FY，仅提年份不提期间"
                    "（如「2025 年营收」）也算 FY，未提及则空串）；"
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
                # D20 简称归一在出口统一做：LLM 抽「茅台」→「贵州茅台」，
                # 使下方 known 子集守卫不再触发规则兜底（兜底会把
                # 已抽对的简称名单整体换掉）
                u["companies"] = _resolve_companies(u["companies"], corpus)
                trace = ["understand:llm"]
            except Exception:
                trace = ["understand:llm→fallback"]
        else:
            trace = ["understand:rule"]
        if u is None:
            u = rule_understand(q, corpus)
        else:
            # 趋势守卫：小模型实测把趋势问句判成 summary（单文档残渣）
            if u.get("intent") == "summary" and _wants_trend(q):
                u["intent"] = "trend"
            # 公司名纠偏：LLM 幻觉出语料外的公司时，用规则命中的兜底
            known = {d.company for d in corpus.docs.values() if d.company}
            rule_u = rule_understand(q, corpus)
            # 公司漏检守卫：规则的子串匹配是确定性的，LLM 漏抽一家
            # 会让 compare 退化成单公司 metric（samples 实测哪家题）
            if len(rule_u["companies"]) > len(u.get("companies") or []):
                u["companies"] = rule_u["companies"]
            if not set(u["companies"]) <= known:
                if rule_u["companies"]:
                    u["companies"] = rule_u["companies"]
            # 意图守卫：小模型会把「哪家赚得多」判成合法枚举 metric——
            # 公司≥2 且 query 含对比词时强制 compare（多跳分解的前提）
            if (u.get("intent") == "metric" and len(u.get("companies") or []) >= 2
                    and any(w in q for w in _COMPARE_WORDS)):
                u["intent"] = "compare"
            # period 覆盖：query 有明确期间词时规则是确定性子串匹配，
            # LLM 抽取实测会稳定抽错（「第一季度」→H1，v0.6.2 提示词
            # 加长后劣化）——规则非空即覆盖；query 无期间词时规则为空，
            # 保留 LLM 对隐式期间的判断
            if rule_u["period"]:
                u["period"] = rule_u["period"]
            # trend 必要条件：query 含趋势词（小模型会把「预计…多少」
            # 判成 trend 并在 reason 里幻觉数字，实测）。降级目标用规则
            # 意图而非硬编码 metric——同一 query 三次返回三种意图，直接
            # 降 metric 会让 compare 题漏判（samples 哪家题实测）
            if u.get("intent") == "trend" and not _wants_trend(q):
                u["intent"] = rule_u["intent"] if rule_u["intent"] != "trend" else "metric"
            # 对称升级（v0.7 差分信号）：query 命中「第[二两四]季度/单季」
            # 时规则层判 trend（差分要走确定性遍历+derive），小模型却判
            # metric 进 agent 环——模型自己差不出 Q4=FY−Q3，9b 真机
            # quarters 集 6 道派生题全灭的根因。trend/metric 争议一律
            # 以规则层为准（与上面的降级守卫同一家族，方向相反）
            if u.get("intent") == "metric" and rule_u["intent"] == "trend":
                u["intent"] = "trend"
            # 股价/涨跌类守卫：小模型偶发判 metric 拿净利表头行回答股价
            # 问题（实测），语料是财报不含行情——归 other 拒答
            if u.get("intent") == "metric" and any(
                    w in q for w in ("股价", "会涨", "会跌", "涨停", "跌停")):
                u["intent"] = "other"
            # 管理层叙述守卫：「管理层怎么说/怎么讲」是问 MD&A 叙述内容，
            # 小模型偶发判 other 拒答（实测）——有报告词在场时归 summary
            if (u.get("intent") == "other"
                    and any(w in q for w in ("管理层", "董事长", "怎么说的", "怎么讲"))
                    and any(w in q for w in ("中报", "年报", "季报", "报告", "财报"))):
                u["intent"] = "summary"
        # L0 会话记忆（v0.7 Phase 3）：「那 Q3 呢？」——公司/期间/年份/
        # 指标词缺省从上一轮继承（只在字段为空时补，query 提到新实体
        # 不触发）。只继承 understanding 字段与指标词，永不继承自由
        # 文本——切断叙述的传播边（DESIGN-v0.7 §7.3）
        eff = state.get("effective_query", q)
        u.setdefault("years", sorted(query_years(q)))
        if state.get("history"):
            u, eff_extra, inherit_marks = mem.inherit_priors(u, state["history"], q)
            if eff_extra and eff_extra not in eff:
                eff = f"{eff} {eff_extra}"
            # 裸追问意图救援：「中际旭创2026年上半年呢」无指标词 →
            # 规则判 other 直接拒答（实测）。只要 query 自带或继承了
            # 任一数据信号（公司/期间/年份/指标词），这是数据追问不是
            # 闲聊——归 metric，让检索/闸门决定答不答得上，而不是在
            # 理解层就枪毙；真闲聊（无任何信号）保持 other 拒答
            if (u.get("intent") == "other"
                    and (inherit_marks or u.get("companies") or u.get("period")
                         or u.get("years")
                         or any(h in q for h in mem._METRIC_HINTS))):
                u["intent"] = "metric"
                trace = trace + ["understand:inherit→metric"]
            trace = trace + inherit_marks
        return {**state, "understanding": u, "effective_query": eff,
                "trace": trace}

    def route(state: GraphState) -> GraphState:
        """纯函数分发（v0.5）：消费 understand 的 intent——理解已在
        understand 节点完成（LLM 或规则），这里只做分发与期间守卫，
        不再调 LLM/关键词。"""
        q = state["query"]
        u = state.get("understanding") or rule_understand(q, corpus)
        intent = u.get("intent", "metric")
        # 期间守卫：query 里的年份若库中全无对应报告，先对账再回答——拒。
        # 年份统一走 query_years（含二位口语年份「15年」→2015，此前
        # 此处内联正则只认 20xx，用户实测 15 年中报静默答了别的年份）
        q_years = query_years(q)
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
            # 年份：query 自带优先；裸追问（「归母呢」）不带年份时用
            # L0 继承进 understanding 的年份（v0.7 Phase 3）
            years = query_years(state["query"]) or set(u.get("years") or [])
            allowed = {d for d, doc in corpus.docs.items()
                       if (not companies or doc.company in companies)
                       and period in doc.period
                       and (not years or doc.period[:4] in years)}
            if allowed:
                scoped = corpus.search(eff, k=TOP_K, doc_filter=filt, doc_scope=allowed)
                if scoped:
                    hits = scoped
        # metric 意图：主要会计数据表页前置（原仅 agent 路有，heuristic
        # 漏挂——茅台 2026H1 扣非 44,464,207,646.01 的数值块无标签头
        # （PDF 抽取丢失，行首残「常性损益的净利润」），词面检索排不进
        # top-8，前置把表页带进 hits 才可达；窗口见 front_key_table）
        if (state.get("understanding") or {}).get("intent") == "metric":
            hits = front_key_table(hits, state["understanding"],
                                   state["query"], corpus)
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
        # 表块拼接（D16+B2）：metric 意图下把同页「口径标签块+光数值块」
        # 拼成整行——检索只召回标签块，光数值块（真值）永远不在场
        if (state.get("understanding") or {}).get("intent") == "metric":
            hits = stitch_adjacent(hits, corpus)
        # 页级证据（small-to-big）：命中块 ±2 窗口的兄弟块一并进入
        # 生成与校验的视野——检索精度不动，上下文在生成侧补全
        page_chunks = page_evidence(hits, corpus)
        fallback: list[str] = []
        # 趋势题：确定性骨架 + LLM 仅做归纳（小模型直生成多期长答案
        # 实测自相矛盾——见 trend_skeleton 注释）。骨架纯代码，模式无关
        # ——heuristic 同样输出逐期骨架（此前被 llm 条件包裹退化成
        # 普通抽取式，doc=N 实测）
        if (state.get("understanding") or {}).get("intent") == "trend":
            qw = _quarter_window(state["query"])
            skeleton, fact_rows, derived = trend_skeleton_ex(
                hits, corpus, quarter_n=qw and qw["n"],
                quarter_year=qw and qw["year"])
            if not skeleton:
                return {**state, "answer": "", "verdict": "refused",
                        "trace": ["generate:trend-no-facts→refuse"]}
            analysis = ""
            narr_trace: list[str] = []
            # LLM 默认在环（v0.6.5）：叙述层默认开——数字闸+方向闸双闸
            # 兜底后叙述风险被架构封住（违例→退骨架=改前行为，保守失败）。
            # 此前默认关是因为小模型把 +90.05% 说成「大幅回落」且无机械
            # 闸可拦；方向闸补上这一维后默认值翻转。DZ_TREND_ANALYSIS：
            # 未设=归纳叙述（默认）；enabled=加深度投资分析；off=骨架
            # 直出（逃生门，延迟敏感场景）。
            tier = os.environ.get("DZ_TREND_ANALYSIS", "").strip().lower()
            deep = tier == "enabled"
            if (state["mode"] == "llm" and llmx.llm_enabled() and tier != "off"):
                # 单季统一视图（v0.6.6）：各行为日历单季（派生行已标注），
                # 环比可比——叙述不必再被「口径不可比」捆住手脚；且问
                # 上升/下降时首句必须直接给方向结论（用户问的就是这个）
                caliber_rule = (
                    "①数据行均为**单季口径**（标注「派生」的为累计数差分"
                    "所得，如 Q2=H1−Q1），单季之间**环比可比**，跨年同季"
                    "看同比" if qw else
                    "①H1 是半年累计、Q1 是单季、Q3 是前三季累计——绝对值"
                    "不可直接比较，跨期方向只看同比增速")
                first_sent = ("；⑤若问题问上升/下降，第一句必须直接给出"
                              "方向结论并带依据数字" if qw else "")
                prompt = (
                    "你是财报分析师。基于给定逐期数据做投资分析（趋势判断/"
                    "盈利质量/风险点）。铁律：" + caliber_rule +
                    "；②只准使用给定数据里的数字，禁止任何其他数字（含"
                    "换算与预测）；③方向结论必须与同比数字的符号一致（正="
                    "增长，负=下降），方向句必须带同比数字；④禁止编造数据外"
                    "的归因（产能/需求/竞争等未被数据提及的因素）" +
                    first_sent + "。三到五句话。" if deep else
                    "你是财报分析师。基于给定逐期数据归纳趋势并给出方向"
                    "结论。铁律：" + caliber_rule +
                    "；②只准使用给定数据里的数字，禁止任何其他数字（含"
                    "换算与预测）；③方向结论必须与同比数字的符号一致（正="
                    "增长，负=下降），方向句必须带同比数字；④禁止编造数据"
                    "外的归因" + first_sent + "。两到四句话。")
                try:
                    analysis = llmx.chat_text(prompt, skeleton,
                                              max_tokens=2048).strip()
                except Exception:
                    analysis = ""
                    narr_trace.append("narration:llm-fail")   # 静默吞=观测盲区
                if analysis == "" and "narration:llm-fail" not in narr_trace:
                    narr_trace.append("narration:empty")
                # 双闸之一·数字闸：叙述里的数字必须全部来自骨架行——表外
                # 数字=幻觉，整段丢弃（宁缺毋滥；引用骨架内数字允许）
                if analysis and not _analysis_numbers_ok(analysis, skeleton):
                    analysis = ""
                    narr_trace.append("number-gate")
                # 双闸之二·方向闸：方向词与骨架同比符号相容——+90.05%
                # 被说成「大幅回落」时数字/引用全对，只有方向错
                if analysis:
                    bad_dir = direction_gate(analysis, skeleton)
                    if bad_dir:
                        analysis = ""
                        narr_trace.append(f"direction-gate({bad_dir})")
                # 裁判自测（非对称信任）：只能否决，永不解锁——双闸先跑
                # 先裁，裁判只看存活叙述；误杀上限=退骨架（保守失败）
                if analysis:
                    verdict = _self_check(state["query"], skeleton, analysis)
                    if verdict:
                        if verdict.get("sensitive"):
                            # 敏感句只剥不重写（重写=教模型绕词表）
                            stripped = _strip_sensitive(analysis)
                            if stripped:
                                analysis = stripped
                                narr_trace.append("selfcheck:veto(sensitive→剥句)")
                            else:
                                analysis = ""
                                narr_trace.append("selfcheck:veto(sensitive)")
                        elif not verdict.get("faithful", True) \
                                or not verdict.get("on_topic", True):
                            # 有界修复 ×1：issues 回灌重写（Reflexion 单步；
                            # 无界反思=小模型死循环入口，与 MAX_RETRIES=1
                            # 同纪律）。修复稿照过双闸，再不过退骨架
                            issues = "；".join(verdict.get("issues") or [])
                            try:
                                repair = llmx.chat_text(
                                    prompt + f"上一稿有这些问题，重写（其余"
                                    f"铁律不变）：{issues}",
                                    skeleton, max_tokens=2048).strip()
                            except Exception:
                                repair = ""
                            if (repair and _analysis_numbers_ok(repair, skeleton)
                                    and not direction_gate(repair, skeleton)):
                                analysis = repair
                                narr_trace.append("selfcheck:repair(1)→ok")
                            else:
                                analysis = ""
                                narr_trace.append("selfcheck:repair(1)")
                        else:
                            narr_trace.append("selfcheck:pass")
            # 产品态呈现（v0.6.8）：结论段 + 紧凑表 + 行内引用——算式/
            # 双引用展开/派生声明/免责属审计信息，进 state['trend_audit']
            # （-v stderr / --json 展开）；verify 对产品态正文照常全量跑
            def _compact(line: str) -> str:
                line = re.sub(r"（= [^）]*，原文未直接披露）", "", line)
                return (line.replace("（单季·派生，", "（派生，")
                            .replace("（单季·派生）", "（派生）")
                            .replace("（单季）", ""))
            compact = "\n".join(_compact(x) for x in skeleton.splitlines())
            body = compact
            if analysis:
                body = f"{'分析' if deep else '归纳'}：{analysis}\n\n{body}"
            state["trend_audit"] = (skeleton + "\n（以上为财报事实梳理，"
                                    "不构成投资建议）")
            # 引用收窄：列表 = 答案文本里实际出现的引用——此前 cite 全部
            # hits（含骨架没采用行的期间/页），引用虚胖且不可回溯到答案
            cited = list(dict.fromkeys(re.findall(r"\[[^\]]+\]", body)))
            ev, seen_docs = [], set()
            for h in hits:
                if h.chunk.doc_id in seen_docs:
                    continue
                seen_docs.add(h.chunk.doc_id)
                ev += corpus.chunks_page(h.chunk.doc_id, h.chunk.page)
            # 骨架行进证据池：换算值（147.9 亿）不在原文（14,789,074,837.80）
            # 里，coverage 按字面重叠必然误降级——骨架是证据的确定性解析
            # 产物（数字闸门同源），本身即合法证据。**按文档去重各一块**
            # （v0.6.6）：hits 是每文档 k=5 连发，取 hits[:6] 只覆盖头两
            # 个文档——单季视图 4 行引用的文档大多没有合成块入池，覆盖
            # 率 0.102 整体误降级（实测）；页取该档首个命中块，与行内
            # 引用同页，保证 citation_in 命中入池
            from duizhang.schemas import Chunk as _C
            sk_seen: set[str] = set()
            for h in hits:
                did = h.chunk.doc_id
                if did in sk_seen:
                    continue
                sk_seen.add(did)
                ev.append(_C(chunk_id=f"sk#{did}", doc_id=did,
                             page=h.chunk.page, seq=0, text=skeleton))
            # 叙述态进 trace：narrate(llm)=叙述存活（含 selfcheck:pass/
            # veto剥句 等后缀）；闸击落时标注退骨架
            if analysis:
                narr_marker = ("generate:trend→narrate(llm)"
                               + ("→" + "→".join(narr_trace) if narr_trace else ""))
            elif narr_trace:
                narr_marker = ("generate:trend→" + "→".join(narr_trace)
                               + "→skeleton")
            else:
                narr_marker = "generate:trend"
            return {**state, "answer": body, "citations": cited,
                    "evidence": ev or hits,
                    "fact_rows": fact_rows, "derived": derived,
                    "trace": [narr_marker]}
        if state["mode"] == "llm" and llmx.llm_enabled():
            # 证据按页聚合（页首标注），块序拼接；数字归一化注释随块；
            # 断词残行先经 split_fix 补全标签——小模型看到「常性损益的
            # 净利润 44,464…」无标签残行时无从执行口径镜像，拿归母数
            # 答扣非题（茅台 2026H1 实测）
            by_page: dict[tuple, list] = {}
            for c in page_chunks:
                by_page.setdefault((c.doc_id, c.page), []).append(c)
            parts = []
            for (doc_id, page), cs in by_page.items():
                body = "\n".join(
                    ans.split_fix(c.text)[:500] +
                    (f"（数字参考：{'；'.join(c.num_notes[:3])}）"
                     if c.num_notes else "") for c in cs)
                parts.append(f"[{doc_id} 第{page}页]\n{body}")
            evidence = "\n\n".join(parts)

            try:
                out = llmx.chat_text(
                    "你是财报问答助手。只依据给定证据回答；每个事实句末尾必须"
                    "标注来源，格式如 [公司_报告期 第N页]。证据来自 PDF 表格"
                    "抽取，数字可能分行或断裂——可把相关数字碎片组织成答案"
                    "并如实标注来源，但禁止编造不在证据里的数字。仔细区分"
                    "指标本身的数值区间与「同比增加/将增加」的变动幅度，"
                    "引用问题所问的那个。问「净利润」而未提扣非时，默认指"
                    "归属于上市公司股东的净利润（非扣非口径）。PDF 断行时它显示为"
                    "「的净利润（元）」开头的数据行，取该行的本期值（第一列数字）。"
                    "反之，问句明确要「扣非/扣除非经常性损益」时，必须取"
                    "「归属于上市公司股东的扣除非经常性损益的净利润」那一行"
                    "的本期值——「净利润（元）」行是归母口径，数值相近但不同"
                    "。问「总收入/营业总收入」时必须取"
                    "「营业总收入」行——「营业收入」是子口径（茅台 2026H1 "
                    "总收入 92,278,072,083.21 vs 营业收入 90,703,260,964.48），"
                    "扣非残行行首已补全标签，按标签取行。"
                    "确实无任何"
                    "相关数字才回答「原文中没有找到依据」。",
                    f"问题：{state['query']}\n\n证据：\n{evidence}")
                no_evidence = any(p in out for p in (
                    "没有找到依据", "未找到依据", "未找到相关", "无法从",
                    "无法确定", "无法回答")) or len(out.strip()) < 15
                # 口径数值闸（D22）：数字在场但行不对（拿归母答扣非、
                # 拿营业收入答总收入）→ 降级抽取式（口径守卫三层已保证
                # 抽取式取对行），数字在场引用级 verify 管不住。对照源
                # 与 verify_citations 同律：页级集合 ∪ hits——证据窗漏掉
                # 的行 hits 里有（总资产题实测窗只到 p3#3 而 #9 在 hits）
                if not no_evidence and caliber_mismatch(
                        state["query"], out,
                        list(page_chunks) + list(hits)):
                    fallback = ["generate:llm→caliber-mismatch→extractive"]
                elif not no_evidence:
                    # 引用匹配空白不敏感（D13）：模型手写空格漂移
                    # （第 8 页/粘连）不该丢掉引用列表
                    cited = list(dict.fromkeys(
                        h.chunk.citation for h in hits[:8]
                        if ans.citation_in(out, h.chunk.citation)))
                    return {**state, "answer": out, "citations": cited,
                            "evidence": page_chunks, "trace": ["generate:llm"]}
                else:
                    # LLM 判无据 → 降级抽取式（永远有真凭实据），而非拒答——
                    # 与 verify 的「降级而非失败」同一纪律
                    fallback = ["generate:llm-no-evidence→extractive"]
            except Exception as e:  # LLM 故障 → 走抽取式，绝不空手而归（留痕）
                # D24：异常类型进标记——宽 except 曾把 caliber_mismatch 的
                # AttributeError 谎报成「LLM 故障」，观测标记必须可诊断
                fallback = [f"generate:llm→fallback({type(e).__name__})"]
        body, cited = ans.extractive_answer(state, hits)
        return {**state, "answer": body, "citations": cited,
                "hits": hits,       # 拼接行进 verify 证据池，口径一致
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
        # 要点种子三源：MD&A 章节锚 + 关键指标检索 + 原命中块。
        # 「管理层怎么说/总结」要的是叙述性内容——MD&A（经营情况讨论
        # 与分析）是唯一可靠章节锚，泛关键词种子此前命中封面/回购明细
        # 表（实测）；长报告头部 12 块是封面/名录（茅台实测空要点）
        mda_probe = corpus.search("经营情况讨论 与分析 管理层", k=2,
                                  doc_filter=filt, doc_scope={target})
        mda_seed = []
        get_page = getattr(corpus, "chunks_page", None)
        if mda_probe and get_page and mda_probe[0].chunk.page:
            m = mda_probe[0].chunk
            mda_seed = get_page(m.doc_id, m.page)[1:7] \
                + get_page(m.doc_id, m.page + 1)[:6]
        company0 = (companies or [""])[0]
        probe = expand_query(f"{company0} 主要会计数据 营业收入 净利润 同比")
        seed_hits = corpus.search(probe, k=6, doc_filter=filt,
                                  doc_scope={target})
        seed = mda_seed + [h.chunk for h in seed_hits][:6] \
            + [h.chunk for h in hits if h.chunk.doc_id == target][:6]
        seen_ids = set()
        seed = [c for c in seed if not (c.chunk_id in seen_ids or seen_ids.add(c.chunk_id))]
        doc_chunks = seed + [c for c in corpus.chunks_of(target, limit=12)
                             if c.chunk_id not in seen_ids]
        head = corpus.docs[target]
        # 打分选行（替代关键词命中即收）：叙事动词加分、表头/会计政策
        # 残渣强减分——「净利润（元）」表头含关键词却无值（实测混入）
        _JUNK_PAT = re.compile(r"（元）\s*$|终止经营|少数股东|抵销|综合收益|"
                               r"会计准则|净亏损以|按所有权")
        _NARR_PAT = re.compile(r"实现|增长|下降|主要系|管理层|公司认为|"
                               r"报告期|展望|同比|预计")
        scored_lines = []
        for c in doc_chunks:
            for sent in ans._sentences(c.text):
                if _JUNK_PAT.search(sent):
                    continue
                sc = 1.0
                if _NARR_PAT.search(sent):
                    sc += 2
                if re.search(r"\d", sent):
                    sc += 1
                if any(w in sent for w in ("营业收入", "净利润")):
                    sc += 1
                scored_lines.append((sc, f"{sent} {c.citation}"))
        scored_lines.sort(key=lambda x: -x[0])
        key_lines = [ln for sc, ln in scored_lines if sc >= 3][:5]
        body = (f"{head.company}（{head.period} {head.kind}）要点：\n"
                + "\n".join(f"· {line}" for line in key_lines))
        return {**state, "answer": body, "citations": [target], "verdict": "pass",
                "trace": [f"summarize:{target}"]}

    def refuse(state: GraphState) -> GraphState:
        return {**state,
                "answer": "原文中没有找到依据。换个问法，或先 ingest 对应报告。",
                "citations": [], "verdict": "refused", "trace": ["refuse"]}

    def agent_research(state: GraphState) -> GraphState:
        """真 Agent 环（v0.6 起，v0.7 升级）：模型在循环里自主决定检索
        什么、何时收手——工具注册表 search / metrics_table / derive
        （白名单，注册表外 action 一律返回 "[]"），步数硬上限。工具层
        故障或零证据时回退固定检索管线（不空手）。出口仍走
        generate→verify，**自主检索不豁免对账闸门**。heuristic 模式不进
        此节点。
        v0.7 Phase 1 ReAct 步级观测：每步 (thought, action, obs, gate)
        四元组进 state['agent_steps']——gate ∈ new/dup/empty 是步级浪费
        率的可测代理（eₖ 侧；DESIGN-v0.7 §3 度量先行），eval --json 聚合
        成 waste_rate 出口。"""
        import time
        pool: dict[str, object] = {}
        u = state.get("understanding") or {}
        derived_ops: list[dict] = []
        steps: list[dict] = []

        def _tool_period(raw_p) -> tuple[str, str]:
            """工具期间参数 → (年, 后缀)。接受 2025FY / FY / 年报 三形。"""
            raw_p = str(raw_p or "").strip()
            year = raw_p[:4] if raw_p[:4].isdigit() else ""
            m = re.search(r"(H1E|Q3E|Q1|H1|Q3|FY)$", raw_p)
            sfx = m.group(1) if m else ""
            if not sfx:
                ph = period_hints(raw_p)
                sfx = ph[0] if ph else ""
            return year, sfx

        def _doc_metric_pool(comp: str, year: str, sfx: str):
            """公司+年+后缀 → (文档, 检索块池)。命中共用 agent 证据池。"""
            if not comp:
                return None, []
            docs = sorted((d for d, doc in corpus.docs.items()
                           if doc.company == comp
                           and (not sfx or doc.period.endswith(sfx))
                           and (not year or doc.period[:4] == year)),
                          key=lambda d: _period_key(corpus.docs[d].period))
            if not docs:
                return None, []
            d = docs[-1]
            hs = corpus.search("主要会计数据 " + expand_query("营业收入 净利润"),
                               k=5, doc_scope={d})
            if not hs:
                hs = corpus.search(expand_query("营业收入 净利润"),
                                   k=5, doc_scope={d})
            chunks = [h.chunk for h in hs]
            get_page = getattr(corpus, "chunks_page", None)
            # ±1 页窗（trend_rows 同款）：FY 年报「主要会计数据」锚页与
            # 检索 top 命中页常差一页，只取 top 页会让 metric_rows 拿不到
            # 锚行 → metrics_table 工具空手（9b 真机 2024FY obs=0 实测）
            if get_page:
                seen_ids = {c.chunk_id for c in chunks}
                for p in list(dict.fromkeys(
                        c.page for c in chunks if c.page))[:3]:
                    for pp in (p - 1, p, p + 1):
                        for c in get_page(d, pp):
                            if c.chunk_id not in seen_ids:
                                seen_ids.add(c.chunk_id)
                                chunks.append(c)
            for h in hs:
                if h.chunk.chunk_id not in pool:
                    pool[h.chunk.chunk_id] = h
            return d, chunks

        def run_search(args: dict) -> str:
            q = str(args.get("query", "")).strip()
            # 工具默认值从 understanding 继承（廉价结构化抽取做先验，
            # 模型显式传参可覆盖）——小模型单跳题常不传 company，
            # 全库检索会被千家同类内容淹没（实测 top1 全是无关公司）
            u_comps = [c for c in (u.get("companies") or []) if c]
            comp = str(args.get("company", "") or "") or \
                (u_comps[0] if len(u_comps) == 1 else "")
            # D25：模型会把简称原样写进工具参数（茅台），run_search
            # 精确匹配两轮空手 → agent:fallback——参数侧也过 D20 归一
            if comp:
                comp = _resolve_companies([comp], corpus)[0]
            # 期间以理解为准、参数只补空白：理解是模型对原问题的结构化
            # 决定（含规则回填），工具参数是检索战术——小模型实测会把
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

        def run_metrics_table(args: dict) -> str:
            """metrics_table 工具（Phase 4）：单期主要会计数据 → 结构化
            行 JSON。确定性解析（与 trend 层同原语 metric_rows），模型
            不再从碎片拼数。"""
            comp = _resolve_companies(
                [str(args.get("company", "") or "")], corpus)[0]
            if not comp:
                return "[]"
            year, sfx = _tool_period(args.get("period", ""))
            d, chunks = _doc_metric_pool(comp, year, sfx)
            if not d:
                return "[]"
            rows = metric_rows(chunks,
                               cumulative=corpus.docs[d].period.endswith("Q3"))
            if not rows:
                return "[]"
            return json.dumps(
                [{"metric": r["label"], "value": r["yi"], "yoy": r["yoy"],
                  "doc": d, "period": corpus.docs[d].period,
                  "quote": r["seg"][:120]} for r in rows],
                ensure_ascii=False)

        def run_derive(args: dict) -> str:
            """derive 工具（Phase 4 差分算子）：Q2=H1−Q1、Q4=FY−Q3。
            原币精度计算；操作数行不可解析时返回 error 对象（模型可见
            的保守失败），绝不打印错数。结果登记进 derived_ops——
            verify 出口按「操作数必须真在场于证据池」重算 ±0.01。"""
            comp = _resolve_companies(
                [str(args.get("company", "") or "")], corpus)[0]
            tgt = str(args.get("period", "") or "").upper()
            if not comp or tgt not in ("Q2", "Q4"):
                return "[]"
            years = sorted(query_years(state["query"]))
            year = str(args.get("year", "") or "").strip("年") \
                or (years[-1] if years else "")
            if not year.isdigit():
                return json.dumps({"error": "缺年份参数"}, ensure_ascii=False)
            word = str(args.get("metric", "") or "营收")
            label, key = ("归母净利", "净利润") if (
                "净" in word or "利润" in word) else ("营收", "营业收入")
            hi_tag, lo_tag = ("H1", "Q1") if tgt == "Q2" else ("FY", "Q3")
            d_hi, ch_hi = _doc_metric_pool(comp, year, hi_tag)
            d_lo, ch_lo = _doc_metric_pool(comp, year, lo_tag)
            r_hi = metric_rows(ch_hi, requires=((label, key),))
            r_lo = metric_rows(ch_lo, requires=((label, key),))
            if not (r_hi and r_lo):
                return json.dumps(
                    {"error": f"{comp} {year}{hi_tag}/{year}{lo_tag} 的"
                              f"{label}行不可解析，无法差分"},
                    ensure_ascii=False)
            hi, lo = r_hi[0], r_lo[0]
            val = hi["raw"] - lo["raw"]
            if val <= 0:
                return json.dumps({"error": "差分为负，操作数行可疑"},
                                  ensure_ascii=False)
            hi_cite = (ch_hi[0].citation if ch_hi else "")
            lo_cite = (ch_lo[0].citation if ch_lo else "")
            derived_ops.append({"label": f"{comp}_{year}{tgt}{label}",
                                "value": val, "a": hi["raw"], "b": lo["raw"]})
            return json.dumps(
                {"metric": label, "period": f"{year}{tgt}",
                 "value_yi": f"{val / 1e8:,.1f} 亿",
                 "formula": (f"{year}{hi_tag} {hi['yi']} − "
                             f"{year}{lo_tag} {lo['yi']}"),
                 "derived": True,
                 "citations": [hi_cite, lo_cite],
                 "note": "派生值：原文未直接披露；引用须带两个操作数来源"},
                ensure_ascii=False)

        def _run_tool(name: str, a: dict) -> str:
            if name == "search":
                return run_search(a)
            if name == "metrics_table":
                return run_metrics_table(a)
            if name == "derive":
                return run_derive(a)
            return "[]"      # 注册表白名单外：封闭世界契约第 2 条

        n_steps = n_calls = 0
        agent_queries: list[str] = []  # Agent 实际用的检索词（写进
        # effective_query：证据是靠这些词找到的，下游抽取式/verify 打分
        # 必须与之一致——原始口语 query 与证据零重叠会误拒，实测）
        seen: dict[str, str] = {}      # 幂等缓存：小模型会把同一调用原样重复 6 步（实测）
        stopped = "budget"
        try:
            msgs = [
                {"role": "system", "content":
                    "你是财报研究 agent。可用工具：search（检索证据块）、"
                    "metrics_table（取某期主要会计数据行）、derive（单季差分"
                    "Q2=H1−Q1、Q4=FY−Q3，原文不披露单季时用它，禁止自己心算）。"
                    "对比类问题逐家公司检索；证据足够回答后停止调用工具并"
                    "简述结论。每次检索换不同角度（换指标词/换公司/换期间），"
                    "不重复。对比结论必须与所引数字的大小关系一致。"},
                {"role": "user", "content": state["query"]},
            ]
            for _ in range(MAX_AGENT_STEPS):
                t_step, n0 = time.perf_counter(), len(steps)
                resp = llmx.chat_with_tools(msgs, AGENT_TOOLS)
                n_steps += 1
                thought = (resp.get("content") or "").strip()
                if not resp["tool_calls"]:
                    stopped = "self"
                    steps.append({"i": n_steps, "thought": thought[:80],
                                  "action": "stop(self)", "obs": 0,
                                  "obs_doc": "", "gate": "-", "ms": 0})
                    break
                msgs.append({"role": "assistant", "content": resp["content"],
                             "tool_calls": [{"function": {
                                 "name": tc["name"], "arguments": tc["arguments"]}}
                                 for tc in resp["tool_calls"]]})
                fresh = False
                for tc in resp["tool_calls"]:
                    key = json.dumps(tc["arguments"], sort_keys=True,
                                     ensure_ascii=False)
                    action = f"{tc['name']}{json.dumps(tc['arguments'], ensure_ascii=False)[:64]}"
                    step = {"i": n_steps, "thought": thought[:80],
                            "action": action, "ms": 0}
                    steps.append(step)
                    if key in seen:      # 重复调用：回缓存结果并提示换角度
                        msgs.append({"role": "tool",
                                     "content": seen[key] + "（重复检索：结果同上。"
                                                "请换检索角度或停止调用工具。）"})
                        step.update(obs=0, obs_doc="", gate="dup")
                        continue
                    fresh = True
                    n_calls += 1
                    obs = _run_tool(tc["name"], tc["arguments"])
                    seen[key] = obs
                    msgs.append({"role": "tool", "content": obs})
                    # 观测摘要：条数/首文档（步级观测的可读半边）
                    try:
                        parsed_obs = json.loads(obs)
                    except (ValueError, TypeError):
                        parsed_obs = None
                    if isinstance(parsed_obs, list):
                        n_obs = len(parsed_obs)
                        first_doc = (parsed_obs[0].get("doc", "")
                                     if parsed_obs and isinstance(parsed_obs[0], dict) else "")
                    elif isinstance(parsed_obs, dict):
                        n_obs = 1 if "error" not in parsed_obs else 0
                        first_doc = str(parsed_obs.get("doc", ""))
                    else:
                        n_obs, first_doc = 0, ""
                    step.update(obs=n_obs, obs_doc=first_doc,
                                gate="new" if n_obs else "empty")
                dt = round((time.perf_counter() - t_step) * 1000)
                for s in steps[n0:]:
                    s["ms"] = dt        # 该步耗时（LLM+工具观测，Phase 1 观测面）
                if not fresh and pool:   # 整批全是重复 = 模型没有新问题要问
                    stopped = "重复收手"
                    break
        except Exception:
            n_steps = -1
            steps = []
        if not pool:   # 工具层故障/模型没找到证据 → 固定管线兜底
            u = state.get("understanding") or rule_understand(state["query"], corpus)
            base = decompose(state) if u.get("intent") == "compare" else retrieve(state)
            return {**base, "agent_steps": steps, "trace": ["agent:fallback"] + base["trace"]}
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
        # 表块拼接（D16 同款，agent 路补挂）：前置/自采的块是原始块，
        # 标签块与数值块分离着进抽取——扣非题抽取式拿不到「标签+数字」
        # 整行，非经常性损益明细噪声行（词面含损益）反而当选（实测）
        hits_out = stitch_adjacent(hits_out, corpus)
        # metric 意图：主要会计数据表前置（封顶+归因跳过，见 front_key_table）
        hits_out = front_key_table(hits_out, u, state["query"], corpus)
        eff = state.get("effective_query", state["query"])
        for q in agent_queries:
            if q not in eff:
                eff = f"{eff} {q}"
        return {**state, "hits": hits_out, "effective_query": eff,
                "agent_steps": steps, "derived": derived_ops,
                "trace": [f"agent:{n_steps}步/{n_calls}次tool/{len(pool)}块证据/{stopped}"]}

    def trend_research(state: GraphState) -> GraphState:
        """跨期研究（v0.6.2）：遍历该公司报告期（时间序，最近 6 期）、
        每期 doc_scope 检索关键指标、按时间升序合并——确定性遍历，
        双模式共用；比让 Agent 自己逐期调工具可靠（小模型会漏期/死循环）。
        出口照常 generate→verify，多期不豁免闸门。"""
        u = state.get("understanding") or rule_understand(state["query"], corpus)
        companies = [c for c in (u.get("companies") or []) if c] or [""]
        quarterly = "季" in state["query"]          # 季度问法 → 季度粒度报告期
        # v0.7 Phase 4 单季解锁：显式单季问法（单季/每季/各季/分季/
        # 第X季度）→ 上限 8 期且 FY 回收（Q4=FY−Q3 差分的操作数；
        # 普通「季度」问法保持旧行为 cap 6 无 FY，eval 不回归）
        want_quarters = _QUARTER_Q_RE.search(state["query"]) or any(
            w in state["query"] for w in ("每季", "各季", "分季", "每季度"))
        # 「近N个季度」单季统一视图（v0.6.6）：同样需要 FY 回收（Q4 差分
        # 操作数），且窗口随 N 放大（4 个日历季跨两年取原料）
        qw = _quarter_window(state["query"])   # 模型语义的规则兜底
        if qw:
            want_quarters = True
        sfxs = ("Q1", "H1", "Q3", "FY", "H1E") if want_quarters \
            else ("Q1", "H1", "Q3", "H1E")
        cap = max(8, (qw or {}).get("n", 0) * 2 + 2) if want_quarters else 6
        periods = sorted(
            (d for d, doc in corpus.docs.items()
             if doc.company in companies
             and (not qw or not qw["year"]
                  or doc.period[:4] == qw["year"])   # 年份切片：该年全粒度
             and (not quarterly or any(doc.period.endswith(sfx)
                                       for sfx in sfxs))),
            key=lambda d: _period_key(corpus.docs[d].period))[-cap:]
        hits: list = []
        for d in periods:                           # 时间升序：趋势叙事方向
            # 锚定「主要会计数据」节——一行式营收/净利润在那里；
            # 泛查会捞到利润表深处的碎片（实测终止经营表头残渣）
            hits += corpus.search("主要会计数据 " + expand_query("营业收入 净利润"),
                                  k=5, doc_filter=companies[0], doc_scope={d})
        return {**state, "hits": hits,
                "trace": [f"trend:{len(periods)}期"]}

    def memory_probe(state: GraphState) -> GraphState:
        """L1 读路径（v0.7 Phase 3）：同题键命中 → 直接组装答案，跳过
        检索与推理。**读不豁免**（公理 2）：拼出的答案与缓存证据行当场
        过 verify，不过闸照样掉头走全链路——缓存的是证据，不是结论。
        未命中零开销透传（sqlite 部分索引探针，毫秒级）。"""
        key = _fact_key(state)
        if key is not None:
            state = {**state, "fact_key": key}   # 写侧与 L0 轮摘要共用
        if not (key and mem.enabled()):
            return state
        status, row = mem.read_fact(corpus, key)
        if status == "stale":
            return {**state, "trace": ["memory:stale(指纹失效→supersede)"]}
        if status != "hit":
            return state
        quotes = mem.fact_citations(corpus, row)
        label = mem.CALIBER_LABEL.get(key["caliber"], key["caliber"])
        val = row.get("value_raw") or (
            f"{row['value_std']:,.2f}" if row.get("value_std") else "")
        if not val:
            return {**state, "trace": ["memory:hit(空值)→全链路"]}
        yoy = str(row.get("yoy") or "").rstrip("%")
        answer = (f"{key['company']} {key['period']} {label}为 {val} 元"
                  + (f"（同比 {yoy}%）" if yoy else ""))
        cited = [f"[{c.get('doc_id')} 第{c.get('page')}页]" if c.get("page")
                 else f"[{c.get('doc_id')}]"
                 for c in row.get("citations") or []]
        answer += " " + " ".join(cited)
        # 读不豁免：缓存证据行当池，答案当场过对账闸门
        verdict, cov = ans.verify(answer, quotes)
        if verdict == "pass":
            return {**state, "answer": answer, "citations": cited,
                    "verdict": "pass", "coverage": cov, "memory_hit": True,
                    "hits": [],
                    "trace": [f"memory:hit({key['period']}·{key['caliber']}"
                              f"·{row.get('created_at', '')[:10]})"]}
        return {**state,
                "trace": [f"memory:hit→verify:{verdict}→全链路"]}

    def plan(state: GraphState) -> GraphState:
        """Planner-Executor（v0.7 Phase 2）：多跳问题先出结构化子目标
        （LLM 规划、只此一处调用），确定性校验器过白名单/先验/上限，
        逐子目标执行检索 + 检查点验证（数字在场+指标词命中；不过有界
        重试 ×1，再不过显式弃）。弃掉的子目标进 plan_gaps——verify 出口
        追加「未覆盖」声明，**不允许静默吞掉缺口**。校验失败整体回退
        agent_research（现路径，不冒险）。"""
        u = state.get("understanding") or {}
        q = state["query"]
        subgoals: list[dict] = []
        try:
            r = llmx.chat_json(
                "你是财报研究规划器。把问题拆成不超过 5 个检索子目标。"
                "每个子目标是 {action, company, period, metric}：action "
                "只能是 search；company 用全称；period 用 Q1/H1/Q3/FY；"
                "metric 用指标词（营收/净利润/扣非/净资产/总资产）。"
                "跨期对比逐年各拆一个；多公司对比逐家各拆一个。"
                '只输出 JSON：{"subgoals": [...]}',
                f"问题：{q}\n已识别公司：{'、'.join(u.get('companies') or []) or '（无）'}\n"
                f"已识别期间：{u.get('period') or '（无）'}",
                max_tokens=384)
            raw = r.get("subgoals") if isinstance(r, dict) else None
            subgoals = _validate_plan(raw if isinstance(raw, list) else [],
                                      u, q, corpus)
        except Exception:
            subgoals = []
        if not subgoals:
            return {**state, "subgoals": [], "plan_gaps": [],
                    "trace": ["plan:invalid→agent"]}
        pool: dict[str, object] = {}
        ledger: list[dict] = []
        gaps: list[str] = []
        eff_terms: list[str] = []
        years = query_years(q)
        for sg in subgoals:
            entry = {**sg, "status": "pending", "reason": ""}
            period = sg["period"] or u.get("period") or ""
            scope = {d for d, doc in corpus.docs.items()
                     if (not sg["company"] or doc.company == sg["company"])
                     and (not period or period in doc.period)
                     and (not years or doc.period[:4] in years)} or None
            text = (f"{sg['metric']} 主要会计数据" if sg["metric"]
                    else "主要会计数据 营业收入 净利润")
            hits = corpus.search(expand_query(text), k=6,
                                 doc_filter=sg["company"], doc_scope=scope)
            if not _ckpt_ok(hits, sg["metric"]):
                # 有界重试 ×1（Phase 2）：扩词重查，非无界改写
                hits2 = corpus.search(
                    expand_query(text + " 同比 营业收入 净利润"), k=8,
                    doc_filter=sg["company"], doc_scope=scope)
                merged = {h.chunk.chunk_id: h for h in list(hits) + list(hits2)}
                hits = list(merged.values())
            if _ckpt_ok(hits, sg["metric"]):
                entry["status"] = "done"
                for h in hits:
                    if h.chunk.chunk_id not in pool:
                        pool[h.chunk.chunk_id] = h
                if sg["metric"] and sg["metric"] not in eff_terms:
                    eff_terms.append(sg["metric"])
            else:
                entry["status"] = "failed"
                entry["reason"] = "检索不足"
                gaps.append(f"{sg['company'] or '（公司未识别）'}"
                            f"{sg['period'] or ''}的{sg['metric'] or '指标'}")
            ledger.append(entry)
        if not pool:            # 全军覆没 → 回退 agent 环（不空手）
            return {**state, "subgoals": ledger, "plan_gaps": gaps,
                    "trace": ["plan:0命中→agent"]}
        # 证据按文档轮转（agent 环同款教训：单方大文档不能包揽前排）
        by_doc: dict[str, list] = {}
        for h in pool.values():
            by_doc.setdefault(h.chunk.doc_id, []).append(h)
        hits_out: list = []
        r_i = 0
        while len(hits_out) < TOP_K and any(len(v) > r_i for v in by_doc.values()):
            for chunks_ in by_doc.values():
                if r_i < len(chunks_) and len(hits_out) < TOP_K:
                    hits_out.append(chunks_[r_i])
            r_i += 1
        hits_out = stitch_adjacent(hits_out, corpus)
        hits_out = front_key_table(hits_out, u, q, corpus)
        eff = state.get("effective_query", q)
        for t in eff_terms:
            if t not in eff:
                eff = f"{eff} {t}"
        n_done = sum(1 for e in ledger if e["status"] == "done")
        n_fail = len(ledger) - n_done
        marker = f"plan:{len(ledger)}子目标/完成{n_done}" + \
                 (f"/弃{n_fail}(检索不足)" if n_fail else "")
        return {**state, "hits": hits_out, "subgoals": ledger,
                "plan_gaps": gaps, "effective_query": eff, "trace": [marker]}

    def _fact_key(state: GraphState) -> dict | None:
        """metric 键抽取（L1 前置条件）：单公司 + 单年份 + 期间后缀 +
        指标词命中才成键——键不完整时探针不开（宁可检索，不模糊命中）。"""
        u = state.get("understanding") or {}
        if u.get("intent") != "metric":
            return None
        comps = [c for c in (u.get("companies") or []) if c]
        if len(comps) != 1:
            return None
        # 年份：query 现抽优先，裸追问（「归母呢」）不带年份时用
        # L0 继承进 understanding 的年份——否则键建不起来，指标词
        # 随之无法传给下一轮（实测链式追问 metric 继承全断）
        years = sorted(query_years(state["query"])) or list(u.get("years") or [])
        if len(years) != 1:
            return None
        suffix = u.get("period") or ""
        if not suffix:
            return None
        mk = mem.metric_lexicon(state.get("effective_query", state["query"]))
        if not mk:
            return None
        return {"company": comps[0], "period": f"{years[0]}{suffix}",
                "metric": mk["metric"], "caliber": mk["caliber"]}

    def _memory_write(state: GraphState) -> list[str]:
        """L1 落账（verify:pass 之后调，写闸最后一道在这里）：trend 按
        fact_rows 逐行写；metric 按 fact_key 单行写（口径闸触发过的
        一律不写——D22 闸保护的是「数字在场」，写闸要的是「行也对」）。"""
        import datetime
        tr = "→".join(state.get("trace") or [])
        if any(bad in tr for bad in ("caliber-mismatch", "fallback(",
                                     "no-evidence", "derived-mismatch")):
            return []
        facts: list[dict] = []
        now = datetime.datetime.now().isoformat(timespec="seconds")
        for e in state.get("fact_rows") or []:
            comp = corpus.docs[e["doc_id"]].company \
                if e["doc_id"] in corpus.docs else ""
            if not comp:
                continue
            for rw in e["rows"]:
                metric, caliber = (("营收", "营业收入") if rw["label"] == "营收"
                                   else ("净利", "归母"))
                facts.append({
                    "company": comp, "period": e["period"],
                    "metric": metric, "caliber": caliber,
                    "value_raw": f"{rw['raw']:,.2f}", "value_std": rw["raw"],
                    "unit": "元", "yoy": rw.get("yoy") or "",
                    "citations": [{"doc_id": e["doc_id"], "page": e.get("page"),
                                   "quote": rw["seg"]}],
                    "coverage": state.get("coverage"),
                    "verdict": "pass", "created_at": now,
                    "source_doc_ids": [e["doc_id"]]})
        key = state.get("fact_key")
        if key:
            docs = [d for d, doc in corpus.docs.items()
                    if doc.company == key["company"]
                    and doc.period == key["period"]]
            if docs:
                hs = corpus.search(
                    "主要会计数据 " + expand_query("营业收入 净利润"),
                    k=5, doc_scope={docs[0]})
                chunks = [h.chunk for h in hs]
                get_page = getattr(corpus, "chunks_page", None)
                if get_page and hs and hs[0].chunk.page:
                    chunks += [c for c in get_page(docs[0], hs[0].chunk.page)
                               if c.chunk_id not in {x.chunk_id for x in chunks}]
                label = ("营收" if key["metric"] == "营收" else "归母净利")
                req = mem.CALIBER_LABEL.get(key["caliber"], key["caliber"])
                rows = metric_rows(chunks, cumulative=key["period"].endswith("Q3"),
                                   requires=((label, req),))
                if rows:
                    rw = rows[0]
                    facts.append({
                        "company": key["company"], "period": key["period"],
                        "metric": key["metric"], "caliber": key["caliber"],
                        "value_raw": f"{rw['raw']:,.2f}", "value_std": rw["raw"],
                        "unit": "元", "yoy": rw.get("yoy") or "",
                        "citations": [{"doc_id": docs[0],
                                       "page": chunks[0].page if chunks else 0,
                                       "quote": rw["seg"]}],
                        "coverage": state.get("coverage"),
                        "verdict": "pass", "created_at": now,
                        "source_doc_ids": [docs[0]]})
        ops = [mem.write_fact(corpus, f) for f in facts]
        ops = [o for o in ops if o != "skip"]
        if not ops:
            return [] if not facts else ["memory:write(skip)"]
        if set(ops) == {"noop"}:
            return ["memory:noop"]
        parts = [f"{o}×{ops.count(o)}" for o in sorted(set(ops))]
        return [f"memory:write({'+'.join(parts)})"]

    def verify_citations(state: GraphState) -> GraphState:
        a = state.get("answer", "")
        # 证据池 = 页级集合 ∪ hits（同页窗口块 citation 串相同，被引池
        # 自动扩到整页窗口）——跨块半句不再误降级；页窗口只装 hits[:4]
        # 的页而 extractive 从 hits[:8] 抽句，rank 5–8 的被引块（拼接行
        # 常排中段）必须也在池里，否则正确答案被误标 degraded
        pool = list(state.get("evidence") or []) + list(state.get("hits", []))
        verdict, cov = ans.verify(a, pool)
        extra: list[str] = []
        # 派生值出口对账（v0.7 Phase 4）：登记的派生数按证据池重算
        # （操作数必须真在场），|a−b−value| ≤ 0.01——对账闸门管派生值
        if verdict == "pass" and not _derived_ok(state.get("derived"), pool):
            verdict = "degraded"
            extra.append("verify:derived-mismatch→extractive")
        if verdict == "pass":
            out = {**state, "verdict": "pass", "coverage": cov,
                   "trace": [f"verify:pass({cov})"]}
        elif verdict == "refused":
            return refuse({**state, "coverage": cov,
                           "trace": ["verify:empty→refuse"]})
        else:
            # degraded：LLM 答案对不上账 → 降级为抽取式（永远有真凭实据）
            body, cited = ans.extractive_answer(state, state["hits"])
            if body:
                out = {**state, "answer": body, "citations": cited,
                       "verdict": "degraded", "coverage": cov,
                       "trace": [f"verify:degraded({cov})→extractive"] + extra}
            else:
                return refuse({**state, "coverage": cov,
                               "trace": [f"verify:degraded({cov})→refuse"] + extra})
        # 部分答案红线（v0.7 Phase 2）：Planner 弃掉的子目标显式声明，
        # 不允许静默吞掉缺口（DESIGN-v0.7 §4）
        gaps = [g for g in (state.get("plan_gaps") or []) if g]
        if gaps:
            out["answer"] = (f"{out.get('answer', '').rstrip()}\n（未覆盖："
                             + "；".join(gaps) + "——检索不足，以上为已覆盖部分）")
        # L1 写闸（v0.7 Phase 3）：verdict=pass ∧ 口径闸未触发 ∧ 非台账
        # 命中（命中重写=write_fact 的 noop，省一次）。读不豁免在
        # memory_probe，写有闸在此——写闸严于读闸（公理 4）
        if out.get("verdict") == "pass" and mem.enabled() \
                and not state.get("memory_hit"):
            marks = _memory_write(out)
            if marks:
                out["trace"] = marks
        return out

    # v0.7 新节点追加在尾部：旧索引（tests 里 nodes[0..11] 的位置访问）不动
    return (understand, route, retrieve, grade_docs, rewrite_query, generate,
            decompose, summarize_doc, refuse, verify_citations, agent_research,
            trend_research, memory_probe, plan)


# ── 条件边 ──────────────────────────────────────────────


def build_graph(corpus: Corpus):
    understand, route, retrieve, grade, rewrite, generate, decompose, summarize, refuse, verify, agent, trend, memory_probe, planner = make_nodes(corpus)

    g = StateGraph(GraphState)

    g.add_node("understand", understand)
    g.add_node("memory_probe", memory_probe)
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
    g.add_node("plan", planner)

    g.add_edge(START, "understand")
    # v0.7：understand 与 route 之间插 L1 探针——命中即终答（内联过
    # verify），未命中零开销透传。eval 隔离（DZ_MEMORY=off）时恒透传
    g.add_edge("understand", "memory_probe")

    def after_memory(state: GraphState) -> str:
        return END if state.get("memory_hit") else "route"

    g.add_conditional_edges("memory_probe", after_memory,
                            {"route": "route", END: END})

    def after_route(state: GraphState) -> str:
        # 期间守卫已在 route 内写明具体原因（缺哪年、现有哪些期间），
        # 直达 END——此前 other→refuse 会用通用文案覆盖掉它
        if state.get("verdict") == "refused":
            return END
        r = state.get("route", "metric")
        # v0.6：llm 模式下 metric/compare 进 Agent 环（模型自主检索）；
        # heuristic 模式走固定管线（兜底 + 回归基线）。
        # v0.7 Phase 2：多跳信号（≥3 家/跨年/≥2 指标族）先进 Planner
        if state.get("mode") == "llm" and r in ("metric", "compare"):
            if _wants_plan(state["query"], state.get("understanding") or {}):
                return "plan"
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
                             "plan": "plan",
                             "trend_research": "trend_research",
                             "refuse": "refuse",
                             END: END})

    # v0.7 Phase 2：Planner 校验失败/零命中 → 回退 Agent 环（现路径）
    def after_plan(state: GraphState) -> str:
        return "generate" if state.get("subgoals") else "agent_research"

    g.add_conditional_edges("plan", after_plan,
                            {"generate": "generate",
                             "agent_research": "agent_research"})

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


def ask(corpus: Corpus, query: str, mode: str = "auto",
        history: list[dict] | None = None) -> GraphState:
    """一条命令问答：建图 → stream(values) 逐节点计时（P1 观测性：
    -v/--json 的 trace 每节点带耗时）。history：L0 会话记忆（chat 传入）。"""
    app = build_graph(corpus)
    import time

    from duizhang.schemas import new_state
    st: GraphState | None = None
    marks: list[float] = []
    t0 = time.perf_counter()
    for snap in app.stream(new_state(query, mode, history), stream_mode="values"):
        st = snap
        marks.append(time.perf_counter())
    if st is None:                        # stream 不可用 → invoke 兜底
        st = app.invoke(new_state(query, mode, history))
    if len(marks) >= 2:
        st["timing"] = [round((marks[i] - marks[i - 1]) * 1000)
                        for i in range(1, len(marks))]
    st["elapsed"] = round(time.perf_counter() - t0, 2)
    return st
