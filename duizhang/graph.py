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
    # （总收入题 0.8b 只见 p5 营业收入行，实测）。扫到凑满 top_n 个
    # 不同页为止，总量仍受 cap 硬顶。窗仍锚各页首个命中 ±2——同页
    # 全命中并窗实测净负（0.8b 上下文稀释，0.778 vs 0.852，D14 cap
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
    表头占 #1#2），0.8b 从分季度表抓单季数当全年。前置后 LLM 证据头部
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
    years = query_years(query)
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
    时，答案的大数必须出现在该口径标签行的数字集里——0.8b 实测拿归母
    44,516,880,421.86 答扣非题、营业收入 90,703,260,964.48 答总收入
    题：数字都在证据页，verify 引用级照样 pass（gate≠correctness，
    §9 边界的第一处收口——引用级对账之上加口径级数值对账）。证据里
    没有该口径行 → False（检索缺口不归闸门背锅，交给降级抽取式）。
    """
    if ("扣非" not in query and "扣除非" not in query and "总收入" not in query
            and "净资产" not in query and "总资产" not in query):
        return False
    ans_nums = _line_nums(answer)
    if not ans_nums:
        return False
    # 逐行修复断词残片（块级修复只看块首，多行块的后续行残片会漏）
    lines = [ans.split_fix(ln) for c in chunks for ln in c.text.splitlines()]
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
    for good in good_sets:
        if good and not (ans_nums & good):
            return True
    return False


def _parse_metric_row(text: str) -> tuple[str, str] | None:
    """主要会计数据行 → (本期亿值, 同比%)。本期取第一个非百分数、
    ≥1e6 的大数（防「4.842 546」这类残缺数字误入）；同比取最后的
    百分比。解析不出返回 None（该指标省略，不打印错数）。"""
    nums = re.findall(r"\d[\d,]*(?:\.\d+)?%?", text)
    pct = None
    for n in nums:
        if n.endswith("%"):
            pct = n
    vals = []
    for n in nums:
        if not n.endswith("%"):
            try:
                v = float(n.replace(",", ""))
            except ValueError:
                continue
            if v >= 1e6:
                vals.append(v)
    if not vals:
        return None
    yi = vals[0] / 1e8
    return f"{yi:,.1f} 亿", (pct.rstrip("%") + "%") if pct else ""


def trend_skeleton(hits, corpus=None) -> str:
    """趋势数据层（确定性）：每期解析为结构化行
    「· 期间：营收 X 亿（同比 Y）；归母净利 Z 亿（同比 W）[引用]」。
    原始表格行（本期/同期/同比混排）可读性差（用户实测「很乱」），
    且 0.8b 需要干净输入才能做分析——数据层干净了分析层才放得开。
    断词工件（「润（元）…」=净利润行）在此同样注入。"""
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

        def pick(require: str) -> str | None:
            cands: list[tuple[int, str]] = []
            for c in pool:
                for raw in c.text.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    for frag, full in (("润（元）", "净利润"), ("入（元）", "营业收入"),
                                       ("者权益（元）", "归属于上市公司股东的净资产"),
                                       ("常性损益的净利润", "扣除非经常性损益的净利润")):
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
                    if not re.search(r"\d", line):
                        continue
                    # D21 行级取数：表块无标点，句级切分整块一句，
                    # pick(净利润) 与 pick(营业收入) 命中同一句 → 归母
                    # 取到营收值（茅台 2025Q1 实测 506.0/506.0 同值、
                    # verify 满分放行）。行内从指标词起截，取其后首个
                    # 大数——同行多口径混排也各取各的。
                    seg = line[idx:]
                    score = 2
                    if "%" in seg:
                        score += 2
                    if any(w in line for w in ("适用", "不适用", "差异", "终止经营",
                                                "少数股东", "会计准则")):
                        score -= 6          # 段落噪声（2026H1 会计准则段实测）
                    cands.append((score, seg))
            # 名次回退（D21 补面）：打分最高的行解析不过时（MD&A 散文
            # 「454.03 亿元」带 % 赢表行、却过不了 ≥1e6 原币门槛），按
            # 分数降序取第一个可解析行——只取最佳会让该指标整行丢失
            # （茅台 2025H1/2026H1 归母缺失实测根因）。
            for score, seg in sorted(cands, key=lambda x: -x[0]):
                if score <= 0:
                    break
                if _parse_metric_row(seg):
                    return seg
            return None

        parts = []
        for label, key in (("营收", "营业收入"), ("归母净利", "净利润")):
            row = pick(key)
            if row:
                parsed = _parse_metric_row(row)
                if parsed:
                    yi, pct = parsed
                    parts.append(f"{label} {yi}（同比 {pct}）" if pct
                                 else f"{label} {yi}")
        if parts:
            period_tag = doc_id.rsplit("_", 1)[-1][-2:]
            basis = {"H1": "半年累计", "Q1": "单季", "Q3": "前三季累计",
                     "FY": "全年"}.get(period_tag, "")
            basis_note = f"（{basis}）" if basis else ""
            lines.append(f"· {doc_id}{basis_note}：" + "；".join(parts) +
                         f" {pool[0].citation}")
    return "\n".join(lines)


def _analysis_numbers_ok(analysis: str, rows: str) -> bool:
    """投资分析数字闸门：分析中每个数字（年份除外）必须出现在数据行里
    ——表外数字=幻觉，整个分析段丢弃（宁缺毋滥，但不再一刀切禁数字：
    引用表内数字的分析是允许且需要的）。"""
    allowed = set(re.findall(r"[\d,]*\d(?:\.\d+)?%?", rows))
    allowed |= {a.rstrip("%") for a in allowed}
    for n in re.findall(r"[\d,]*\d(?:\.\d+)?%?", analysis):
        if re.fullmatch(r"20\d{2}", n):
            continue
        if n not in allowed and n.rstrip("%") not in allowed:
            return False
    return True


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
            # 趋势守卫：0.8b 实测把趋势问句判成 summary（单文档残渣）
            if u.get("intent") == "summary" and any(w in q for w in _TREND_WORDS):
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
            # trend 必要条件：query 含趋势词（0.8b 会把「预计…多少」
            # 判成 trend 并在 reason 里幻觉数字，实测）。降级目标用规则
            # 意图而非硬编码 metric——同一 query 三次返回三种意图，直接
            # 降 metric 会让 compare 题漏判（samples 哪家题实测）
            if u.get("intent") == "trend" and not any(w in q for w in _TREND_WORDS):
                u["intent"] = rule_u["intent"] if rule_u["intent"] != "trend" else "metric"
            # 股价/涨跌类守卫：0.8b 偶发判 metric 拿净利表头行回答股价
            # 问题（实测），语料是财报不含行情——归 other 拒答
            if u.get("intent") == "metric" and any(
                    w in q for w in ("股价", "会涨", "会跌", "涨停", "跌停")):
                u["intent"] = "other"
            # 管理层叙述守卫：「管理层怎么说/怎么讲」是问 MD&A 叙述内容，
            # 0.8b 偶发判 other 拒答（实测）——有报告词在场时归 summary
            if (u.get("intent") == "other"
                    and any(w in q for w in ("管理层", "董事长", "怎么说的", "怎么讲"))
                    and any(w in q for w in ("中报", "年报", "季报", "报告", "财报"))):
                u["intent"] = "summary"
        return {**state, "understanding": u, "trace": trace}

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
            years = query_years(state["query"])
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
        # 趋势题：确定性骨架 + LLM 仅做归纳（0.8b 直生成多期长答案
        # 实测自相矛盾——见 trend_skeleton 注释）。骨架纯代码，模式无关
        # ——heuristic 同样输出逐期骨架（此前被 llm 条件包裹退化成
        # 普通抽取式，doc=N 实测）
        if (state.get("understanding") or {}).get("intent") == "trend":
            skeleton = trend_skeleton(hits, corpus)
            if not skeleton:
                return {**state, "answer": "", "verdict": "refused",
                        "trace": ["generate:trend-no-facts→refuse"]}
            analysis = ""
            # 投资分析默认关（DZ_TREND_ANALYSIS=enabled 开启）：0.8b 三组
            # 实测语义级不可靠——数字闸门全过（数字来自表格）但仍把
            # +90.05% 说成「大幅回落」、+69.40% 说成「暴跌」、累计/单季
            # 混淆——数字对≠结论对，语义闸门无法机械化，只能按模型分档
            if (state["mode"] == "llm" and llmx.llm_enabled()
                    and os.environ.get("DZ_TREND_ANALYSIS") == "enabled"):
                try:
                    analysis = llmx.chat_text(
                        "你是财报分析师。基于给定逐期数据做投资分析（趋势"
                        "判断/盈利质量/风险点）。铁律：①H1 是半年累计、Q1 "
                        "是单季——绝对值不可直接比较，跨期比较只能用同比"
                        "增速；②只准使用给定数据里的数字，禁止任何其他数字"
                        "（含换算与预测）；③禁止编造数据外的归因（产能/"
                        "需求/竞争等未被数据提及的因素）。三到五句话。",
                        skeleton, max_tokens=768).strip()
                except Exception:
                    pass
                # 数字闸门：分析里的数字必须全部来自数据行——表外数字
                # =幻觉，整个分析段丢弃（宁缺毋滥；引用表内数字允许）
                if analysis and not _analysis_numbers_ok(analysis, skeleton):
                    analysis = ""
            body = skeleton
            if analysis:
                body = f"{body}\n\n分析：{analysis}"
            body += "\n（以上为财报事实梳理，不构成投资建议）"
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
            # 产物（数字闸门同源），本身即合法证据
            from duizhang.schemas import Chunk as _C
            for h in hits[:6]:
                ev.append(_C(chunk_id=f"sk#{h.chunk.doc_id}", doc_id=h.chunk.doc_id,
                             page=h.chunk.page, seq=0, text=skeleton))
            return {**state, "answer": body, "citations": cited,
                    "evidence": ev or hits, "trace": ["generate:trend"]}
        if state["mode"] == "llm" and llmx.llm_enabled():
            # 证据按页聚合（页首标注），块序拼接；数字归一化注释随块；
            # 断词残行先经 split_fix 补全标签——0.8b 看到「常性损益的
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
                    "（0.8b 实测拿 1,582,876,128.67 答扣非题，真值"
                    " 1,568,385,287.19）。问「总收入/营业总收入」时必须取"
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
            except Exception:  # LLM 故障 → 走抽取式，绝不空手而归（留痕）
                fallback = ["generate:llm→fallback"]
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
        # 证据池 = 页级集合 ∪ hits（同页窗口块 citation 串相同，被引池
        # 自动扩到整页窗口）——跨块半句不再误降级；页窗口只装 hits[:4]
        # 的页而 extractive 从 hits[:8] 抽句，rank 5–8 的被引块（拼接行
        # 常排中段）必须也在池里，否则正确答案被误标 degraded
        pool = list(state.get("evidence") or []) + list(state.get("hits", []))
        verdict, cov = ans.verify(a, pool)
        if verdict == "pass":
            return {**state, "verdict": "pass", "coverage": cov,
                    "trace": [f"verify:pass({cov})"]}
        if verdict == "refused":
            return refuse({**state, "coverage": cov,
                           "trace": ["verify:empty→refuse"]})
        # degraded：LLM 答案对不上账 → 降级为抽取式（永远有真凭实据）
        body, cited = ans.extractive_answer(state, state["hits"])
        if body:
            return {**state, "answer": body, "citations": cited, "verdict": "degraded",
                    "coverage": cov,
                    "trace": [f"verify:degraded({cov})→extractive"]}
        return refuse({**state, "coverage": cov,
                       "trace": [f"verify:degraded({cov})→refuse"]})

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
