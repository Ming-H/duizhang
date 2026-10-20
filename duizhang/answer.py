"""生成与对账校验。

extractive：零 LLM 的抽取式作答——从命中块里挑证据句拼装，
每句挂引用。演示永不翻车的底气所在。
verify：引用覆盖率校验——答案句与被引块的 token 重叠率低于阈值
即降级/拒答。「答案必须能对账」的最后一道闸。
"""

from __future__ import annotations

import re

from duizhang.index import word_tokens
from duizhang.ingest import expand_query
from duizhang.schemas import GraphState, Retrieved

_SENT_SPLIT = re.compile(r"(?<=[。；！？])\s*")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) >= 6]


# 文档元词：财报里每句都带（公司/报告期/年份），作为重叠信号只有噪声
_META_WORDS = {"公司", "股份", "报告", "年度", "半年度", "季度", "一季", "三季",
               "情况", "说明", "有限公司", "证券", "代码", "简称", "公告", "编号"}


def _content_tokens(toks, extra_meta: tuple[str, ...] = (),
                    meta_chars: frozenset[str] = frozenset()) -> set:
    """元词不计分。extra_meta（如公司名整串）按子串匹配；meta_chars
    （公司名的字符集）按「token 全部字符都在其中」匹配——jieba 会把
    公司名切碎（海伦哲→海伦+哲），整串匹配拦不住碎片。页眉句
    「公司名+报告标题」靠 query 里的公司名/期间得分，事实句反而落选
    （全量库实测）。元词做子串匹配：第一季度⊃季度、年度报告⊃报告。"""
    out = set()
    for t in toks:
        if t in _META_WORDS or re.fullmatch(r"20\d{2}", t):
            continue
        if any(m and m in t for m in extra_meta):
            continue
        if any(m in t for m in _META_WORDS):
            continue
        if meta_chars and set(t) <= meta_chars:
            continue
        out.add(t)
    return out


# PDF 断词工件修复表（模块级共用）：表格跨块断词让标签头丢失——
# 数字行以残片开头（「润（元）…」=净利润、「入（元）…」=营业收入、
# 「者权益（元）…」=归属于上市公司股东的净资产、「常性损益的净利
# 润」=扣除非经常性损益的净利润，标签「扣除非经|常性损益」被切走）。
# 不修复则 query 完整词与证据零重叠（真值行落选），口径镜像也无从
# 匹配（0.8b 看到无标签残行，拿归母数答扣非题）。graph.trend_skeleton
# 的 pick() 与 graph.generate 的 LLM 证据组装维护同族映射，三处同步
SPLIT_FIX = (("润（元）", "净利润"), ("入（元）", "营业收入"),
             ("额（元）", "总资产额"), ("者权益（元）", "归属于上市公司股东的净资产"),
             ("常性损益的净利润", "扣除非经常性损益的净利润"))


def split_fix(text: str) -> str:
    """行首断词残片补全为完整指标名（证据侧行级修复，见 SPLIT_FIX）。"""
    for frag, full in SPLIT_FIX:
        if text.lstrip().startswith(frag):
            return full + " " + text
    return text


def extractive_answer(state: GraphState, hits: list[Retrieved],
                      max_sents: int = 4) -> tuple[str, list[str]]:
    """从 top 命中块抽证据句：与 query 的内容词重叠（元词不计分）+ 领域加权。
    打分词先经 expand_query：口语词（营收/revenue）与证据术语
    （营业总收入）零词元重叠会空手拒答（heuristic innolight 实测）。"""
    q = _content_tokens(word_tokens(expand_query(state["effective_query"])))
    # 问「原因/变动」时给归因句加权：财报的因果句式是「主要系…影响」「主要由于…所致」
    wants_reason = any(w in state["query"] for w in ("原因", "为什么", "为何", "变动"))
    _REASON_PAT = re.compile(r"主要系|主要因|主要由于|所致|影响")

    def _inject(text: str) -> str:
        return split_fix(text)

    scored = []
    # 抽取窗 12（原 8 与 TOP_K 对齐）：D14 前置会把主要会计数据表页
    # 8+ 块顶到队头，利润表/附注的真值行被挤到 rank 9+——问「营业
    # 总收入」的真值在利润表页（茅台 2026H1 p30 实测 rank10 落选）
    for h in hits[:12]:
        company = h.chunk.doc_id.split("_", 1)[0]   # doc_id = 公司_报告期
        for s in _sentences(_inject(h.chunk.text)):
            # 口径守卫（D16 同款，与 trend pick() 一致）：问句未提「扣」
            # 时扣非口径行不作候选——rewrite 给 effective_query 追加的
            # 通用词（净利润/扣非净利润）会让拼接后的扣非行词面全中，
            # 实测顶掉营收行（25 年年报营收答成扣非 10,710,053,246.51）
            if ("扣除" in s or "扣非" in s) and "扣" not in state["query"]:
                continue
            stoks = _content_tokens(set(word_tokens(s)) | set(word_tokens(h.chunk.text[:40])),
                                    extra_meta=(company,), meta_chars=frozenset(company))
            overlap = len(q & stoks)
            if overlap == 0:
                continue          # 零重叠句只靠 num_notes 加分混进答案（实测净资产句）
            # 数字句优先（财报问答大多数时候在问数字）
            has_num = bool(re.search(r"\d", s)) or bool(h.chunk.num_notes)
            score = overlap * (1.4 if has_num else 1.0) + (0.2 if h.chunk.num_notes else 0.0)
            # 数字问句（「多少」）下，无数字句强降权：表头行（「归属于
            # 上市公司股东的净利润（元）」）词面完美匹配扩展词却无值，
            # 会压过真值行（扩展词引入后实测）
            if not has_num and any(w in state["query"] for w in ("多少", "几")):
                score *= 0.3
            # 指标口径配对降权（D16 家族，P0 泛化实测）：问句点名指标 X
            # 时，只谈同胞指标 Y 的句子与 X 共享标签前缀词（归属于上市
            # 公司股东的…），打分虚高冒名顶替。三对（降权不跳过——双
            # 口径问句「扣非和归母差多少」仍要看到对方）：
            # ① 问扣非：无「扣除/扣非」的净利润句是归母口径（茅台
            #    2026H1 实测 445.17 亿散文句顶掉 44,464,207,646.01）
            # ② 问净利润：净资产句共享前缀词（实测净资产块 4.4 分把
            #    扣非行 1.6 分挤出前四，44,464 落选）
            # ③ 问总收入：营业收入句是子口径不是总口径（真值
            #    92,278,072,083.21 被营业收入行冒充，2026H1 p30）
            q_raw = state["query"]
            if (("扣非" in q_raw or "扣除非" in q_raw)
                    and "净利润" in s and "扣除" not in s and "扣非" not in s):
                score *= 0.3
            if ("净利润" in q_raw or "净利" in q_raw) and "净资产" not in q_raw \
                    and "总资产" not in q_raw and "净资产" in s and "净利润" not in s:
                score *= 0.3
            if "总收入" in q_raw and "营业收入" in s and "总收入" not in s:
                score *= 0.3
            # ④ 问总资产：净利润族句共享「归属于…的」前缀词族（expand
            #    的净资产→所有者权益 token 给它 2 重叠，实测顶掉 1 重叠
            #    的总资产真值行，innolight 30/31 回退）
            if ("总资产" in q_raw and "净利" not in q_raw
                    and "净利润" in s and "总资产" not in s):
                score *= 0.3
            # ⑤ 问总资产且未问收入：利润表头句吃 rewrite 追加词+expand
            #    收入族映射的三重叠（实测 4.2 分顶掉总资产行 1.6 分）
            if ("总资产" in q_raw and "收入" not in q_raw and "营收" not in q_raw
                    and ("营业收入" in s or "总收入" in s) and "总资产" not in s):
                score *= 0.3
            if wants_reason and _REASON_PAT.search(s):
                score += 1.5
            scored.append((score, s, h.chunk))
    scored.sort(key=lambda x: -x[0])
    picked, cited, picked_chunks = [], [], []
    for score, s, chunk in scored:
        if score <= 0:
            continue
        picked.append(f"{s} {chunk.citation}")
        if chunk.citation not in cited:
            cited.append(chunk.citation)
        picked_chunks.append(chunk)
        if len(picked) >= max_sents:
            break
    if not picked:
        return "", []
    # 数字注释跟随被选中的块（关键数字句常在第 2 名开外，固定取 top1
    # 会让「≈4.19亿」类换算缺席——它正是「多少亿」问法的命中通道）
    notes: list[str] = []
    for chunk in picked_chunks:
        for n in chunk.num_notes:
            if n not in notes:
                notes.append(n)
    note_str = "；".join(notes[:2])
    body = "\n".join(f"· {p}" for p in picked)
    if note_str:
        body = f"{body}\n（数字参考：{note_str}）"
    return body, cited


# ── 对账校验 ────────────────────────────────────────────


def _cite_norm(s: str) -> str:
    """引用匹配的空白归一（D13）：小模型手写引用的空格会漂移
    （「第 8 页」「doc_id第8页」粘连），精确子串匹配不认 → 有据答案
    全 degraded、compare 结论句被抽取式顶掉（2026-10-19 llm 实测
    cov=0.8 仍 degraded）。剥尽空白再比，引用语义（doc+页）不变。"""
    return re.sub(r"\s+", "", s)


def citation_in(answer: str, citation: str) -> bool:
    """答案里是否出现该引用（空白不敏感，D13 引用格式解耦）。"""
    return _cite_norm(citation) in _cite_norm(answer)


def citation_coverage(answer: str, hits: list[Retrieved]) -> float:
    """答案句与证据池的 token 重叠率均值。

    证据池 = 答案中实际引用的块（citation 串出现在答案里）；答案没有
    可识别引用时回退 hits[:4]（LLM 生成路径只看过前 4 块）。
    抽取式从 hits[:8] 抽句（答案块常在第 4 名开外），第 5–8 名块支撑
    的句子若不计入证据池，自家答案会被自家 verify 误判 degraded。

    两侧 token 均经 _content_tokens（公司名/年份/期间元词不计）——
    与 extractive 打分同一口径：答案说「欧科亿 2026 年」不构成证据
    相关性，数字与指标词才是。证据含数字归一化注释（LLM 常以
    「6.44亿」作答而非原文「644,051,473.82」）。

    只考核数字句：财报问答的编造风险集中在数字句，框架句
    （「根据提供的证据…」「因此…」）不该被要求与证据重叠——
    LLM 对比答案常因行文结构被误杀。年份（2026）不算数字证据，
    否则「2026 年第一季度对比如下」这类框架句仍会被考核。
    全答案无数字句时回退全句。
    """
    # 句界含换行：LLM 引用行常以换行而非句号收尾（句号在引用后），
    # 只按标点切会把整段答案并成一句，覆盖率失真
    body = re.sub(r"\[[^\]]*\]", "", answer)
    all_sents = [s.strip() for s in re.split(r"(?<=[。；！？])\s*|\n+", body)
                 if len(s.strip()) >= 6]
    if not all_sents:
        return 0.0
    sents = [s for s in all_sents if re.search(r"\d", re.sub(r"20\d{2}", "", s))] \
        or all_sents
    # 兼容 Retrieved 与裸 Chunk（v0.6.1 页级证据是 Chunk 列表）
    def _chunk(h):
        return h.chunk if hasattr(h, "chunk") else h

    cited = [h for h in hits if citation_in(answer, _chunk(h).citation)]
    pool = cited or hits[:4]
    companies = tuple(dict.fromkeys(
        _chunk(h).doc_id.split("_", 1)[0] for h in pool))
    mchars = frozenset("".join(companies))
    evidence = " ".join(f"{_chunk(h).text} {' '.join(_chunk(h).num_notes)}"
                        for h in pool)
    etoks = _content_tokens(word_tokens(evidence), extra_meta=companies,
                            meta_chars=mchars)
    cov = []
    for s in sents:
        stoks = _content_tokens(word_tokens(s), extra_meta=companies,
                                meta_chars=mchars)
        if not stoks:
            continue
        cov.append(len(stoks & etoks) / len(stoks))
    return sum(cov) / len(cov) if cov else 0.0


def has_citation(answer: str, hits: list[Retrieved] | None = None) -> bool:
    """答案是否带引用：正则覆盖 PDF 块与 LLM 手写格式；hits 兜底
    MD/TXT 块——page=0 的引用是 [doc_id]（无「第N页」、无 `_\d{4}`
    期间后缀），正则永远漏判。正则在空白归一后的答案上跑：小模型
    手写引用的空格漂移（第 8 页）不能杀掉有据答案（D13）。"""
    if re.search(r"\[[^\]]+第\d+页\]|\[[^\]]+_\d{4}[A-Z0-9]*\]", _cite_norm(answer)):
        return True
    return bool(hits) and any(
        citation_in(answer, (h.chunk if hasattr(h, "chunk") else h).citation)
        for h in hits)


COVERAGE_THRESHOLD = 0.45   # 抽取式天然高；LLM 生成低于此值说明在编


def verify(answer: str, hits: list[Retrieved]) -> tuple[str, float]:
    """返回 (verdict, coverage)：pass / degraded（降级重写为抽取式由图负责）。"""
    if not answer.strip():
        return "refused", 0.0
    cov = citation_coverage(answer, hits)
    if cov >= COVERAGE_THRESHOLD and has_citation(answer, hits):
        return "pass", round(cov, 3)
    return "degraded", round(cov, 3)
