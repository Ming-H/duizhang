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


def extractive_answer(state: GraphState, hits: list[Retrieved],
                      max_sents: int = 4) -> tuple[str, list[str]]:
    """从 top 命中块抽证据句：与 query 的内容词重叠（元词不计分）+ 领域加权。
    打分词先经 expand_query：口语词（营收/revenue）与证据术语
    （营业总收入）零词元重叠会空手拒答（heuristic innolight 实测）。"""
    q = _content_tokens(word_tokens(expand_query(state["effective_query"])))
    # 问「原因/变动」时给归因句加权：财报的因果句式是「主要系…影响」「主要由于…所致」
    wants_reason = any(w in state["query"] for w in ("原因", "为什么", "为何", "变动"))
    _REASON_PAT = re.compile(r"主要系|主要因|主要由于|所致|影响")
    # PDF 断词工件注入：主要会计数据表的标签行（无数字表块）被空表头
    # 过滤丢弃后，数字行以断词残片开头——「润（元）…」实为「净利润」、
    # 「入（元）…」实为「营业收入/收入」。不注入则 query 的完整词与
    # 证据零重叠，真值行落选（实测 2025Q1 归母净利行）
    _SPLIT_FIX = (("润（元）", "净利润"), ("入（元）", "营业收入"),
                  ("额（元）", "总资产额"))

    def _inject(text: str) -> str:
        for frag, full in _SPLIT_FIX:
            if text.lstrip().startswith(frag):
                return full + " " + text
        return text

    scored = []
    for h in hits[:8]:   # 与 TOP_K 对齐：表格大块常占前排，答案块在 6 名开外
        company = h.chunk.doc_id.split("_", 1)[0]   # doc_id = 公司_报告期
        for s in _sentences(_inject(h.chunk.text)):
            stoks = _content_tokens(set(word_tokens(s)) | set(word_tokens(h.chunk.text[:40])),
                                    extra_meta=(company,), meta_chars=frozenset(company))
            overlap = len(q & stoks)
            # 数字句优先（财报问答大多数时候在问数字）
            has_num = bool(re.search(r"\d", s)) or bool(h.chunk.num_notes)
            score = overlap * (1.4 if has_num else 1.0) + (0.2 if h.chunk.num_notes else 0.0)
            # 数字问句（「多少」）下，无数字句强降权：表头行（「归属于
            # 上市公司股东的净利润（元）」）词面完美匹配扩展词却无值，
            # 会压过真值行（扩展词引入后实测）
            if not has_num and any(w in state["query"] for w in ("多少", "几")):
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

    cited = [h for h in hits if _chunk(h).citation in answer]
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
    期间后缀），正则永远漏判。"""
    if re.search(r"\[[^\]]+第\d+页\]|\[[^\]]+_\d{4}[A-Z0-9]*\]", answer):
        return True
    return bool(hits) and any(
        (h.chunk if hasattr(h, "chunk") else h).citation in answer for h in hits)


COVERAGE_THRESHOLD = 0.45   # 抽取式天然高；LLM 生成低于此值说明在编


def verify(answer: str, hits: list[Retrieved]) -> tuple[str, float]:
    """返回 (verdict, coverage)：pass / degraded（降级重写为抽取式由图负责）。"""
    if not answer.strip():
        return "refused", 0.0
    cov = citation_coverage(answer, hits)
    if cov >= COVERAGE_THRESHOLD and has_citation(answer, hits):
        return "pass", round(cov, 3)
    return "degraded", round(cov, 3)
