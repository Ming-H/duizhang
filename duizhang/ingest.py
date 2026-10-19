"""摄取：PDF/MD → 带元数据的 Doc → 表格感知分块的 Chunk 列表。

三条设计决策：
1. 元数据白捡——巨潮财报命名「公司_报告期类型.pdf」自带公司与期间，
   解析文件名而非让 LLM 猜。
2. 表格感知分块——财报的核心事实住在表格里；纯字符切分把表格拦腰
   切断等于销毁证据。连续短行（无句号、含多空格/数字密集）聚成
   表格原子块，永不跨块。
3. 数字归一化注释——检索对「XX 亿」类问法天然失灵，归一化注释让
   词法检索直接命中数字量级（句读 judou 的同款换算，纯程序零幻觉）。
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from duizhang.schemas import Chunk, Doc

# 研报：20260703-东吴证券-并行科技-920493-标题.pdf → (券商, 日期, 研报)
_REPORT_PAT = re.compile(r"^(\d{8})-([^-]+)-(.+)\.pdf$")

# 文件名 → (period, kind)
_PERIOD_PAT = [
    (re.compile(r"(\d{4})年年度报告"), "FY"),
    (re.compile(r"(\d{4})年半年度报告"), "H1"),
    (re.compile(r"(\d{4})年第一季度报告"), "Q1"),
    (re.compile(r"(\d{4})年第三季度报告"), "Q3"),
    (re.compile(r"(\d{4})年半年度业绩预告"), "H1E"),
    (re.compile(r"(\d{4})年第三季度业绩预告"), "Q3E"),
]

_KIND_LABEL = {"FY": "年报", "H1": "半年报", "Q1": "一季报", "Q3": "三季报",
               "H1E": "半年度业绩预告", "Q3E": "三季度业绩预告",
               "ANALYSIS": "分析", "研报": "研报"}

# 指标同义词：查询扩展用（检索前把口语指标名映射到财报术语）
METRIC_SYNONYMS = {
    "营收": ["营业收入", "营业总收入", "主营收入"],
    "收入": ["营业收入", "营业总收入"],
    "利润": ["净利润", "归属于上市公司股东的净利润", "扣非净利润"],
    "净利": ["净利润", "归属于上市公司股东的净利润"],
    "毛利": ["毛利率", "营业成本"],
    "市值": ["总市值", "市值"],
    "在手": ["手持"],
    "赚": ["净利润", "归属于母公司所有者的净利润"],
    "盈利": ["净利润", "归属于母公司所有者的净利润"],
    "eps": ["基本每股收益", "每股收益"],
    "revenue": ["营业收入", "营业总收入"],
    "profit": ["净利润", "归属于上市公司股东的净利润"],
    "income": ["营业收入", "营业总收入"],
    "roe": ["加权平均净资产收益率", "净资产收益率"],
}


def parse_filename(path: str | Path) -> tuple[str, str, str]:
    """从文件名解析 (company, period, kind)。

    三类通道：① 财报（公司_报告期类型）② 券商研报（日期-券商-标题）
    ③ 兜底（company=文件名主干，kind=ANALYSIS/文本）。
    """
    name = Path(path).stem
    m = _REPORT_PAT.match(Path(path).name)
    if m:
        date, broker, _title = m.group(1), m.group(2), m.group(3)
        return broker, date, "研报"
    for pat, tag in _PERIOD_PAT:
        m = pat.search(name)
        if m:
            company = pat.sub("", name).rstrip("_－- ")
            return company, f"{m.group(1)}{tag}", _KIND_LABEL[tag]
    # 兜底：公司_分析 / 裸 txt。主干必须是可读公司名（含 CJK）——
    # 日期开头的外部分析（20261009_永和…_CANSLIM分析.md）主干是纯
    # 数字，静默入库会产出 20261009_MD 这类垃圾 doc_id（污染检索与
    # 引用）；拒收让 cli 以 ⚠ 跳过（P0.5 实测库内留过一具）
    company = re.split(r"[__－-]", name)[0]
    if not re.search(r"[一-鿿]", company):
        raise ValueError(f"文件名解析不出公司名（主干={company!r}），拒收")
    return company, "TXT" if Path(path).suffix.lower() == ".txt" else "MD", "ANALYSIS"


# ── 数字归一化（句读基因） ──────────────────────────────

_NUM_RE = re.compile(r"([1-9]\d{0,2}(?:,\d{3})+(?:\.\d+)?)\s*(亿元|万元|元)")


def normalize_numbers(text: str) -> list[str]:
    """把带千分位的大数统一换算成「亿」可读注释（句读基因，纯程序零幻觉）。"""
    notes = []
    for m in _NUM_RE.finditer(text):
        raw = m.group(1).replace(",", "")
        unit = m.group(2)
        try:
            v = float(raw)
        except ValueError:
            continue
        if unit == "亿元":
            notes.append(f"≈{v:,.2f}亿")
        elif unit == "万元":
            yi = v / 1e4
            notes.append(f"≈{yi:,.2f}亿" if yi >= 1 else f"≈{v:,.2f}万")
        else:  # 元
            yi, wan = v / 1e8, v / 1e4
            if yi >= 1:
                notes.append(f"≈{yi:,.2f}亿")
            elif wan >= 1:
                notes.append(f"≈{wan:,.2f}万")
    return notes[:6]  # 每块至多 6 条，防噪声


# ── 表格感知分块 ────────────────────────────────────────

_TABLE_HINT = re.compile(r"[\d%．.]")           # 数字密度
_SENT_END = re.compile(r"[。；！？；]")


def _is_table_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _SENT_END.search(s):
        return False
    digits = len(_TABLE_HINT.findall(s))
    return digits >= max(2, len(s) * 0.25) or ("  " in s and digits >= 2)


def chunk_page(text: str, max_chars: int = 420) -> list[tuple[str, bool]]:
    """单页 → [(块文本, 是否表格)]。表格行聚簇；正文按段落边界切。"""
    lines = [ln.strip() for ln in text.splitlines()]
    out, para, table = [], [], []

    def flush_para():
        if para:
            buf = ""
            for ln in para:
                if len(buf) + len(ln) + 1 > max_chars and buf:
                    out.append((buf, False))
                    buf = ln
                else:
                    buf = f"{buf}{ln}" if not buf else f"{buf} {ln}"
            if buf:
                out.append((buf, False))
            para.clear()

    def flush_table():
        if table:
            out.append((" ".join(table), True))
            table.clear()

    for ln in lines:
        if not ln:
            flush_table()
            flush_para()
            continue
        if _is_table_line(ln):
            flush_para()
            table.append(ln)
            # 单表过长（跨页大表截断在页边界，天然安全）
            if len(" ".join(table)) > max_chars * 2:
                flush_table()
        else:
            flush_table()
            para.append(ln)
    flush_table()
    flush_para()
    return out


# ── 读取器 ──────────────────────────────────────────────


def read_pdf(path: str | Path) -> list[str]:
    """PDF → 每页文本。抽取失败（扫描版）返回空串页并靠上游提示。"""
    r = PdfReader(str(path))
    return [(p.extract_text() or "") for p in r.pages]


def read_md(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [text]


def strip_boilerplate(pages: list[str], page_count: int) -> list[str]:
    """页眉/页脚剥离：在 ≥30% 页面逐字重复的短行是装订噪声（公司名+报告标题、
    页码等）。经典文档级频率分析（trafilatura 同思路），只对多页 PDF 生效。"""
    if page_count < 6:
        return pages
    from collections import Counter

    def _norm(ln: str) -> str:
        # 抹掉数字：页眉常与页码同行（「…报告 4」），逐页不同的数字会让
        # 同一句页眉被当成不同行，逃过 30% 频率剥离（实测海伦哲 2026Q1
        # 的 13 页页眉全部残留、挤掉数字块进 top-k）。真正文行撞上
        # 「数字等价 + ≥30% 页频 + 8-60 字长」三重门槛的概率极低。
        return re.sub(r"[\s\d]+", "", ln)

    freq = Counter()
    for pt in pages:
        for ln in {raw.strip() for raw in pt.splitlines() if raw.strip()}:
            freq[_norm(ln)] += 1
    threshold = max(2, int(page_count * 0.3))
    # 归一化后比对：PDF 抽取的空格位置逐页漂移（2026年/2026 年），
    # 逐字比对会让页眉逃逸
    boiler = {n for n, c in freq.items()
              if c >= threshold and 8 <= len(n) <= 60}
    if not boiler:
        return pages
    return ["\n".join(raw for raw in pt.splitlines()
                      if _norm(raw.strip()) not in boiler)
            for pt in pages]


def keep_block(block: str, is_table: bool) -> bool:
    """空表头过滤：表格块内「≥3 位且非年份」的数字少于 2 个的是纯列名
    块（含「项目 2026年半年度 2025年半年度」这类纯年份列头——年份是
    列标签不是数据），无证据价值、纯检索噪声——落库前丢弃。"""
    if not is_table:
        return True
    nums = [m for m in re.findall(r"\d{3,}", block)
            if not re.fullmatch(r"20\d{2}", m)]
    return len(nums) >= 2


def ingest_file(path: str | Path) -> tuple[Doc, list[Chunk]]:
    company, period, kind = parse_filename(path)
    doc_id = f"{company}_{period}"
    src = str(path)
    if str(path).lower().endswith(".pdf"):
        pages = read_pdf(path)          # 只读一次（v0.4 前读了两遍）
        pages = strip_boilerplate(pages, len(pages))
    else:
        pages = read_md(path)   # md 与 txt 同走纯文本通道

    doc = Doc(doc_id=doc_id, company=company, period=period,
              kind=kind, source=src, n_pages=len(pages))
    chunks: list[Chunk] = []
    for pno, ptext in enumerate(pages, start=1):
        if len(ptext.strip()) < 20:
            continue
        for seq, (block, is_table) in enumerate(chunk_page(ptext)):
            if len(block.strip()) < 15 or not keep_block(block, is_table):
                continue
            notes = normalize_numbers(block)
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#p{pno}#{seq}",
                doc_id=doc_id, page=pno if str(path).lower().endswith(".pdf") else 0,
                seq=seq, text=block, is_table=is_table, num_notes=notes,
            ))
    return doc, chunks


def expand_query(query: str) -> str:
    """查询扩展：① 指标同义（营收→营业收入）② 归因同义（原因→主要系）。

    财报写因果用「主要系…影响」「主要由于…所致」，与口语问法零词面重叠，
    不扩则归因表块根本进不了召回——这是词法检索在财报域的固有盲区，
    显式补上比假装它不存在诚实。
    """
    extra: list[str] = []
    for slang, formal in METRIC_SYNONYMS.items():
        if slang in query.lower():
            extra.extend(formal[:2])
    if any(w in query for w in ("原因", "为什么", "为何", "变动", "下滑", "增长")):
        extra += ["主要系", "主要由于", "说明"]
    extra = list(dict.fromkeys(extra))          # 同义词表多键命中会重复
    return f"{query} {' '.join(extra)}".strip() if extra else query
