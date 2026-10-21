import pytest

from duizhang.graph import build_graph
from duizhang.ingest import chunk_page, expand_query, normalize_numbers, parse_filename
from duizhang.schemas import new_state

SAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """走生产路径建库：Store 写入 → Corpus 加载（与 CLI 完全一致）。"""

    from duizhang.index import Corpus
    from duizhang.ingest import ingest_file, parse_filename
    from duizhang.store import Store
    db = tmp_path_factory.mktemp("dz") / "test.db"
    store = Store(db)
    for f in sorted(SAMPLES.glob("*.pdf")) + sorted(SAMPLES.glob("*.md")):
        try:
            parse_filename(f)          # 与 cli 同闸：解析失败拒收不入库
        except ValueError:
            continue
        store.add_doc(*ingest_file(f))
    store.close()
    return Corpus.load(db)


@pytest.fixture(scope="module")
def app(corpus):
    return build_graph(corpus)


# ── 摄取层 ──────────────────────────────────────────────


def test_parse_filename_metadata():
    company, period, kind = parse_filename("欧科亿_2026年半年度业绩预告.pdf")
    assert (company, period, kind) == ("欧科亿", "2026H1E", "半年度业绩预告")
    _, period2, kind2 = parse_filename("海伦哲_2026年第一季度报告.pdf")
    assert (period2, kind2) == ("2026Q1", "一季报")


def test_normalize_numbers_unit_capture():
    # 万元 → 亿；元 → 亿/万；亿元原样。单位捕获错位曾把 4.19亿 写成 4.19万。
    assert "≈4.19亿" in normalize_numbers("净利润 41,914.72 万元")
    assert "≈6.44亿" in normalize_numbers("营业收入 644,051,473.82 元")
    assert "≈1,234.56亿" in normalize_numbers("募投 1,234.56 亿元")


def test_table_chunk_atomicity():
    table_text = ("项目 本期 上期\n"
                  "营业收入 100,000.00 80,000.00\n"
                  "净利润 20,000.00 15,000.00\n\n"
                  "这是一段正常正文，含有句号。")
    blocks = chunk_page(table_text)
    tables = [b for b, is_t in blocks if is_t]
    assert len(tables) == 1
    assert "营业收入" in tables[0] and "净利润" in tables[0]  # 表格不跨块


def test_metric_synonym_expansion():
    assert "营业收入" in expand_query("营收怎么样")


def test_metric_synonym_guimu():
    """归母 → 归属于上市公司股东的净利润：chunk 级探针曾因缺此映射
    MISS（归母是财报社区通用缩写，正文从不这么写）。"""
    assert "归属于上市公司股东的净利润" in expand_query("归母净利润怎么样")


def test_metric_synonym_jingzichan():
    """净资产 → 归属于上市公司股东的净资产 / 归属于母公司所有者权益：
    归母净资产在主要会计数据表叫前者（行标签常被断词成「者权益（元）」），
    资产负债表叫后者——双映射让两条路都命中（净资产题实测 kw=N）。"""
    eq = expand_query("归属于上市公司股东的净资产多少")
    assert "归属于母公司所有者权益" in eq


def test_parse_filename_rejects_non_company_fallback():
    """P0.5：兜底通道抽出的「公司名」是日期/纯数字（20261009_永和…分析.md
    → company=20261009 → doc_id 20261009_MD 垃圾入库）时拒收——
    cli 以 ⚠ 跳过，不静默降级。含 CJK 主干的分析文件照常 ANALYSIS。"""
    from duizhang.ingest import parse_filename
    with pytest.raises(ValueError):
        parse_filename("20261009_永和股份三季预告_静水CANSLIM分析.md")
    company, _period, kind = parse_filename("永和股份_分析.md")
    assert (company, kind) == ("永和股份", "ANALYSIS")


def test_period_hints_exclusive():
    """「半年度」包含「年度」：顺序判定会让 FY 混进半年度问题的
    期间过滤，年报块随之淹没正确期间（2026-10-14 全量库实测）。"""
    from duizhang.graph import period_hints
    assert period_hints("欧科亿 2026 年半年度预计净利润多少") == ["H1"]
    assert period_hints("2026 半年报营收") == ["H1"]
    assert period_hints("中报怎么样") == ["H1"]
    assert period_hints("2025 年年度报告营业收入") == ["FY"]
    assert period_hints("年报摘要") == ["FY"]
    assert period_hints("第一季度营业收入") == ["Q1"]
    assert period_hints("一季度营收") == ["Q1"]        # 口语「一季度」同样要命中
    assert period_hints("三季报预告") == ["Q3"]
    assert period_hints("二季度净利润") == ["H1"]      # 财报日历：Q2=半年报
    assert period_hints("四季度全年展望") == ["FY"]
    assert period_hints("今天天气怎么样") == []


def test_period_hints_bare_year_is_fy():
    """D18 裸年份→FY：query 只有年份没有期间词时，默认年报口径；
    否则期间过滤为空 → 各期间文档混排，半年报/一季报的数顶掉年报数
    （2026-10-19 innolight 库实测 6 处失败共同根因）。年份判定复用
    query_years（含口语二位「15年」）。无年份仍不限定。"""
    from duizhang.graph import period_hints
    assert period_hints("中际旭创 2025 年营业收入多少") == ["FY"]
    assert period_hints("15 年营收") == ["FY"]          # 口语二位年份同口径
    assert period_hints("营收多少") == []                # 无年份 → 不限定
    assert period_hints("今天天气怎么样") == []          # 领域外不因年份误限定


# ── 生成与对账层 ────────────────────────────────────────


def _hit(chunk_id, doc_id, page, seq, text, rank):
    from duizhang.schemas import Chunk, Retrieved
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, page=page,
                                 seq=seq, text=text), score=0.1, rank=rank)


def test_coverage_counts_cited_chunks_beyond_rank4():
    """抽取式从 hits[:8] 抽句（答案块常在第 4 名开外，见 extractive 注释），
    覆盖率的证据池必须含「答案实际引用的块」——否则自家抽取式答案
    被自家 verify 误判 degraded（全量库实测 trace: verify:degraded(0.37)）。"""
    from duizhang.answer import citation_coverage
    filler = [_hit(f"f{i}", "d", 1, i, "无关内容占位句子。", i + 1) for i in range(4)]
    cited = _hit("c5", "d", 2, 0, "预计 2026 年半年度净利润 36,000 万元。", 5)
    answer = f"预计 2026 年半年度净利润 36,000 万元。 {cited.chunk.citation}"
    assert citation_coverage(answer, filler + [cited]) >= 0.45


def test_verify_passes_for_md_doc_citations():
    """MD/TXT 块 page=0，引用格式是 [doc_id]（无「第N页」也没有 `_\d{4}`
    期间后缀），正则漏判会让 MD 文档的抽取式答案永远 degraded。"""
    from duizhang.answer import verify
    h = _hit("永和股份_MD#p0#0", "永和股份_MD", 0, 0,
             "永和股份三季度预告归母净利润预计增长。", 1)
    answer = f"永和股份三季度预告归母净利润预计增长。 {h.chunk.citation}"
    verdict, _cov = verify(answer, [h])
    assert verdict == "pass"


def test_citation_matching_survives_space_drift():
    """D13 引用格式解耦：小模型手写引用的空格会漂移（「第 8 页」/
    「doc_id第8页」粘连），chunk.citation 精确子串匹配不认 → has_citation
    挂 → 有据答案全 degraded、compare 题结论句被抽取式顶掉（2026-10-19
    llm 模式实测：cov=0.8 仍 degraded）。匹配必须对空白不敏感。"""
    from duizhang.answer import citation_coverage, has_citation, verify
    h = _hit("c1", "中际旭创_2025FY", 8, 0,
             "2025 年营业收入 38,239,935,640.67 元。", 1)
    fact = "2025 年营业收入 38,239,935,640.67 元"
    for drift in ("[中际旭创_2025FY 第 8 页]",    # 数字两侧加空格
                  "[中际旭创_2025FY第8页]"):      # doc 与第N页粘连
        answer = f"{fact} {drift}"
        assert has_citation(answer, [h]), drift
        assert citation_coverage(answer, [h]) >= 0.45, drift
        assert verify(answer, [h]) == ("pass", citation_coverage(answer, [h])), drift


def test_extractive_notes_follow_picked_chunks():
    """数字注释应来自「被选中句所在的块」而非固定 top1——关键数字句常在
    第 2 名开外，注释缺席会让「≈4.19亿」类问法打不中（eval kw 实测）。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import new_state
    h1 = _hit("n1", "d", 1, 0, "欧科亿业绩预告情况的自愿性披露公告。", 1)
    h2 = _hit("n2", "d", 2, 1, "预计 2026 年半年度净利润 41,922.46 万元。", 2)
    h2.chunk.num_notes = ["≈4.19亿"]
    st = new_state("欧科亿 2026 年半年度预计净利润多少亿元", "heuristic")
    body, _cited = extractive_answer(st, [h1, h2])
    assert "41,922.46" in body          # 数字句被选中
    assert "≈4.19亿" in body            # 且注释跟着出现


def test_extractive_skips_boilerplate_header_sentences():
    """页眉句（公司名+报告标题）靠 query 里的公司名/期间得分，真实数字句
    反而落选（全量库实测：海伦哲题 4 句全是同一条页眉）。公司名与元词
    子串（第一季度⊃季度）都不得计分。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import new_state
    header = _hit("h1", "海伦哲_2026Q1", 10, 0,
                  "徐州海伦哲专用车辆股份有限公司 2026年第一季度报告", 1)
    fact = _hit("h2", "海伦哲_2026Q1", 3, 1,
                "报告期内，公司实现营业收入 520,053,589 元，同比增长 15.66%。", 2)
    st = new_state("海伦哲第一季度营业收入多少", "heuristic")
    body, _cited = extractive_answer(st, [header, fact])
    assert "520,053,589" in body          # 数字句入选
    assert "专用车辆" not in body         # 页眉句落选


def test_verify_pool_includes_beyond_rank4_hits():
    """verify 证据池 = 页级集合 ∪ hits：page_evidence 只装 hits[:4] 的页
    窗口，extractive 却从 hits[:8] 抽句——rank 5–8 被引块缺席会让正确
    答案被误标 degraded（拼接行常排中段，2025Q1 扣非实测 0.395）。
    与 coverage 层 test_coverage_counts_cited_chunks_beyond_rank4 同哲学。"""
    from duizhang.graph import make_nodes
    from duizhang.index import Corpus
    from duizhang.schemas import new_state
    nodes = make_nodes(Corpus.__new__(Corpus))       # verify 不查库
    verify = nodes[9]
    fact = _hit("c6", "中际旭创_2025Q1", 3, 5,
                "归属于上市公司股东的扣除非经常性损益的净利润（元）"
                " 1,568,385,287.19 58.39%", 6)      # rank 6：页窗口外
    other = _hit("c1", "中际旭创_2025FY", 94, 0,
                 "营业收入 38,239,935,640.67 元。", 1)
    st = new_state("中际旭创 2025 一季度扣非净利润多少", "heuristic")
    st.update(hits=[other, fact], evidence=[other.chunk],   # 页窗口只见 rank1
              answer=f"扣除非经常性损益后的净利润 1,568,385,287.19 元"
                     f" {fact.chunk.citation}",
              citations=[fact.chunk.citation])
    out = verify(st)
    assert out["verdict"] == "pass", out.get("trace")
    assert out["coverage"] >= 0.45    # 覆盖率进 state（此前只在 trace 字符串里）

def test_extractive_splits_fix_jingzichan_row():
    """断词工件（D16 族）：主要会计数据表的「归属于上市公司股东的
    净资产（元）」行被切成「者权益（元） 20,714,619,471.62 …」——
    query 的「净资产/股东」与残片零重叠，真值行落选（净资产题实测
    kw=N）。注入映射后应抽中。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import new_state
    broken = _hit("z", "中际旭创_2025Q1", 3, 11,
                  "者权益（元） 20,714,619,471.62 19,133,887,012.66 8.26%", 1)
    st = new_state("中际旭创 2025 一季度归属于上市公司股东的净资产多少",
                   "heuristic")
    body, _c = extractive_answer(st, [broken])
    assert "20,714,619,471" in body


def test_extractive_caliber_guard_koufei_vs_revenue():
    """口径守卫（extractive 层，D16 同款）：问句未提扣非时，扣非行不得
    抢答营收/净利问句——rewrite 会给 effective_query 追加通用词
    （净利润/扣非净利润），拼接后的扣非行标签+数字齐全、词面全中，
    实测顶掉营收行（25 年年报营收答成扣非 10,710,053,246.51）。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import new_state
    koufei = _hit("kf", "中际旭创_2025FY", 8, 4,
                  "归属于上市公司股东的扣除非经常性损益的净利润（元）"
                  " 10,710,053,246.51 5,068,356,338.29 111.31%", 1)
    revenue = _hit("rev", "中际旭创_2025FY", 8, 1,
                   "营业收入（元） 38,239,935,640.67 23,862,159,738.37 60.25%", 2)
    st = new_state("中际旭创 25 年年报营收多少", "heuristic")
    # 模拟 rewrite 后的 effective_query：含通用追加词（净利润/同比）
    st["effective_query"] = "中际旭创 25 年年报营收多少 营业收入 净利润 预计 同比"
    body, _c = extractive_answer(st, [koufei, revenue])
    assert "38,239,935,640.67" in body       # 营收行胜出
    assert "10,710,053,246.51" not in body   # 扣非行不得抢答


def test_front_key_table_cap_and_reason_skip():
    """D14 前置封顶：主要会计数据表整页前置（长页 12+ 块）会淹没 Agent
    自己收集的证据；归因题（原因/为什么/变动）要的是叙事证据，表格行
    前置只会挤掉归因句——跳过。"""
    from duizhang.graph import front_key_table
    from duizhang.schemas import Doc
    key_page = [_hit(f"k{i}", "中际旭创_2025Q1", 3, i, f"表行{i} 1,234,567.8", i)
                for i in range(12)]
    agent_hit = _hit("a1", "中际旭创_2025FY", 94, 0,
                     "营业收入 38,239,935,640.67", 1)

    class FakeCorpus:
        docs = {"中际旭创_2025Q1":
                Doc("中际旭创_2025Q1", "中际旭创", "2025Q1", "一季报", "x", 13)}

        def search(self, query, k=8, doc_filter="", doc_scope=None):
            return [_hit("probe", "中际旭创_2025Q1", 3, 0, "主要会计数据", 1)]

        def chunks_page(self, doc_id, page):
            return [h.chunk for h in key_page]

    u = {"intent": "metric", "companies": ["中际旭创"], "period": "Q1"}
    out = front_key_table([agent_hit], u, "中际旭创 2025 一季度营收多少",
                          FakeCorpus())
    assert len(out) == 1 + 8                      # 封顶 8 块，不整页
    assert out[0].chunk.chunk_id == "k0"          # 表页块在头部
    # 归因题：跳过前置，Agent 证据原样
    out2 = front_key_table([agent_hit], u, "中际旭创 2025 一季度净利润变动原因",
                           FakeCorpus())
    assert out2 == [agent_hit]


def test_trend_generate_cites_only_skeleton_docs():
    """trend 引用收窄：引用列表必须等于骨架行里实际出现的引用——
    此前 cite 全部 hits 的引用（含骨架没采用行的期间/页），引用列表
    虚胖且不可回溯到答案文本。"""
    from duizhang.graph import make_nodes
    from duizhang.schemas import new_state

    class _C:                      # trend 分支只需页查询能力
        @staticmethod
        def chunks_page(doc_id, page):
            return []

    generate = make_nodes(_C())[5]
    used = _hit("u1", "中际旭创_2025Q1", 3, 1,
                "营业收入（元） 6,674,176,487.23 4,842,546,924.23 37.82%", 1)
    unused = _hit("x1", "中际旭创_2024Q3", 2, 1,
                  "经营情况讨论与非财务叙述段落。", 2)   # 无可解析行
    st = new_state("中际旭创最近几个季度的营收趋势", "heuristic")
    st["understanding"] = {"intent": "trend", "companies": ["中际旭创"],
                           "period": "", "reason": "rule"}
    st["hits"] = [used, unused]
    out = generate(st)
    assert used.chunk.citation in out["citations"]
    assert unused.chunk.citation not in out["citations"]   # 收窄：不出现在答案里的不引用


def test_stitch_adjacent_table_rows():
    """D16+B2 同页相邻表块拼接：PDF 主要会计数据表把「口径标签块」与
    「光数值块」切成相邻块（2025Q1 扣非：seq4 标签 / seq5 数值
    1,568,385,287.19）。光数值块无词面可与 query 命中，检索只召回标签块
    ——句级打分丢无数字句，扣非题答成归母值（seq18 页 10 标签+数字
    齐全抢答）。拼接把同页相邻的标签块+数值块合成整行。"""
    from duizhang.graph import stitch_adjacent
    label = _hit("a#p3#4", "中际旭创_2025Q1", 3, 4,
                 "归属于上市公司股东的扣除 非经常性损益的净利润 （元）", 1)
    nums = _hit("a#p3#5", "中际旭创_2025Q1", 3, 5,
                "1,568,385,287.19 990,216,569.13 58.39%", 2)
    complete = _hit("a#p3#1", "中际旭创_2025Q1", 3, 1,
                    "营业收入（元） 6,674,176,487.23 4,842,546,924.23 37.82%", 3)

    class C:
        @staticmethod
        def chunks_page(doc_id, page):
            return [complete.chunk, label.chunk, nums.chunk]

    out = stitch_adjacent([label], corpus=C())
    assert len(out) == 1
    merged = out[0].chunk
    assert "1,568,385,287.19" in merged.text          # 数值并入整行
    assert merged.text.index("扣除") < merged.text.index("1,568")  # 标签在前
    # 完整行（标签+数字同块）不拼接、不受影响
    assert stitch_adjacent([complete], corpus=C())[0].chunk.text \
        == complete.chunk.text


def test_rule_understand_resolves_short_company_alias():
    """D20 简称击穿 doc_scope：query 用简称「茅台」而语料公司是全称
    「贵州茅台」，全名子串匹配 miss → companies 空 → 检索泄漏全库
    （实测「茅台 2026 半年度扣非」答成欧科亿 4.19 亿、引用四家混池
    仍 verify:pass；「茅台…趋势」trend:0期 拒答）。双向子串唯一命中
    时归一为全称；多命中歧义与通用后缀 token 不归一。"""

    class Corp:
        def __init__(self, *companies):
            class D:
                pass
            self.docs = {f"d{i}": D() for i in range(len(companies))}
            for d, co in zip(self.docs.values(), companies):
                d.company = co

    from duizhang.graph import rule_understand
    c = Corp("贵州茅台", "中际旭创", "欧科亿")
    assert rule_understand("茅台 2026 年半年度扣非净利润多少",
                           c)["companies"] == ["贵州茅台"]
    assert rule_understand("茅台近几年归母净利润趋势",
                           c)["companies"] == ["贵州茅台"]
    r = rule_understand("对比茅台和中际旭创 2025 年谁赚得多", c)
    assert set(r["companies"]) == {"贵州茅台", "中际旭创"}
    assert r["intent"] == "compare"
    # 歧义：两家共享 token「科技」→ 不归一
    c2 = Corp("日联科技", "海康科技")
    assert rule_understand("科技公司哪家营收高", c2)["companies"] == []
    # 通用后缀 token 即便唯一命中也不归一
    c3 = Corp("日联科技")
    assert rule_understand("科技公司营收多少", c3)["companies"] == []


def test_trend_skeleton_line_level_rows():
    """D21 行级抽取：主要会计数据表块无标点，句级切分把整块当一句，
    pick(净利润) 与 pick(营业收入) 命中同一句 → 归母=营收同值、部分
    期归母整行缺失（茅台实测 506.0/506.0），verify 满分放行——数字
    确实都在引用页，闸门管不住取错行。行级切分 + 指标词之后取首个
    大数，两指标独立取值。"""
    from duizhang.graph import trend_skeleton
    blk = _hit("a#p1#1", "贵州茅台_2025Q1", 1, 1,
               "营业收入（元） 50,605,268,444.40 45,024,567,890.12 12.4%\n"
               "归属于上市公司股东的净利润（元） 26,852,113,255.66 "
               "24,022,335,566.77 11.8%", 1)

    class C:
        @staticmethod
        def chunks_page(doc_id, page):
            return [blk.chunk]

    out = trend_skeleton([blk], corpus=C())
    assert "营收 506.1 亿" in out
    assert "归母净利 268.5 亿" in out        # 句级时这里与营收同值（506.1）


def test_extractive_caliber_mirror_prefers_koufei_row():
    """口径镜像（D16 补面）：问句明确要扣非时，不含「扣除/扣非」的
    净利润句是归母口径——词面重叠相同（净利润）但口径不符。茅台
    2026H1 实测：445.17 亿归母散文句顶掉 44,464,207,646.01 扣非行
    （问句「扣非」与证据「扣除非经常性损益」零词面重叠，守卫放行
    了但打分赢不了）。降权让位，降权不跳过（「扣非和归母差多少」
    类双口径问句仍可见归母句）。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import Chunk, Retrieved
    prose = Retrieved(chunk=Chunk(chunk_id="m#p6#2", doc_id="贵州茅台_2026H1",
                                  page=6, seq=2,
                                  text="归属于上 市公司股东的净利润 445.17 亿元，同比下降 1.95%。"),
                      score=0.5, rank=1)
    row = Retrieved(chunk=Chunk(chunk_id="m#p5#7", doc_id="贵州茅台_2026H1",
                                page=5, seq=7,
                                text="归属于上市公司股东的扣除非经常性损益的净利润（元） "
                                     "44,464,207,646.01 41,322,554,123.45 7.60%"),
                    score=0.4, rank=2)
    st = {"query": "茅台 2026 年半年度扣非净利润多少",
          "effective_query": "茅台 2026 年半年度扣非净利润多少"}
    body, _ = extractive_answer(st, [prose, row])
    assert "44,464,207,646.01" in body
    assert body.index("44,464,207,646.01") < body.index("445.17")   # 扣非行在前


def test_trend_pick_survives_koufei_label_tail_on_same_line():
    """D21 行级守卫精化：主要会计数据表块行内拼接——归母行与下一个
    「扣非」标签尾巴（…扣除非经）同处一行，整行含「扣除」就跳过会把
    归母行陪葬（茅台 2025H1/2026H1 归母缺失实测）。守卫改为只看指标
    词前 20 字的标签上下文：归母自己的标签（归属于上市公司股东的）
    不含扣除 → 放行；扣非行的标签含扣除 → 照旧跳过。"""
    from duizhang.graph import trend_skeleton
    blk = _hit("a#p5#2", "贵州茅台_2025H1", 5, 2,
               "营业收入 89,389,354,416.84 81,930,977,667.75 9.10 "
               "利润总额 62,778,000,766.82 57,578,281,653.44 9.03 "
               "归属于上市公司股东的净利润 45,402,962,298.10 "
               "41,695,610,983.37 8.89 归属于上市公司股东的扣除非经",
               1)

    class C:
        @staticmethod
        def chunks_page(doc_id, page):
            return [blk.chunk]

    out = trend_skeleton([blk], corpus=C())
    assert "营收 893.9 亿" in out
    assert "归母净利 454.0 亿" in out      # 整行守卫时：扣除陪葬 → 缺失


def test_trend_pick_falls_back_when_best_line_fails_parse():
    """D21 名次回退：MD&A 散文句（「净利润 454.03 亿元，同比增长
    8.89%」）带 % 打分高于表行（无 % 号），但亿元级散文过不了
    ≥1e6 原币门槛——只取最佳一行会 解析 None→该指标整行丢失
    （茅台 2025H1/2026H1 归母缺失实测根因）。按分数降序回退到
    第一个可解析行。"""
    from duizhang.graph import trend_skeleton
    table = _hit("a#p5#2", "贵州茅台_2025H1", 5, 2,
                 "营业收入 89,389,354,416.84 81,930,977,667.75 9.10 "
                 "利润总额 62,778,000,766.82 57,578,281,653.44 9.03 "
                 "归属于上市公司股东的净利润 45,402,962,298.10 "
                 "41,695,610,983.37 8.89", 1)
    prose = _hit("a#p7#1", "贵州茅台_2025H1", 7, 1,
                 "一是主要指标稳定向好。上半年，公司营业总收入 910.94 亿元，"
                 "同比增长 9.16%；归属于上 市公司股东的净利润 454.03 亿元，"
                 "同比增长 8.89%，", 2)

    class C:
        @staticmethod
        def chunks_page(doc_id, page):
            return []

    out = trend_skeleton([table, prose], corpus=C())
    assert "归母净利 454.0 亿" in out       # 散文赢打分但解析不过 → 回退表行


def test_extractive_sibling_metric_demotion():
    """指标口径配对降权（D16 家族）：问句点名指标 X 时，只谈同胞指标
    Y 的句子与 X 共享标签前缀词（归属于上市公司股东的…），打分虚高
    冒名顶替——② 问扣非/净利润时净资产句让位（茅台 2026H1 实测
    净资产块 4.4 分顶掉扣非行 1.6 分，44,464,207,646.01 落选）；
    ③ 问总收入时营业收入句让位（子口径冒充总口径，真值 92,278,
    072,083.21，P1#5）。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import Chunk, Retrieved

    def _mk(cid, page, seq, text, rank):
        return Retrieved(chunk=Chunk(chunk_id=cid, doc_id="贵州茅台_2026H1",
                                     page=page, seq=seq, text=text),
                         score=0.5, rank=rank)

    jingzichan = _mk("m#p5#6", 5, 6,
                     "归属于上市公司股东的净资产 251,253,594,419.50 "
                     "244,637,811,032.18 2.70 总资产 309,050,784,569.31", 1)
    koufei = _mk("m#p5#4", 5, 4,
                 "常性损益的净利润 44,464,207,646.01 45,390,247,623.82 "
                 "-2.04 经营活动产生的现金流量净额 70,690,750,119.06", 2)
    st = {"query": "茅台 2026 年半年度扣非净利润多少",
          "effective_query": "茅台 2026 年半年度扣非净利润多少"}
    body, _ = extractive_answer(st, [jingzichan, koufei])
    assert "44,464,207,646.01" in body            # 断词修复+净资产让位后扣非行当选

    shouru = _mk("m#p30#1", 30, 1,
                 "营业收入 90,703,260,964.48 89,389,354,416.84 1.47", 1)
    zongshouru = _mk("m#p30#2", 30, 2,
                     "营业总收入 92,278,072,083.21 91,083,690,144.10 1.31", 2)
    # 前置（D14）把主要会计数据表页 8 块顶到队头后，利润表真值行会被
    # 挤到 rank 9+——抽取窗须扫到 12，否则总收入题永远看不见利润表
    pad = [_mk(f"m#p5#{i}", 5, i, "附注说明文字", 10 + i) for i in range(8)]
    st2 = {"query": "贵州茅台 2026 半年度营业总收入多少",
           "effective_query": "贵州茅台 2026 半年度营业总收入多少"}
    body2, _ = extractive_answer(st2, pad + [shouru, zongshouru])
    assert "92,278,072,083.21" in body2
    assert body2.index("92,278,072,083.21") < body2.index("90,703,260,964.48")

    # ④⑤ 问总资产：净利润族句共享「归属于…的」前缀词族；利润表头句
    # （营业总收入…其中：营业收入…）吃 rewrite 追加词+expand 收入族
    # 映射的三重叠（4.2 分），实测顶掉 1 重叠的总资产真值行 31,582,
    # 872,453.01（innolight 30/31 回退根因）
    jingli = _mk("x#p10#1", 10, 1,
                 "1.归属于母公司所有者的净利润 1,582,876,128.67 1,009,266,297.74", 1)
    p9 = _mk("x#p9#1", 9, 1,
             "一、营业总收入 6,674,176,487.23 4,842,546,924.23 "
             "其中：营业收入 6,674,176,487.23 4,842,546,924.23", 2)
    zongzichan = _mk("x#p3#9", 3, 9,
                     "总资产（元） 31,582,872,453.01 28,866,276,555.26 9.41%", 3)
    st3 = {"query": "中际旭创 2025 年一季度末总资产多少",
           "effective_query": "中际旭创 2025 年一季度末总资产多少 "
                              "预计 营业收入 净利润 同比"}
    body3, _ = extractive_answer(st3, [jingli, p9, zongzichan])
    assert "31,582,872,453.01" in body3
    for other in ("1,582,876,128.67", "6,674,176,487.23"):
        if other in body3:
            assert body3.index("31,582,872,453.01") < body3.index(other)


def test_caliber_mismatch_gate():
    """口径数值闸（D22）：问句点名口径（扣非/总收入）时，答案大数必须
    出现在该口径标签行的数字集里——0.8b 实测拿归母 44,516,880,421.86
    答扣非题、营业收入 90,703,260,964.48 答总收入题，数字都在证据页
    verify 照样 pass（引用级闸门够不着行选择，§9 边界第一处收口）。
    证据无该口径行 → False（检索缺口不归闸门背）。"""
    from duizhang.graph import caliber_mismatch
    from duizhang.schemas import Chunk

    def _c(text):
        return Chunk(chunk_id="x", doc_id="贵州茅台_2026H1", page=5, seq=1,
                     text=text)

    chunks = [_c("归属于上市公司股东的净利润 44,516,880,421.86 45,402,962,298.10\n"
                 "常性损益的净利润 44,464,207,646.01 45,390,247,623.82")]
    assert caliber_mismatch("扣非净利润多少", "净利润 44,516,880,421.86 元",
                            chunks) is True
    assert caliber_mismatch("扣非净利润多少", "扣非 44,464,207,646.01 元",
                            chunks) is False
    assert caliber_mismatch("营业收入多少", "90,703,260,964.48", chunks) is False
    zong = [_c("一、营业总收入 92,278,072,083.21 其中：营业收入 "
               "90,703,260,964.48")]
    assert caliber_mismatch("营业总收入多少", "总收入 90,703,260,964.48 元",
                            zong) is True
    assert caliber_mismatch("营业总收入多少", "总收入 92,278,072,083.21 元",
                            zong) is False
    assert caliber_mismatch("扣非净利润多少", "无数字答案", chunks) is False
    # 标签/数值分块行（D16 场景）：标签行无数字，good-set 取其下一行
    split_rows = [_c("归属于上市公司股东的扣除 非经常性损益的净利润 （元）"),
                  _c("1,568,385,287.19 990,216,569.13 58.39%")]
    assert caliber_mismatch("扣非净利润多少", "扣非 1,582,876,128.67 元",
                            split_rows) is True
    assert caliber_mismatch("扣非净利润多少", "扣非 1,568,385,287.19 元",
                            split_rows) is False


def test_page_evidence_stratifies_pages():
    """页级证据分层取样：前置（D14）会把主要会计数据表页 8+ 块顶到
    hits 队头，原 hits[:top_n] 切片全落在同一页——利润表/MD&A 页永远
    进不了 LLM 证据（总收入题 0.8b 只见 p5 营业收入行）。改为扫到凑
    满 top_n 个不同页为止。"""
    from duizhang.graph import page_evidence

    def _h(cid, page, seq):
        return _hit(cid, "贵州茅台_2026H1", page, seq, f"块{cid}", 1)

    page5 = [_hit(f"a#5#{j}", "贵州茅台_2026H1", 5, j, f"p5块{j}", j + 1)
             for j in range(4)]
    page30 = [_hit("a#30#8", "贵州茅台_2026H1", 30, 8, "p30利润表", 5)]

    class C:
        @staticmethod
        def chunks_page(doc_id, page):
            if page == 5:
                return page5 and [h.chunk for h in page5]
            return [page30[0].chunk]

    out = page_evidence(page5 + page30, corpus=C(), top_n=2, window=0)
    got_pages = {(c.doc_id, c.page) for c in out}
    assert (page5[0].chunk.doc_id, 5) in got_pages
    assert (page30[0].chunk.doc_id, 30) in got_pages   # 分层后第二页在场


def test_trend_skeleton_rejects_koufei_for_default_caliber():
    """口径守卫：skeleton 的「归母净利」默认归母口径——含「扣除/
    扣非」的行是另一口径，词面同样含「净利润」，不加守卫时先到
    先得会拿扣非数当归母（1,568,385,287.19 vs 1,582,876,128.67）。"""
    from duizhang.graph import trend_skeleton
    koufei = _hit("a#p3#5", "中际旭创_2025Q1", 3, 5,
                  "扣除非经常性损益的净利润（元） 1,568,385,287.19 "
                  "990,216,569.13 58.39%", 1)      # 扣非行放前排：先到先得
    guimu = _hit("a#p3#3", "中际旭创_2025Q1", 3, 3,
                 "润（元） 1,582,876,128.67 1,009,266,297.74 56.83%", 2)
    sk = trend_skeleton([koufei, guimu])
    assert "15.8 亿" in sk           # 归母 1,582,876,128.67 → 15.8 亿
    assert "15.7 亿" not in sk       # 扣非 15.7 亿不得顶替归母口径


def test_period_filter_is_doc_level():
    """期间硬过滤必须是文档级：命中级后过滤下，大文档（年报几百块）会把
    目标期间块挤出 top-8，exact 落空回退未过滤结果——拿年报答一季报
    （全量库实测：欧科亿 Q1 同比题拿 2025 年报 29.30% 作答）。"""
    from duizhang.graph import make_nodes
    from duizhang.schemas import Chunk, Doc, Retrieved, new_state

    class FakeCorpus:
        def __init__(self):
            self.docs = {
                "欧科亿_2025FY": Doc("欧科亿_2025FY", "欧科亿", "2025FY", "年报", "x", 100),
                "欧科亿_2026Q1": Doc("欧科亿_2026Q1", "欧科亿", "2026Q1", "一季报", "y", 5),
            }

        def search(self, query, k=8, doc_filter="", doc_scope=None):
            if doc_scope:   # 文档级限定生效 → 只回目标期间
                return [Retrieved(Chunk("q1", "欧科亿_2026Q1", 3, 0,
                                        "营业收入同比 113.49"), 0.1, 1)]
            # 模拟大文档淹没：top-8 全是年报块
            return [Retrieved(Chunk(f"fy{i}", "欧科亿_2025FY", 88, i, "年报营业收入"),
                              0.1, i + 1) for i in range(8)]

    _understand, _route, retrieve, *_ = make_nodes(FakeCorpus())
    st = new_state("欧科亿一季度营业收入同比增长多少")
    out = retrieve({**st, "effective_query": "欧科亿 一季度 营业收入 同比"})
    assert {h.chunk.doc_id for h in out["hits"]} == {"欧科亿_2026Q1"}


def test_retrieve_year_scope():
    """期间过滤必须含年份维度：问 2025Q1 时 2024Q1/2026Q1 的块不该
    入场（中际旭创三年语料实测：三期 Q1 块混排在答案里）。"""

    class YearCorpus:
        def __init__(self):
            from duizhang.schemas import Doc
            self.docs = {f"中际旭创_{y}Q1": Doc(f"中际旭创_{y}Q1", "中际旭创",
                                                f"{y}Q1", "一季报", "x", 5)
                         for y in (2024, 2025, 2026)}

        def search(self, query, k=8, doc_filter="", doc_scope=None):
            from duizhang.schemas import Chunk, Retrieved
            return [Retrieved(Chunk(f"{d}#c", d, 9, 0, "营业总收入 1"),
                              0.1, i + 1)
                    for i, d in enumerate(doc_scope or [])][:k]

    from duizhang.graph import make_nodes
    retrieve = make_nodes(YearCorpus())[2]
    st = new_state("中际旭创 2025 年第一季度营业收入多少", "heuristic")
    out = retrieve(st)
    assert {h.chunk.doc_id for h in out["hits"]} == {"中际旭创_2025Q1"}


def test_ingest_drops_empty_table_headers():
    """空表头块（纯列名、≥3 位数字少于 2 个）是检索噪声（实测 17% 的
    表格块）——落库前丢弃；有数据的表格块与正文块不受影响。"""
    from duizhang.ingest import keep_block
    assert keep_block("项目 2026年半年度 2025年半年度", True) is False   # 空表头
    assert keep_block("营业收入 644,051,473.82 301,672,910.35", True) is True
    assert keep_block("这是一段正常正文，含有句号。", False) is True


def test_page_evidence_window():
    """页级证据组装：top-4 命中去重页 → 同页 ±2 窗口（命中块中心，
    ≤5 块/页）→ 总量硬顶 20。半句的后续与同页表格行由此进入上下文
    （top-8 每页最多 1 块，兄弟块从不自动在场——实测）。"""

    class PageCorpus:
        def __init__(self):
            from duizhang.schemas import Chunk
            self.pages = {("d", 5): [Chunk(f"d#p5#{i}", "d", 5, i,
                                           f"第5页第{i}句。") for i in range(8)],
                          ("d", 9): [Chunk(f"d#p9#{i}", "d", 9, i,
                                           f"第9页第{i}句。") for i in range(3)]}

        def chunks_page(self, doc_id, page):
            return self.pages.get((doc_id, page), [])

    from duizhang.graph import page_evidence
    hits = [_hit("d#p5#4", "d", 5, 4, "第5页第4句。", 1),
            _hit("d#p9#1", "d", 9, 1, "第9页第1句。", 2)]
    chunks = page_evidence(hits, PageCorpus())
    ids = [c.chunk_id for c in chunks]
    assert ids == ["d#p5#2", "d#p5#3", "d#p5#4", "d#p5#5", "d#p5#6",  # ±2 窗口
                   "d#p9#0", "d#p9#1", "d#p9#2"]                        # 全页仅3块


def test_generate_stores_page_evidence_for_verify(monkeypatch):
    """generate 把页级证据写进 state['evidence']，verify 闸门用它做
    证据池——同页窗口块的 citation 串相同（doc+页码），被引池自动
    扩到整页窗口，跨块半句不再误降级。"""

    class EvidCorpus:
        def chunks_page(self, doc_id, page):
            from duizhang.schemas import Chunk
            return [Chunk("d#p5#1", "d", 5, 1, "兄弟块：同比增长 50%。"),
                    Chunk("d#p5#2", "d", 5, 2, "命中块本块。")]

    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    generate = make_nodes(EvidCorpus())[5]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: False)
    st = generate({**new_state("d 营收", "heuristic"),
                   "hits": [_hit("d#p5#2", "d", 5, 2, "命中块本块。", 1)]})
    assert st.get("evidence") and len(st["evidence"]) == 2      # 含兄弟块


def test_extractive_expands_query_terms():
    """抽取式打分必须用扩展后的检索词：query 口语词（营收/revenue）与
    证据财报术语（营业总收入）零词元重叠 → 空手拒答（heuristic
    innolight 实测 T4 两题 refused）。"""
    from duizhang.answer import extractive_answer
    from duizhang.schemas import new_state
    h = _hit("c1", "中际旭创_2026H1", 55, 0,
             "一、营业总收入 41,777,861,795.03 14,789,074,837.80", 1)
    st = new_state("中际旭创 2026 半年度营收多少", "heuristic")
    body, _ = extractive_answer(st, [h])
    assert "41,777,861,795" in body


def test_trend_skeleton_mode_independent():
    """trend 确定性骨架不依赖 LLM——heuristic 模式同样输出逐期骨架
    （此前骨架分支被 llm 条件包裹，heuristic 趋势题退化为普通抽取式，
    innolight 实测 doc=N）。"""

    class TrendCorpus2:
        docs = {}

        def chunks_page(self, doc_id, page):
            from duizhang.schemas import Chunk
            return [Chunk(f"{doc_id}#c", doc_id, page, 0,
                          "营业收入（元） 6,674,176,487.23 4,842,546,924.23"
                          " 37.82%")]

    from duizhang.graph import make_nodes
    generate = make_nodes(TrendCorpus2())[5]
    st = generate({**new_state("中际旭创最近几个季度营收趋势", "heuristic"),
                   "hits": [_hit("中际旭创_2025Q1#c", "中际旭创_2025Q1",
                                 3, 0, "营业收入（元） 6,674,176,487.23"
                                 " 4,842,546,924.23 37.82%", 1)],
                   "understanding": {"intent": "trend",
                                     "companies": ["中际旭创"],
                                     "period": "", "reason": "r"}})
    assert "中际旭创_2025Q1" in st["answer"]          # 骨架在场
    assert "不构成投资建议" in st["trend_audit"]   # v0.6.8 审计层


def test_two_digit_year_parsing():
    """口语「15年中报」指 2015——二位年份必须解析，库里没有 2015
    应触发期间守卫拒答，而非静默答成 2025（用户实测）。"""
    from duizhang.graph import query_years
    assert query_years("中际旭创15年中报") == {"2015"}
    assert query_years("2026 年报和 25 年一季报") == {"2026", "2025"}
    assert query_years("营收 100 元") == set()          # 金额不误判为年份


def test_summarize_anchors_mda_section():
    """「管理层是怎么说的」应锚定 MD&A（经营情况讨论与分析）章节，
    排除表头残渣与会计政策行——种子检索此前命中封面/回购明细表
    （用户实测），章节锚定是唯一可靠路径。"""

    class MdaCorpus:
        def __init__(self):
            from duizhang.schemas import Doc
            self.docs = {"中际旭创_2025H1": Doc(
                "中际旭创_2025H1", "中际旭创", "2025H1", "半年报", "x", 188)}

        def search(self, q, k=6, doc_filter="", doc_scope=None):
            from duizhang.schemas import Chunk, Retrieved
            if "经营" in q:      # MD&A 探针 → 章节首块
                return [Retrieved(chunk=Chunk("mda", "中际旭创_2025H1", 10, 0,
                    "第三节 管理层讨论与分析"), score=0.3, rank=1)]
            return [Retrieved(chunk=Chunk("cover", "中际旭创_2025H1", 1, 0,
                "中际旭创 2025 中报 管理层关于股份回购的实施公告"),
                score=0.2, rank=1)]

        def chunks_of(self, doc_id, limit=12):
            return []

        def chunks_page(self, doc_id, page):
            from duizhang.schemas import Chunk
            return [Chunk("mda#1", doc_id, page, 1,
                          "报告期内，公司实现营业收入 147.9 亿元，AI 算力需求"
                          "持续高景气，管理层认为高端光模块产能稳步释放。"),
                    Chunk("mda#2", doc_id, page, 2,
                          "归属于上市公司股东的扣除 非经常性损益 的净利润 （元）"),
                    Chunk("mda#3", doc_id, page, 3,
                          "子公司向本公司出售资产所发生的未实现内部交易损益抵销。")]

    from duizhang.graph import make_nodes
    summarize = make_nodes(MdaCorpus())[7]
    st = summarize({**new_state("中际旭创 2025 中报管理层是怎么说的", "heuristic"),
                    "understanding": {"intent": "summary",
                                      "companies": ["中际旭创"],
                                      "period": "H1", "reason": "r"}})
    assert "营业收入" in st["answer"] and "管理层" in st["answer"]  # 叙事行
    assert "未实现内部交易损益" not in st["answer"]                 # 会计政策排除
    assert "非经常性损益 的净利润 （元）" not in st["answer"]       # 表头排除


def test_trend_intent_and_period_order():
    """trend 意图：趋势词触发、优先级低于 compare；报告期按时间序
    （period 后缀映射月份——字典序会把 H1 排在 Q1 前，时间上是错的）；
    「季度」类问法过滤季度粒度。"""
    from duizhang.graph import _period_key, rule_understand
    u = rule_understand("中际旭创最近几个季度营收的趋势", _MiniCorpus())
    assert u["intent"] == "trend"
    u2 = rule_understand("对比欧科亿和海伦哲 2026 一季报营收", _MiniCorpus())
    assert u2["intent"] == "compare"                       # compare 更具体，优先
    assert _period_key("2026Q1") < _period_key("2026H1") < _period_key("2026Q3") \
        < _period_key("2026FY") < _period_key("2027Q1")


def test_trend_skeleton_structured_rows():
    """趋势骨架 = 结构化行（期间/营收亿/同比/净利亿），非原始表格行：
    本期/同期/同比混排一串数字可读性差（用户实测「很乱」）；断词行
    「润（元）…」解析为净利；解析失败该指标省略而非打印错数。"""
    from duizhang.graph import trend_skeleton
    hits = [_hit("d#p8#1", "中际旭创_2025H1", 8, 1,
                 "营业收入（元） 14,789,074,837.80 10,798,594,185.75"
                 " 10,798,594,185.75 36.95%", 1),
            _hit("d#p8#3", "中际旭创_2025H1", 8, 3,
                 "润（元） 3,995,115,384.70 2,358,421,051.42 69.40%", 2)]
    out = trend_skeleton(hits)
    assert "中际旭创_2025H1" in out
    assert "147.9 亿" in out and "36.95%" in out       # 营收本期→亿+同比
    assert "40.0 亿" in out and "69.40%" in out        # 断词净利行也解析
    assert "10,798,594,185.75" not in out              # 原始串不再直接输出


def test_trend_analysis_number_gate(monkeypatch):
    """投资分析放开 LLM，但分析里的数字必须来自数据行（可验证闸门，
    而非一刀切禁数字）——出现表外数字整个分析段丢弃，宁缺毋滥。"""
    from duizhang.graph import trend_skeleton
    rows = trend_skeleton([_hit("d#p8#1", "中际旭创_2025H1", 8, 1,
                                "营业收入（元） 14,789,074,837.80 10,798,594,185.75"
                                " 36.95%", 1)])
    gate = __import__("duizhang.graph", fromlist=["_analysis_numbers_ok"])
    ok = gate._analysis_numbers_ok("2025H1 营收 147.9 亿（+36.95%），增长强劲。", rows)
    bad = gate._analysis_numbers_ok("预计 2027 年营收将达 800 亿。", rows)
    assert ok is True and bad is False


def test_understand_period_override_and_trend_necessity(app, monkeypatch):
    """v0.6.2 实测回归：①「第一季度」被 0.8b 稳定抽成 H1——query 有
    明确期间词时规则是确定性匹配，应覆盖 LLM 的非空错值（原守卫只
    补空）；②「预计净利润多少」被判 trend（无趋势词）——trend 的
    必要条件是 query 含趋势词，否则降级 metric。"""
    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    understand = make_nodes(_MiniCorpus())[0]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    # ① LLM 给错期间（H1），query 明确说第一季度 → 覆盖为 Q1
    monkeypatch.setattr(llmx, "chat_json", lambda s, u, **kw: {
        "intent": "compare", "companies": ["欧科亿", "海伦哲"],
        "period": "H1", "reason": "一季度比较"})
    st = understand(new_state("对比欧科亿和海伦哲 2026 年第一季度的营业收入", "llm"))
    assert st["understanding"]["period"] == "Q1"
    # ② LLM 判 trend 但 query 无趋势词 → 降级 metric
    monkeypatch.setattr(llmx, "chat_json", lambda s, u, **kw: {
        "intent": "trend", "companies": ["欧科亿"], "period": "H1",
        "reason": "预测"})
    st2 = understand(new_state("欧科亿 2026 年半年度预计净利润多少", "llm"))
    assert st2["understanding"]["intent"] == "metric"


def test_understand_trend_guard(app, monkeypatch):
    """0.8b 实测把趋势问句判成 summary——守卫纠正（与 compare 守卫同款）。"""
    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    understand = make_nodes(_MiniCorpus())[0]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    monkeypatch.setattr(llmx, "chat_json", lambda s, u, **kw: {
        "intent": "summary", "companies": ["欧科亿"], "period": "",
        "reason": "概览"})
    st = understand(new_state("欧科亿最近几个季度营收的趋势", "llm"))
    assert st["understanding"]["intent"] == "trend"


def test_trend_research_sweeps_periods():
    """跨期研究：遍历该公司报告期（最近6期、时间升序）、每期 doc_scope
    检索、合并进 hits——趋势问题从单文档残渣变为多期证据。"""

    class TrendCorpus:
        def __init__(self):
            from duizhang.schemas import Doc, Retrieved
            self.docs = {f"中际旭创_{p}": Doc(f"中际旭创_{p}", "中际旭创",
                                              p, "报告", "x", 10)
                         for p in ("2024Q1", "2024H1", "2025Q1", "2025H1",
                                   "2026Q1", "2026H1", "2025FY", "2024FY")}
            self._R = Retrieved

        def search(self, q, k=8, doc_filter="", doc_scope=None):
            from duizhang.schemas import Chunk
            out = []
            for i, d in enumerate(sorted(doc_scope or [])):
                out.append(self._R(chunk=Chunk(f"{d}#c", d, 8, 0,
                                               f"{d} 营业收入 100 同比增长"),
                                   score=0.1, rank=i + 1))
            return out[:k]

    from duizhang.graph import make_nodes
    trend = make_nodes(TrendCorpus())[11]
    st = trend({**new_state("中际旭创最近几个季度营收趋势", "heuristic"),
                "understanding": {"intent": "trend", "companies": ["中际旭创"],
                                  "period": "", "reason": "r"}})
    docs = [h.chunk.doc_id for h in st["hits"]]
    # 季度粒度、最近 6 期、时间升序（2024Q1 → 2026H1，FY 被季度问法排除）
    assert docs == ["中际旭创_2024Q1", "中际旭创_2024H1", "中际旭创_2025Q1",
                    "中际旭创_2025H1", "中际旭创_2026Q1", "中际旭创_2026H1"]
    assert any(t.startswith("trend:") for t in st["trace"])


# ── 查询理解（understand 节点） ─────────────────────────


class _MiniCorpus:
    """规则理解的假语料：三家公司不同期间。"""

    def __init__(self):
        from duizhang.schemas import Doc
        self.docs = {
            "欧科亿_2026Q1": Doc("欧科亿_2026Q1", "欧科亿", "2026Q1", "一季报", "x", 5),
            "欧科亿_2025FY": Doc("欧科亿_2025FY", "欧科亿", "2025FY", "年报", "x", 100),
            "海伦哲_2026Q1": Doc("海伦哲_2026Q1", "海伦哲", "2026Q1", "一季报", "y", 5),
        }


def test_rule_understand_multi_company_compare():
    """规则理解：多公司全量命中（替代旧 _doc_filter 只回首个），
    对比词 + 公司≥2 → compare 意图。"""
    from duizhang.graph import rule_understand
    u = rule_understand("对比欧科亿和海伦哲 2026 一季报的营业收入增长", _MiniCorpus())
    assert u["intent"] == "compare"
    assert set(u["companies"]) == {"欧科亿", "海伦哲"}
    assert u["period"] == "Q1"


def test_rule_understand_single_metric():
    from duizhang.graph import rule_understand
    u = rule_understand("欧科亿 2026 年半年度预计净利润多少", _MiniCorpus())
    assert u["intent"] == "metric"
    assert u["companies"] == ["欧科亿"]
    assert u["period"] == "H1"


def test_understand_node_llm_and_fallback(app, monkeypatch):
    """understand 节点：llm 模式一次结构化抽取；LLM 故障回退规则并留痕。"""
    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    understand = make_nodes(_MiniCorpus())[0]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    monkeypatch.setattr(llmx, "chat_json", lambda s, u, **kw: {
        "intent": "compare", "companies": ["欧科亿", "海伦哲"],
        "period": "Q1", "reason": "对比两家"})
    st = understand(new_state("对比欧科亿和海伦哲 2026 一季报营收", "llm"))
    assert st["understanding"]["companies"] == ["欧科亿", "海伦哲"]
    assert "understand:llm" in st["trace"]

    def boom(s, u, **kw):
        raise RuntimeError("llm down")

    monkeypatch.setattr(llmx, "chat_json", boom)
    st2 = understand(new_state("欧科亿 2026 年半年度预计净利润多少", "llm"))
    assert st2["understanding"]["intent"] == "metric"     # 规则兜底
    assert any("understand:llm→fallback" in t for t in st2["trace"])


def test_understand_normalizes_loose_intent(app, monkeypatch):
    """常思模型偶尔回中文意图（glm-5.3 实测「对比营收」）——含对比
    语义且多公司时归一为 compare，不能一律掉到 metric。"""
    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    understand = make_nodes(_MiniCorpus())[0]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    monkeypatch.setattr(llmx, "chat_json", lambda s, u, **kw: {
        "intent": "对比营收", "companies": ["欧科亿", "海伦哲"],
        "period": "Q1", "reason": "对比两家"})
    st = understand(new_state("对比欧科亿和海伦哲 2026 一季报营收", "llm"))
    assert st["understanding"]["intent"] == "compare"


def test_route_consumes_understanding():
    """route 纯函数化：消费 understand 的 intent 分发（不再自己调
    LLM/关键词），compare 意图透传；无 understanding 时规则兜底。"""
    from duizhang.graph import make_nodes
    _understand, route, *_ = make_nodes(_MiniCorpus())
    st = new_state("随便什么query，关键词不该决定路由", "heuristic")
    st = {**st, "understanding": {"intent": "summary", "companies": ["欧科亿"],
                                  "period": "", "reason": "LLM 说要概览"}}
    assert route(st)["route"] == "summary"
    st2 = {**st, "understanding": {"intent": "compare",
                                   "companies": ["欧科亿", "海伦哲"],
                                   "period": "Q1", "reason": "对比"}}
    assert route(st2)["route"] == "compare"
    out3 = route(new_state("欧科亿 2026 年半年度预计净利润多少", "heuristic"))
    assert out3["route"] == "metric"


def test_retrieve_scopes_from_understanding():
    """retrieve 的 doc_scope 由 understanding 的公司+期间构建——
    LLM 理解直接驱动检索限定，替代旧的字符串子串猜测。"""

    class ScopeCorpus:
        def __init__(self):
            self.docs = _MiniCorpus().docs
            self.calls = []

        def search(self, q, k=8, doc_filter="", doc_scope=None):
            self.calls.append(doc_scope)
            return []

    sc = ScopeCorpus()
    from duizhang.graph import make_nodes
    _u, _r, retrieve, *_ = make_nodes(sc)
    st = new_state("欧科亿一季度营收", "heuristic")
    st = {**st, "understanding": {"intent": "metric", "companies": ["欧科亿"],
                                  "period": "Q1", "reason": "r"}}
    retrieve(st)
    assert sc.calls[-1] == {"欧科亿_2026Q1"}   # 该公司 + 该期间的文档集合


def test_compare_decomposes_and_merges_both_companies():
    """compare 多跳：按公司 doc_scope 各自检索、交错合并——两家证据
    都在场；某公司无命中时另一家照常（不空手）。"""
    from duizhang.graph import build_graph
    from duizhang.schemas import Chunk, Doc, Retrieved

    class DecomposeCorpus:
        def __init__(self):
            self.docs = {
                "欧科亿_2026Q1": Doc("欧科亿_2026Q1", "欧科亿", "2026Q1", "一季报", "x", 5),
                "海伦哲_2026Q1": Doc("海伦哲_2026Q1", "海伦哲", "2026Q1", "一季报", "y", 5),
            }

        def search(self, q, k=8, doc_filter="", doc_scope=None):
            out = [Retrieved(Chunk(f"{d}#c{i}", d, 1, i,
                                   f"{d} 营业收入 100,000 万元 同比增长"),
                              0.2, i + 1)
                   for d in (doc_scope or []) for i in range(2)]
            return out[:k]

    app = build_graph(DecomposeCorpus())
    st = app.invoke(new_state("对比欧科亿和海伦哲 2026 一季报的营业收入增长", "heuristic"))
    assert st["route"] == "compare"
    assert any("decompose:" in t for t in st["trace"])
    assert {h.chunk.doc_id for h in st["hits"]} == {"欧科亿_2026Q1", "海伦哲_2026Q1"}
    assert st["answer"]


def test_boilerplate_strips_page_number_suffixed_headers():
    """页眉与页码同行（「…2026年第一季度报告 4」）时，逐字比对把每页
    当成不同行，页眉逃过 30% 频率剥离——归一化必须抹掉数字。"""
    from duizhang.ingest import strip_boilerplate
    pages = [f"徐州海伦哲专用车辆股份有限公司 2026年第一季度报告 {i}\n营业收入 100\n"
             for i in range(1, 11)]
    out = strip_boilerplate(pages, 10)
    assert all("徐州海伦哲" not in p for p in out)   # 页眉（含页码后缀）剥离
    assert all("营业收入" in p for p in out)         # 正文不受影响


class _AgentCorpus:
    """agent 循环的假语料：记录 search 调用，返回固定命中。"""

    def __init__(self):
        from duizhang.schemas import Doc, Retrieved
        self.docs = {
            "欧科亿_2026Q1": Doc("欧科亿_2026Q1", "欧科亿", "2026Q1", "一季报", "x", 5),
            "海伦哲_2026Q1": Doc("海伦哲_2026Q1", "海伦哲", "2026Q1", "一季报", "y", 5),
        }
        self.calls = []
        self._R = Retrieved

    def search(self, query, k=8, doc_filter="", doc_scope=None):
        from duizhang.schemas import Chunk
        self.calls.append((query, doc_scope))
        out = []
        for d in (doc_scope or list(self.docs)):
            for i in range(3):          # 每家 3 块：复现「单方连续占位」的插入序
                out.append(self._R(chunk=Chunk(f"{d}#c{i}", d, 1, i,
                                                f"{d} 营业收入 100,00{i} 万元"),
                                    score=0.2, rank=len(out) + 1))
        return out[:k]


def test_agent_research_loop_bounded_and_fallback(app, monkeypatch):
    """真 Agent 环：模型自主决定调 search、何时收手；步数硬上限 6；
    工具失败/无证据回退固定检索管线（不空手）。"""
    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    nodes = make_nodes(_AgentCorpus())
    agent = nodes[10]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)

    # 场景 1：搜一次即收手
    replies = iter([
        {"content": "", "tool_calls": [
            {"name": "search", "arguments": {"query": "营业收入",
                                             "company": "欧科亿", "period": "Q1"}}]},
        {"content": "证据足够", "tool_calls": []},
    ])
    monkeypatch.setattr(llmx, "chat_with_tools", lambda m, t, **kw: next(replies))
    st = agent(new_state("欧科亿一季度营收", "llm"))
    assert st["hits"] and st["hits"][0].chunk.doc_id == "欧科亿_2026Q1"
    assert any(t.startswith("agent:") for t in st["trace"])
    # Agent 的检索词写进 effective_query——下游抽取式/verify 用它打分，
    # 原始口语 query（「哪家赚得多」）与证据零重叠会误拒（实测）
    assert "营业收入" in st["effective_query"]

    # 场景 2：不停调用且每次换参数（绕开重复检测）→ 步数封顶
    counter = {"n": 0}

    def varying(m, t, **kw):
        counter["n"] += 1
        return {"content": "", "tool_calls": [
            {"name": "search", "arguments": {"query": f"营收{counter['n']}"}}]}

    monkeypatch.setattr(llmx, "chat_with_tools", varying)
    st2 = agent(new_state("欧科亿一季度营收", "llm"))
    assert any("6步" in t for t in st2["trace"])          # 硬上限
    assert any("/budget" in t for t in st2["trace"])

    # 场景 3：工具层故障 → 回退固定管线
    def boom(m, t, **kw):
        raise RuntimeError("tools down")
    monkeypatch.setattr(llmx, "chat_with_tools", boom)
    st3 = agent(new_state("欧科亿一季度营收", "llm"))
    assert st3["hits"]                                       # 回退检索仍有证据
    assert any("agent:fallback" in t for t in st3["trace"])

    # 场景 4：小模型死循环（同一调用原样重复）→ 重复检测提前收手
    repeat = {"content": "", "tool_calls": [
        {"name": "search", "arguments": {"query": "营收", "company": "欧科亿"}}]}
    monkeypatch.setattr(llmx, "chat_with_tools", lambda m, t, **kw: repeat)
    st4 = agent(new_state("欧科亿一季度营收", "llm"))
    assert st4["hits"]                                       # 首轮证据保留
    assert any("重复收手" in t for t in st4["trace"])
    assert not any("6步" in t for t in st4["trace"])         # 没烧满预算

    # 场景 5：多家证据按文档轮转输出——单方信息密度高不能挤掉另一方
    # （decompose 交错合并的同款教训，pool 也适用）
    seq = iter([
        {"content": "", "tool_calls": [
            {"name": "search", "arguments": {"query": "营收", "company": "欧科亿"}}]},
        {"content": "", "tool_calls": [
            {"name": "search", "arguments": {"query": "营收", "company": "海伦哲"}}]},
        {"content": "够了", "tool_calls": []},
    ])
    monkeypatch.setattr(llmx, "chat_with_tools", lambda m, t, **kw: next(seq))
    st5 = agent(new_state("对比欧科亿和海伦哲营收", "llm"))
    docs_order = [h.chunk.doc_id for h in st5["hits"][:4]]
    assert docs_order[0] != docs_order[1]                    # 前两块来自不同公司

    # 场景 6：模型不传 company/period → 从 understanding 继承（小模型
    # 单跳题实测不传参，全库检索被淹没）
    no_comp = iter([
        {"content": "", "tool_calls": [
            {"name": "search", "arguments": {"query": "预计净利润"}}]},
        {"content": "够", "tool_calls": []},
    ])
    monkeypatch.setattr(llmx, "chat_with_tools", lambda m, t, **kw: next(no_comp))
    st6 = agent({**new_state("欧科亿 2026 一季度预计净利润", "llm"),
                 "understanding": {"intent": "metric", "companies": ["欧科亿"],
                                   "period": "Q1", "reason": "r"}})
    assert st6["hits"]
    assert all(h.chunk.doc_id == "欧科亿_2026Q1" for h in st6["hits"])

    # 场景 7：工具参数与理解冲突时以理解为准——0.8b 实测把「半年度」
    # 幻觉成 FY 传参，显式错误参数覆盖先验后年报块洪水
    wrong_p = iter([
        {"content": "", "tool_calls": [
            {"name": "search", "arguments": {"query": "净利润",
                                             "period": "FY"}}]},   # 错误参数
        {"content": "够", "tool_calls": []},
    ])
    monkeypatch.setattr(llmx, "chat_with_tools", lambda m, t, **kw: next(wrong_p))
    st7 = agent({**new_state("欧科亿 2026 一季度预计净利润", "llm"),
                 "understanding": {"intent": "metric", "companies": ["欧科亿"],
                                   "period": "Q1", "reason": "r"}})
    assert all(h.chunk.doc_id == "欧科亿_2026Q1" for h in st7["hits"])


def test_understand_period_backfill_from_rule(app, monkeypatch):
    """小模型抽 period 时灵时不灵（实测空串→检索不限期间→年报块淹没）：
    空值时从规则确定性匹配回填——与公司名纠偏对称。"""
    from duizhang import llm as llmx
    from duizhang.graph import make_nodes
    understand = make_nodes(_MiniCorpus())[0]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    monkeypatch.setattr(llmx, "chat_json", lambda s, u, **kw: {
        "intent": "metric", "companies": ["欧科亿"], "period": "", "reason": "r"})
    st = understand(new_state("欧科亿 2026 年半年度预计净利润多少", "llm"))
    assert st["understanding"]["period"] == "H1"      # 规则回填


# ── 图行为 ──────────────────────────────────────────────


def _ask(app, q):
    return app.invoke(new_state(q, "heuristic"))


def test_metric_answer_has_citation(app):
    st = _ask(app, "欧科亿 2026 年半年度预计净利润多少")
    assert st["verdict"] == "pass"
    assert "[欧科亿_2026H1E" in st["answer"]
    assert any(k in st["answer"] for k in ("36,000", "42,000", "4.19亿"))


def test_summary_route_not_hijacked(app):
    st = _ask(app, "总结一下欧科亿业绩预告")
    assert st["route"] == "summary"
    assert "要点" in st["answer"]


def test_period_guard_refuses_future_year(app):
    """期间守卫要给出具体原因（库里有哪些期间、缺哪年）——此前被
    通用 refuse 文案覆盖（茅台示例实测暴露）。"""
    st = _ask(app, "欧科亿 2030 年营收预测")
    assert st["verdict"] == "refused"
    assert "现有期间" in st["answer"]          # 具体提示保留到最终答案


def test_summarize_keylines_from_retrieval_not_head():
    """长报告（184-236 页）要点藏在第 10 页后的经营分析，文档头部
    12 块是封面/董事名录——要点抓取必须以检索命中块为种子（茅台
    102 页年报实测要点为空）。"""

    class SummaryCorpus:
        def __init__(self):
            from duizhang.schemas import Doc
            self.docs = {
                "中际旭创_2024FY": Doc("中际旭创_2024FY", "中际旭创",
                                       "2024FY", "年报", "x", 236),
                "中际旭创_2025FY": Doc("中际旭创_2025FY", "中际旭创",
                                       "2025FY", "年报", "x", 229)}

        def search(self, q, k=6, doc_filter="", doc_scope=None):
            from duizhang.schemas import Chunk, Retrieved
            scope = doc_scope or set()
            if any("2025FY" in d for d in scope):
                return [Retrieved(chunk=Chunk(
                    "c1", "中际旭创_2025FY", 12, 0,
                    "中际旭创 2025 年报：报告期内实现营业收入 1,000,000 万元，"
                    "同比增长 50%。"),
                    score=0.05, rank=1)]
            if not scope:   # 复现真实排序：无范围时旧年报块抢占
                return [Retrieved(chunk=Chunk(
                    "c0", "中际旭创_2024FY", 10, 0,
                    "中际旭创 2024 年报：经营情况讨论与分析。"),
                    score=0.05, rank=1)]
            return []

        def chunks_of(self, doc_id, limit=12):
            from duizhang.schemas import Chunk
            return [Chunk(f"h{i}", doc_id, 1, i, "董事 杜某 2013 年 3 月")
                    for i in range(12)]          # 头部全是名录

    from duizhang.graph import make_nodes
    summarize = make_nodes(SummaryCorpus())[7]
    st = summarize({**new_state("总结一下中际旭创 2025 年报", "heuristic"),
                    "understanding": {"intent": "summary",
                                      "companies": ["中际旭创"],
                                      "period": "FY", "reason": "r"}})
    assert "营业收入" in st["answer"]            # 要点来自检索命中块
    assert st["verdict"] == "pass"
    assert st["citations"] == ["中际旭创_2025FY"]  # 年份对：不是 2024FY


def test_period_aware_retrieval(app):
    st = _ask(app, "海伦哲第一季度营业收入多少")
    assert st["hits"][0].chunk.doc_id == "海伦哲_2026Q1"


def test_corrective_loop_bounded(app):
    st = _ask(app, "欧科亿一季报归母净利润多少")
    assert st.get("retries", 0) <= 1          # 纠错环有界
    assert st["verdict"] in ("pass", "degraded")


def test_eval_judge_expect_docs():
    """对比题判定：expect_docs 列表里的每个期望文档都要在 hits 或
    citations 中出现——只中一家不算过。"""
    from duizhang.evalx import judge
    case = {"q": "对比", "expect_keywords": ["k"], "expect_docs": ["欧科亿_2026Q1", "海伦哲_2026Q1"]}
    both = {"answer": "k", "citations": [],
            "hits": [_hit("a", "欧科亿_2026Q1", 1, 0, "k", 1),
                     _hit("b", "海伦哲_2026Q1", 1, 0, "k", 2)]}
    assert judge(case, both)["ok"] is True
    one_only = {"answer": "k", "citations": [],
                "hits": [_hit("a", "欧科亿_2026Q1", 1, 0, "k", 1)]}
    assert judge(case, one_only)["ok"] is False


def test_coverage_evidence_includes_num_notes():
    """LLM 常用数字注释的表述作答（6.44亿）而非原文数字（644,051,473.82），
    证据池必须包含 num_notes——与 grade/extractive 的口径一致。"""
    from duizhang.answer import citation_coverage
    h = _hit("n1", "欧科亿_2026Q1", 11, 0,
             "一、营业总收入 644,051,473.82 301,672,910.35", 1)
    h.chunk.num_notes = ["≈6.44亿"]
    answer = "欧科亿 2026 年第一季度营业总收入约 6.44亿。 [欧科亿_2026Q1 第11页]"
    assert citation_coverage(answer, [h]) >= 0.45


def test_coverage_grades_numeric_sentences_only():
    """LLM 的对比答案含框架句（「根据提供的证据…如下」「因此…」）——
    财报问答的编造风险集中在数字句，框架句不该拉低覆盖率均值；
    全答案无数字句时回退全句考核（摘要类）。"""
    from duizhang.answer import citation_coverage
    h1 = _hit("a", "欧科亿_2026Q1", 11, 0,
              "一、营业总收入 644,051,473.82 301,672,910.35", 1)
    h2 = _hit("b", "海伦哲_2026Q1", 10, 0,
              "一、营业总收入 520,053,589.94 345,593,109.95", 2)
    answer = ("根据提供的证据，两家公司 2026 年第一季度营业收入对比如下。"
              f"欧科亿营业收入 644,051,473.82 元 {h1.chunk.citation}。"
              f"海伦哲营业收入 520,053,589.94 元 {h2.chunk.citation}。"
              "因此欧科亿高于海伦哲。")
    assert citation_coverage(answer, [h1, h2]) >= 0.45


# ── 评估 ────────────────────────────────────────────────


def test_llm_fallback_visible_in_trace(app, monkeypatch):
    """LLM 决策点故障：静默降级到规则（图不变），但 trace 必须留痕——
    否则 --mode llm 演示失败时无法察觉，还以为是规则在回答。"""
    from duizhang import graph as _g  # noqa: F401  确认可导入
    from duizhang import llm as llmx

    def boom(system, user, **kw):
        raise RuntimeError("llm down")

    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    monkeypatch.setattr(llmx, "chat_json", boom)
    monkeypatch.setattr(llmx, "chat_text", boom)
    st = app.invoke(new_state("欧科亿 2026 年半年度预计净利润多少", "llm"))
    assert any("fallback" in t for t in st["trace"])
    assert st["answer"]                    # 降级后仍有答案（绝不空手而归）


def test_eval_judge_requires_doc_hit():
    """评估口径：非拒答题的 ✓ 必须同时要求「关键词命中」与「检索命中」，
    否则逐题 ✓ 与 retrieval@3 汇总对不上。summary 路由不填 hits 但填
    citations——判定两条通道都要认。"""
    from duizhang.evalx import judge
    case = {"q": "x", "expect_keywords": ["k"], "expect_doc": "d1"}
    # 关键词中了，但检索既无 hits 也无 citations → 不算过
    assert judge(case, {"answer": "k", "hits": [], "citations": [],
                        "verdict": "pass"})["ok"] is False
    # citations 通道：summary 路由的形态
    assert judge(case, {"answer": "k", "hits": [], "citations": ["d1"],
                        "verdict": "pass"})["ok"] is True
    # hits 通道：metric 路由的形态
    st = {"answer": "k", "citations": [], "verdict": "pass",
          "hits": [_hit("c", "d1", 1, 0, "k", 1)]}
    assert judge(case, st)["ok"] is True
    # 应拒答题：拒了即过
    refused = {"q": "y", "expect_refused": True}
    assert judge(refused, {"answer": "", "hits": [], "citations": [],
                           "verdict": "refused"})["ok"] is True


def test_eval_dataset_passes_thresholds(corpus):
    from duizhang.evalx import run_eval
    s = run_eval(corpus, mode="heuristic", verbose=False)
    assert s["answer_hit"] >= 0.9
    assert s["refusal"] == 1.0
    assert s["retrieval_top3"] >= 0.75


def test_fts_search_fuses_extra_vector_ranking(tmp_path):
    """P1.2 三路 RRF：extra_ranking（向量路）作为第三排名列表进 RRF。
    动机场景：口语「归母净资产」与报表「归属于母公司所有者权益」
    零词面重叠，双轨 FTS 全 miss——向量语义命中由第三路补位。"""
    from duizhang import fts
    from duizhang.schemas import Chunk, Doc
    from duizhang.store import Store
    s = Store(tmp_path / "v.db")
    s.add_doc(Doc("测试司_2025Q1", "测试司", "2025Q1", "一季报", "x", 1),
              [Chunk("测试司_2025Q1#p1#0", "测试司_2025Q1", 1, 0,
                     "归属于母公司所有者权益合计 20,714,619,471.62")])
    con = s.con
    docs = {"测试司_2025Q1": Doc("测试司_2025Q1", "测试司", "2025Q1",
                                 "一季报", "x", 1)}
    # 双轨基线：词面零重叠 → 空
    assert fts.search(con, docs, "归母净资产") == []
    # 第三路注入该块 → 三路融合命中
    out = fts.search(con, docs, "归母净资产",
                     extra_ranking=["测试司_2025Q1#p1#0"])
    assert out and out[0].chunk.chunk_id == "测试司_2025Q1#p1#0"
    s.close()


def test_quarter_view_unified_series():
    """v0.6.6 单季统一视图：Q1 直取、Q2=H1−Q1、Q3=三季报累计−H1（新
    差分）、Q4=FY−三季报累计——每季一行（营收+归母同列），取最近 n 季。
    用户实测「近4个季度」此前给的是报告期混合口径（单季/半年累计/前三
    季累计混排），日历季度序列在数据层不存在。"""
    from duizhang.graph import _quarter_view

    def _e(doc_id, tag, rows):
        period = doc_id.rsplit("_", 1)[-1]
        return {"doc_id": doc_id, "period": period, "tag": tag,
                "cumulative": tag == "Q3", "rows": rows,
                "citation": f"[{doc_id} 第1页]", "page": 1}

    def _r(label, yi, yoy, raw):
        return {"label": label, "yi": yi, "yoy": yoy, "raw": raw}

    entries = [
        _e("X_2025Q1", "Q1", [_r("营收", "66.7 亿", "37.82%", 6674176487.23),
                               _r("归母净利", "15.8 亿", "56.83%", 1582876128.67)]),
        _e("X_2025H1", "H1", [_r("营收", "147.9 亿", "36.95%", 14789074837.80),
                               _r("归母净利", "40.0 亿", "69.40%", 4.0e9)]),
        _e("X_2025Q3", "Q3", [_r("营收", "250.0 亿", "44.43%", 25000000000.00),
                               _r("归母净利", "71.3 亿", "90.05%", 7.13e9)]),
    ]
    items, derived = _quarter_view(entries, n=4)
    text = "\n".join(t for _, t in items)
    assert "X_2025Q1（单季）：营收 66.7 亿（同比 37.82%）" in text
    assert "X_2025Q2（单季·派生，环比" in text            # H1−Q1 + 环比（v0.6.7）
    assert "营收 81.1 亿" in text
    assert "营收 102.1 亿" in text                          # 三季报累计−H1（新差分）
    assert len(items) == 3
    assert any(d["label"] == "X_2025Q3营收" for d in derived)  # 出口重算登记
    assert any(d["label"] == "X_2025Q2归母净利" for d in derived)


def test_quarter_n_parse():
    """「近N个季度」信号解析：带数字才进单季视图；「最近几个季度/
    近几年」保持报告期视图（innolight/多轮既有题的关键词锚在 as-reported
    值上，无数字切换会整体回退）。"""
    from duizhang.graph import _quarter_n
    assert _quarter_n("近4个季度的营收趋势") == 4
    assert _quarter_n("最近3季怎么样") == 3
    assert _quarter_n("近 2 个季度") == 2
    assert _quarter_n("近几年营收趋势") is None
    assert _quarter_n("最近几个季度的净利润趋势") is None


def test_direction_gate_second_order_pass():
    """方向闸二阶收口（v0.6.6）：方向词前紧邻速率名词（增速/增幅/涨幅/
    增长率）判为二阶——「增速回落」是速率事实且与同比正号相容（146%→37%
    指标仍涨），此前被当一阶词误杀，glm-5.3 的正确方向叙述整段被毙退
    骨架（用户实测 trace: direction-gate(回落…)）。裸「回落」仍拦。"""
    from duizhang.graph import direction_gate
    sk = "· X_2025Q1（单季）：营收 66.7 亿（同比 37.82%）；归母净利 15.8 亿（同比 56.83%）"
    assert direction_gate("同比从 146% 降至 37%，营收增速回落", sk) is None
    assert direction_gate("营收增长稳健", sk) is None
    assert direction_gate("营收大幅回落", sk) is not None


def test_ingest_file_rejects_zero_chunk_pdf(tmp_path):
    """空文档闸门（v0.7 正确性批）：扫描版 PDF（每页抽不出文本）在
    ingest_file 出口显式拒收——修前 0 块文档「成功」入库且 manifest
    指纹锁死，永不重试；库里留一具检索不可见、问答永空手的空壳。"""
    from pypdf import PdfWriter

    from duizhang.ingest import ingest_file
    w = PdfWriter()
    for _ in range(3):
        w.add_blank_page(width=595, height=842)   # A4（空 writer 需显式尺寸）
    p = tmp_path / "空壳公司_2025年年度报告.pdf"
    with open(p, "wb") as f:
        w.write(f)
    with pytest.raises(ValueError, match="扫描版|空文档|0 块"):
        ingest_file(p)


def test_plan_node_replay_whitelist_and_gaps(corpus, monkeypatch):
    """plan 节点 LLM 回放（v0.7 白盒不变量；此前只有 _validate_plan 单测
    与结构冒烟，规划路径的假回放缺失）：白名单外动作只删不造、检查点
    失败的子目标显式进 plan_gaps（不静默吞缺口）、有效子目标证据落池。"""
    import duizhang.llm as llmx
    from duizhang.graph import make_nodes
    plan = make_nodes(corpus)[13]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    replies = iter([
        {"subgoals": [
            {"action": "search", "company": "欧科亿", "period": "Q1",
             "metric": "营收"},
            {"action": "browse", "company": "欧科亿", "period": "Q1"},   # 白名单外：删
            {"action": "search", "company": "不存在的公司", "period": "FY",
             "metric": "净利润"},                                        # 检查点必败：弃
        ]},
    ])
    monkeypatch.setattr(llmx, "chat_json", lambda sys, q, **kw: next(replies))
    st = plan(new_state("对比欧科亿 2026 年一季度营收和净利润", "llm"))
    actions = [e["action"] for e in st["subgoals"]]
    statuses = [e["status"] for e in st["subgoals"]]
    assert "browse" not in actions                       # 只删不造
    assert statuses.count("done") >= 1                   # 欧科亿 Q1 可检索出证据
    assert statuses.count("failed") >= 1                 # 幽灵公司检索不足
    assert st["plan_gaps"] and "不存在的公司" in st["plan_gaps"][0]
    assert any(t.startswith("plan:") and "弃" in t for t in st["trace"])
    assert st["hits"]                                     # 完成子目标的证据在池


def test_plan_node_garbage_output_falls_back(corpus, monkeypatch):
    """规划器输出不可解析（subgoals 非列表）→ 整体回退 agent 环标记，
    不带半成品状态继续走。"""
    import duizhang.llm as llmx
    from duizhang.graph import make_nodes
    plan = make_nodes(corpus)[13]
    monkeypatch.setattr(llmx, "llm_enabled", lambda: True)
    monkeypatch.setattr(llmx, "chat_json",
                        lambda sys, q, **kw: {"subgoals": "垃圾不是列表"})
    st = plan(new_state("跨公司对比题", "llm"))
    assert "plan:invalid→agent" in st["trace"]
    assert not st.get("hits") and not st.get("subgoals")


def test_verify_declares_plan_gaps_not_silent(corpus):
    """部分答案红线（v0.7 Phase 2）：verify:pass 的答案若带着未完成
    子目标，出口必须追加「未覆盖」声明——不允许静默吞掉缺口。"""
    from duizhang.graph import make_nodes
    verify = make_nodes(corpus)[9]
    h = _hit("g1", "欧科亿_2026H1E", 1, 0,
             "预计 2026 年半年度净利润 41,922.46 万元。", 1)
    st = {"answer": "预计净利润 41,922.46 万元 [欧科亿_2026H1E 第1页]",
          "citations": ["[欧科亿_2026H1E 第1页]"],
          "evidence": [h], "hits": [h],
          "plan_gaps": ["不存在的公司 2025FY 的营收"],
          "mode": "llm", "query": "对比题", "trace": []}
    out = verify(st)
    assert out["verdict"] == "pass"
    assert "未覆盖" in out["answer"] and "不存在的公司" in out["answer"]
