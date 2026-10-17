import pytest

from duizhang.graph import build_graph
from duizhang.ingest import chunk_page, expand_query, normalize_numbers, parse_filename
from duizhang.schemas import new_state

SAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """走生产路径建库：Store 写入 → Corpus 加载（与 CLI 完全一致）。"""

    from duizhang.index import Corpus
    from duizhang.ingest import ingest_file
    from duizhang.store import Store
    db = tmp_path_factory.mktemp("dz") / "test.db"
    store = Store(db)
    for f in sorted(SAMPLES.glob("*.pdf")) + sorted(SAMPLES.glob("*.md")):
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
