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
    st = _ask(app, "欧科亿 2030 年营收预测")
    assert st["verdict"] == "refused"
    assert "没有" in st["answer"]


def test_period_aware_retrieval(app):
    st = _ask(app, "海伦哲第一季度营业收入多少")
    assert st["hits"][0].chunk.doc_id == "海伦哲_2026Q1"


def test_corrective_loop_bounded(app):
    st = _ask(app, "欧科亿一季报归母净利润多少")
    assert st.get("retries", 0) <= 1          # 纠错环有界
    assert st["verdict"] in ("pass", "degraded")


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
