import pytest

from duizhang.ingest import chunk_page, normalize_numbers, parse_filename, expand_query
from duizhang.index import Corpus
from duizhang.graph import build_graph
from duizhang.schemas import new_state

SAMPLES = __import__("pathlib").Path(__file__).resolve().parents[1] / "samples"


@pytest.fixture(scope="module")
def corpus():
    from duizhang.ingest import ingest_file
    corp = Corpus()
    for f in sorted(SAMPLES.glob("*.pdf")) + sorted(SAMPLES.glob("*.md")):
        corp.add(*ingest_file(f))
    return corp


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


def test_eval_dataset_passes_thresholds(corpus):
    from duizhang.evalx import run_eval
    s = run_eval(corpus, mode="heuristic", verbose=False)
    assert s["answer_hit"] >= 0.9
    assert s["refusal"] == 1.0
    assert s["retrieval_top3"] >= 0.75
