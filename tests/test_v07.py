"""v0.7 Agent 化回归（DESIGN-v0.7 四支柱）。

覆盖：
· Phase 1 步级观测——steps_summary 的 waste_rate（eₖ 代理）口径
· Phase 2 Planner——_wants_plan 触发 / _validate_plan 白名单（只删不造）/
  _ckpt_ok 检查点 / _derived_ok 出口重算；图结构含 plan 与 memory_probe
· Phase 3 Memory——L1 合并算子（noop/insert/conflict/reingest/restated）、
  指纹失效、写闸（verify pass 才落账）、读不豁免（投毒行被 verify 拦下）、
  开关隔离；L0 轮压缩/继承（公司/期间/年份/指标词）；L2 alias
· Phase 4 工具与差分——AGENT_TOOLS 注册表、_parse_metric_row_raw 原币
  精度、分季度横排守卫、噪声行降权、「利润 」断标签修复（answer 与
  graph 两处同步）、_derive_quarters 算术与年份隔离（回归：只按 tag
  匹配会把 2024Q2 派生到 2025 行）、单季 trend 触发词

真值锚点全部来自 data/贵州茅台 + data/中际旭创 的人工核对（与年报
分季度表交叉验证到分位）。
"""

import json
import sqlite3

import pytest

from tally import memory as mem
from tally.answer import split_fix
from tally.graph import (
    AGENT_TOOLS,
    MAX_SUBGOALS,
    _ckpt_ok,
    _derive_quarters,
    _derived_ok,
    _parse_metric_row_raw,
    _pick_metric_line,
    _validate_plan,
    _wants_plan,
    _wants_trend,
    build_graph,
    metric_rows,
    trend_rows,
)
from tally.schemas import Chunk, Retrieved, new_state


def _hit(chunk_id, doc_id, page, seq, text, rank=0):
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, page=page,
                                 seq=seq, text=text), score=0.1, rank=rank)


def _ch(chunk_id, doc_id, page, seq, text):
    """行级选行原语吃 Chunk（trend_rows 内部才做 Retrieved 解包）。"""
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, page=page, seq=seq,
                 text=text)


# ── Phase 1：步级观测 ───────────────────────────────────


def test_steps_summary_waste_rate():
    """waste_rate = (dup+empty)/gates——eₖ 的可测代理。混合 gate 分布
    手算对齐，防口径漂移。"""
    from tally.evalx import summarize
    rows = [{"kw_hit": True, "doc_hit": True, "expect_refused": False,
             "refused": False, "top1": "d"}] * 2
    gates = [("new", 60), ("dup", 20), ("empty", 20), ("new", 100)]
    states = [{"coverage": 0.9, "verdict": "pass",
               "agent_steps": [{"gate": g, "ms": ms} for g, ms in gates]}]
    s = summarize(rows, states, "heuristic", "agent")
    assert s["steps"] == {"calls": 4, "new": 2, "dup": 1, "empty": 1,
                          "waste_rate": 0.5}


def test_agent_steps_schema_in_state():
    """GraphState 带 agent_steps/subgoals/derived/plan_gaps 四字段，
    --json 出口可序列化（TypedDict total=False 的增字段不破老态）。"""
    st = new_state("测试", "heuristic")
    st.update({"agent_steps": [{"i": 0, "thought": "t", "action": "search{}",
                                "obs": "[]", "obs_doc": "", "gate": "new",
                                "ms": 12}],
               "subgoals": [{"action": "search", "company": "c",
                             "period": "Q1", "metric": "营收",
                             "status": "done"}],
               "plan_gaps": ["cQ1的营收"], "derived": []})
    assert json.dumps({k: st[k] for k in
                       ("agent_steps", "subgoals", "plan_gaps", "derived")},
                      ensure_ascii=False)


# ── Phase 2：Planner-Executor ────────────────────────────


def test_wants_plan_triggers():
    """≥3 公司 / 跨年 / ≥2 指标族才规划；单跳不进（不拿单跳题冒险）。"""
    assert _wants_plan("对比甲、乙、丙 2025 年营收", {"companies": ["甲", "乙", "丙"]})
    assert _wants_plan("甲 2024 年和 2025 年营收", {})
    assert _wants_plan("甲 2025 年营收和归母净利润", {})
    assert not _wants_plan("甲 2025 年营收", {})
    assert not _wants_plan("甲 2025 年年报怎么样", {})


def test_validate_plan_whitelist_dedupe_cap(corpus):
    """校验器只删不造：白名单外动作丢、期间规约、去重、上限封顶。"""
    u = {"companies": ["欧科亿"]}
    sgs = [
        {"action": "search", "company": "欧科亿", "period": "半年报", "metric": "营收"},
        {"action": "metrics_table", "company": "欧科亿", "period": "H1",
         "metric": "营收"},                       # 白名单外 → 删
        {"action": "search", "company": "欧科亿", "period": "H1",
         "metric": "营收"},                       # 与第 1 条规约后重复 → 删
        {"action": "search", "company": "欧科亿", "period": "Q1", "metric": ""},
        {"action": "search", "period": "Q3", "metric": "净利"},  # 无公司→单公司回填
        {"action": "search", "company": "欧科亿", "period": "FY", "metric": "m5"},
        {"action": "search", "company": "欧科亿", "period": "FY", "metric": "m6"},
    ]
    out = _validate_plan(sgs, u, "欧科亿 营收 净利", corpus)
    assert all(sg["action"] == "search" for sg in out)
    assert out[0]["period"] == "H1"                       # 半年报 → H1
    assert out[3]["company"] == "欧科亿"                    # 理解回填
    assert len(out) <= MAX_SUBGOALS
    keys = [(sg["company"], sg["period"], sg["metric"]) for sg in out]
    assert len(keys) == len(set(keys))
    assert _validate_plan([], u, "q", corpus) == []        # 空进空出（不造）


def test_ckpt_ok_requires_number_and_metric():
    """检查点：财报级数字在场 ∧ 指标词命中——缺任一不算过。"""
    ok = _hit("a", "d", 1, 0, "营业收入 12,345,678,901.23 同比 10%", 1)
    assert _ckpt_ok([ok], "营业收入")
    no_num = _hit("b", "d", 1, 1, "营业收入增长主要系销量提升", 1)
    assert not _ckpt_ok([no_num], "营业收入")
    no_metric = _hit("c", "d", 1, 2, "总资产 12,345,678,901.23", 1)
    assert not _ckpt_ok([no_metric], "营业收入")


def test_derived_ok_recomputes_operands():
    """出口重算：操作数必须真在场于证据池 + |a−b−value| ≤ 0.01。"""
    pool = [_hit("a", "d", 1, 0,
                 "营业收入 170,899,152,276.34 120,776,131,874.69", 1)]
    good = {"label": "x", "value": 50123020401.65,
            "a": 170899152276.34, "b": 120776131874.69}
    assert _derived_ok([good], pool)
    wrong = {**good, "value": good["value"] + 1.0}        # 超分位容差
    assert not _derived_ok([wrong], pool)
    ghost = {**good, "a": good["a"] + 5}                  # 操作数不在池
    assert not _derived_ok([ghost], pool)
    assert _derived_ok([], pool)                          # 空登记恒过


def test_graph_has_plan_and_memory_nodes(app):
    """结构面：v0.7 两新节点在图里（llm 路的消费由真机 eval 验）。"""
    names = set(getattr(app, "nodes", {}) or {})
    assert {"plan", "memory_probe"} <= names


# ── Phase 3：Memory 三层 ─────────────────────────────────


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """与 test_tally 同构：生产路径建 samples 库。"""
    from pathlib import Path

    from tally.index import Corpus
    from tally.ingest import ingest_file, parse_filename
    from tally.store import Store
    db = tmp_path_factory.mktemp("dz7") / "t.db"
    store = Store(db)
    samples = Path(__file__).resolve().parents[1] / "samples"
    for f in sorted(samples.glob("*.pdf")) + sorted(samples.glob("*.md")):
        try:
            parse_filename(f)
        except ValueError:
            continue
        store.add_doc(*ingest_file(f))
    store.close()
    return Corpus.load(db)


@pytest.fixture(scope="module")
def app(corpus):
    return build_graph(corpus)


@pytest.fixture()
def fact_store(tmp_path):
    from tally.store import Store
    st = Store(tmp_path / "f.db")
    st.add_doc(*_doc_with_fp())
    yield st
    st.close()


def _doc_with_fp():
    from tally.ingest import Doc
    from tally.schemas import Chunk
    doc = Doc(doc_id="欧科亿_2026H1E", company="欧科亿", period="2026H1E",
              kind="业绩预告", source="x.pdf")
    chs = [Chunk(chunk_id="欧科亿_2026H1E#p1#0", doc_id="欧科亿_2026H1E",
                 page=1, seq=0,
                 text="预计 2026 年半年度净利润 36,000 万元")]
    return doc, chs


def _fact(value="36,000", fps=None):
    return {"company": "欧科亿", "period": "2026H1E", "metric": "净利",
            "caliber": "归母", "value_raw": value,
            "value_std": float(value.replace(",", "")),
            "unit": "元", "yoy": "10",
            "citations": [{"doc_id": "欧科亿_2026H1E", "page": 1,
                           "quote": "预计 2026 年半年度净利润 36,000 万元"}],
            "coverage": 1.0, "verdict": "pass", "run_id": "t",
            "source_doc_ids": ["欧科亿_2026H1E"],
            "doc_fps": fps or {"欧科亿_2026H1E": "fp-1"}}


def test_fact_merge_operators(fact_store):
    """noop → insert → conflict（换值 supersede）→ restated/回写幂等。"""
    assert fact_store.write_fact(_fact()) == "insert"
    assert fact_store.write_fact(_fact()) == "noop"       # 同值同指纹幂等
    live = fact_store._fact_row("欧科亿", "2026H1E", "净利", "归母")
    assert live and live["value_raw"] == "36,000"
    # 同键换值 → conflict：旧行 supersede，新行成为唯一活行
    assert fact_store.write_fact(_fact(value="37,000")) == "conflict"
    live = fact_store._fact_row("欧科亿", "2026H1E", "净利", "归母")
    assert live["value_raw"] == "37,000"
    n = fact_store.con.execute(
        "SELECT COUNT(*) FROM verified_facts WHERE superseded_by IS NOT NULL"
    ).fetchone()[0]
    assert n == 1                                          # 追加不物理删


def test_probe_stale_supersedes(fact_store):
    """指纹失效（公理 6）：源文档指纹漂移 → stale，活行就地 supersede。"""
    fact_store.write_fact(_fact())
    with fact_store.con:
        fact_store.con.execute(                    # 模拟重摄取后指纹变化
            "UPDATE verified_facts SET doc_fps=? WHERE superseded_by IS NULL",
            (json.dumps({"欧科亿_2026H1E": "fp-2"}),))
    status, row = fact_store.probe_fact("欧科亿", "2026H1E", "净利", "归母")
    assert status == "stale" and row is None
    status2, _ = fact_store.probe_fact("欧科亿", "2026H1E", "净利", "归母")
    assert status2 == "miss"                          # 已失效不再命中


def test_memory_switch_isolation(monkeypatch):
    """TALLY_MEMORY=off → L1 关闭（eval 隔离语义）；默认开。"""
    monkeypatch.delenv("TALLY_MEMORY", raising=False)
    assert mem.enabled() is True
    monkeypatch.setenv("TALLY_MEMORY", "off")
    assert mem.enabled() is False


def test_write_gate_requires_verify_pass(app, corpus, monkeypatch):
    """写闸集成：verify pass 的问题落账；落账行键与 fact_key 对齐。
    注意样本里的欧科亿 2026H1E 是业绩预告（值=36,000~42,000 万元的
    区间，非单值）——metric_rows 解析不出单值、不落账，这是保守失败
    的正确行为，不是缺陷。"""
    monkeypatch.delenv("TALLY_MEMORY", raising=False)
    st = app.invoke(new_state("海伦哲 2026 年一季度归母净利润多少", "heuristic"))
    assert st["verdict"] == "pass"
    key = {"company": "海伦哲", "period": "2026Q1", "metric": "净利",
           "caliber": "归母"}
    con = sqlite3.connect(corpus.db_path)
    try:
        row = con.execute(
            "SELECT value_raw FROM verified_facts WHERE company=? AND period=?"
            " AND metric=? AND caliber=? AND superseded_by IS NULL",
            (key["company"], key["period"], key["metric"],
             key["caliber"])).fetchone()
    finally:
        con.close()
    assert row and "42,999,086.73" in row[0]


def test_poisoned_fact_read_not_exempt(app, corpus, monkeypatch):
    """读不豁免（公理 2）：投毒行拼出的答案过不了 verify → 掉头全链路，
    最终答案不含毒值。记忆是误差放大器，读侧闸门是保险丝。"""
    monkeypatch.delenv("TALLY_MEMORY", raising=False)
    poison = _fact(value="9,999,999,999.99")
    con = sqlite3.connect(corpus.db_path)
    with con:
        con.execute(
            "INSERT INTO verified_facts (company, period, metric, caliber,"
            " value_raw, value_std, unit, yoy, citations, coverage, verdict,"
            " run_id, source_doc_ids, doc_fps) VALUES (?,?,?,?,?,?,?,?,?,?,"
            "?,?,?,?)",
            (poison["company"], poison["period"], poison["metric"],
             poison["caliber"], poison["value_raw"], poison["value_std"],
             poison["unit"], poison["yoy"],
             json.dumps(poison["citations"], ensure_ascii=False),
             1.0, "pass", "poison", json.dumps(poison["source_doc_ids"]),
             json.dumps(poison["doc_fps"])))
    con.close()
    st = app.invoke(new_state("欧科亿 2026 年半年度预计净利润多少", "heuristic"))
    assert "9,999,999,999.99" not in (st.get("answer") or "")
    assert st.get("verdict") in ("pass", "degraded")   # 全链路照常出口


def test_l0_append_turn_compression():
    """近 3 轮全量、更早只留结构化字段；窗口 10 轮。"""
    h = []
    for i in range(5):
        h = mem.append_turn(h, {"q": f"q{i}", "u": {"companies": ["c"]},
                                "a": f"答案{i}"})
    assert [t.get("a") for t in h] == [None, None, "答案2", "答案3", "答案4"]
    assert all("u" in t for t in h)                    # 理解字段不压缩
    for i in range(5, 12):
        h = mem.append_turn(h, {"q": f"q{i}", "u": {}, "a": "x"})
    assert len(h) == 10


def test_inherit_priors_fields():
    """公司/期间/年份/指标词缺省继承；自带字段不覆盖（只在空时补）。"""
    h = [{"q": "茅台 2025 年报营收", "u": {"companies": ["贵州茅台"],
          "period": "FY", "years": ["2025"]}, "metric": "营业收入"}]
    u = {"intent": "other", "companies": [], "period": "", "years": []}
    u2, eff, marks = mem.inherit_priors(u, h, "那 Q3 呢")
    assert u2["companies"] == ["贵州茅台"] and u2["period"] == "FY"
    assert u2["years"] == ["2025"]
    assert eff == "营业收入"
    assert len(marks) == 4                              # company/period/year/metric
    # query 自带指标词 → 指标不继承（实体照旧）
    u2b, effb, _ = mem.inherit_priors({"intent": "metric", "companies": [],
                                       "period": "", "years": []}, h,
                                      "归母净利润呢")
    assert effb == "" and u2b["companies"] == ["贵州茅台"]
    # 自带新实体：不继承
    u3 = {"intent": "metric", "companies": ["中际旭创"], "period": "H1",
          "years": ["2026"]}
    u4, eff4, marks4 = mem.inherit_priors(u3, h, "中际旭创2026年上半年营收")
    assert u4["companies"] == ["中际旭创"] and eff4 == "" and marks4 == []


def test_summarize_turn_keeps_years():
    st = {"understanding": {"intent": "metric", "companies": ["c"],
                            "period": "FY", "years": ["2025"]},
          "fact_key": {"caliber": "扣非"}, "answer": "a" * 500}
    t = mem.summarize_turn("q", st)
    assert t["u"]["years"] == ["2025"] and t["metric"] == "扣非"
    assert len(t["a"]) == 400                            # 答案截断


def test_alias_of_l2():
    assert mem.alias_of("旭创") == "中际旭创"
    assert mem.alias_of("不存在") is None


def test_l0_year_inherit_drives_scope(app):
    """年份继承链（回归：u.period 只是后缀，年份丢了会跨年漂移）：
    上一轮 2026Q1 → 裸追问 understanding.years=2026。"""
    h = []
    st = app.invoke(new_state("海伦哲2026年一季度营业收入多少", "heuristic",
                              history=h))
    h.append(mem.summarize_turn("海伦哲2026年一季度营业收入多少", st))
    st2 = app.invoke(new_state("净利润呢", "heuristic", history=h))
    u = st2.get("understanding") or {}
    assert u.get("companies") == ["海伦哲"] and u.get("period") == "Q1"
    assert u.get("years") == ["2026"]


# ── Phase 4：工具注册表与差分算子 ────────────────────────


def test_agent_tool_registry():
    """三工具注册（search/metrics_table/derive），schema 可序列化。"""
    names = [t["function"]["name"] for t in AGENT_TOOLS]
    assert names == ["search", "metrics_table", "derive"]
    for t in AGENT_TOOLS:
        json.dumps(t)                                    # schema 合法 JSON


def test_parse_metric_row_raw_precision():
    """原币精度 + D26 双列：单列取首大数；cumulative 取累计列。"""
    raw, yi, pct = _parse_metric_row_raw(
        "营业收入 38,845,154,206.94 15.29 120,776,131,874.69 16.95")
    assert raw == 38845154206.94 and pct == "15.29%"
    raw2, _, pct2 = _parse_metric_row_raw(
        "营业收入 38,845,154,206.94 15.29 120,776,131,874.69 16.95", True)
    assert raw2 == 120776131874.69 and pct2 == "16.95%"


def test_pick_metric_line_quarterly_guard():
    """分季度横排（4 大数 0 同比）不是 as-reported 行——拒（宁可丢行
    不可取错行：457.8 亿＝Q1 顶掉 1,709 亿实测）。真表行带同比放行。"""
    bad = _ch("a", "d", 6, 5,
               "营业收入 45,775,517,043.29 36,155,460,624.46 "
               "38,845,154,206.94 50,123,020,401.65")
    assert _pick_metric_line([bad], "营业收入") is None
    good = _ch("b", "d", 5, 5,
                "2022年 营业收入 170,899,152,276.34 147,693,604,994.14 "
                "15.71 124,099,843,771.99")
    seg = _pick_metric_line([good], "营业收入")
    assert seg and _parse_metric_row_raw(seg)[0] == 170899152276.34


def test_pick_metric_line_noise_words():
    """分红/留存段引用本期值的行降权（年均 745.6 亿顶掉归母 862.3 亿）。"""
    noise = _ch("a", "d", 35, 8,
                 "最近三个会计年度年均净利润金额(4) 74,559,895,280.83 "
                 "现金分红比例 251.52")
    assert _pick_metric_line([noise], "净利润") is None


def test_split_fix_profit_prefix_synced():
    """「利润 86,228…」断标签修复：answer.split_fix 与 graph pick 两处
    同步（invariant：三处修复表不同步会让归母行在某一路径不可读）。"""
    line = "利润 86,228,146,421.62 74,734,071,550.75 15.38 62,717,467,870.12"
    fixed = split_fix(line)
    assert fixed.startswith("归属于上市公司股东的净利润")
    ch = _ch("a", "d", 5, 7, line)
    seg = _pick_metric_line([ch], "净利润")
    assert seg and _parse_metric_row_raw(seg)[0] == 86228146421.62
    # 非断标签行不误修
    assert split_fix("利润总额 30,303,850,168.56") == "利润总额 30,303,850,168.56"


def _entry(doc_id, rows):
    period = doc_id.rsplit("_", 1)[-1]
    return {"doc_id": doc_id, "period": period, "tag": period[-2:],
            "cumulative": period[-2:] == "Q3", "rows": rows,
            "citation": "[x 第1页]", "page": 1}


def _row(label, raw):
    return {"label": label, "key": "营业收入" if label == "营收" else "净利润",
            "raw": raw, "yi": f"{raw / 1e8:,.1f} 亿", "yoy": "10%",
            "seg": "..."}


def test_derive_quarters_arithmetic_and_year_isolation():
    """Q2=H1−Q1 / Q4=FY−Q3；年份隔离（回归：只按 tag 匹配把 2024Q2
    派生到 2025 行，三年全是同一个 387.9 亿）。"""
    entries = [
        _entry("X_2024Q3", [_row("营收", 120776131874.69)]),
        _entry("X_2024FY", [_row("营收", 170899152276.34)]),
        _entry("X_2025Q1", [_row("营收", 50600957885.78)]),
        _entry("X_2025H1", [_row("营收", 89389354416.84)]),
        _entry("X_2025Q3", [_row("营收", 128453707655.86)]),
        _entry("X_2025FY", [_row("营收", 168838102514.79)]),
    ]
    items, derived = _derive_quarters(entries)
    got = {d["label"]: d["value"] for d in derived}
    assert got["X_2025Q2营收"] == pytest.approx(38788396531.06, abs=0.01)
    assert got["X_2025Q4营收"] == pytest.approx(40384394858.93, abs=0.01)
    assert got["X_2024Q4营收"] == pytest.approx(50123020401.65, abs=0.01)
    # 2024 无 H1/Q1 行 → 2024Q2 不派生（缺操作数保守跳过，绝不打错数）
    assert not any("2024Q2" in k for k in got)
    # 出口重算恒等：a−b=value 分位对齐
    for d in derived:
        assert abs((d["a"] - d["b"]) - d["value"]) <= 0.01


def test_derive_quarters_negative_skipped():
    """操作数倒挂（选错行特征）→ 跳过该行，不出错数。"""
    entries = [
        _entry("X_2025Q1", [_row("营收", 900.0)]),
        _entry("X_2025H1", [_row("营收", 500.0)]),       # H1 < Q1 不可能
    ]
    items, derived = _derive_quarters(entries)
    assert not derived and not items


def test_quarter_question_routes_trend():
    """第二/第四季度需差分 → trend；第一/第三季度直接披露 → metric。"""
    assert _wants_trend("贵州茅台2024年第四季度营收")
    assert _wants_trend("茅台 2025 二季度净利")
    assert _wants_trend("茅台 2025年单季营收")
    assert not _wants_trend("海伦哲第一季度营业收入多少")
    assert not _wants_trend("海伦哲2026年三季度净利")


def test_metric_rows_skeleton_primitives():
    """metric_rows：单一解析原语产 {label, raw, yi, yoy}（四方共用）。"""
    pool = [_ch("a", "d", 1, 0,
                 "营业收入 50,600,957,885.78 45,775,517,043.29 10.54 "
                 "47,775,449,806.87 归属于上市公司股东的净利润 "
                 "26,847,474,238.76 24,065,262,374.15 11.56")]
    rows = metric_rows(pool)
    by_label = {r["label"]: r for r in rows}
    assert by_label["营收"]["raw"] == 50600957885.78
    assert by_label["归母净利"]["yoy"] == "11.56%"


def test_trend_rows_no_corpus_page_expansion():
    """corpus=None（无页扩展）时池=命中块本身——trend_rows 不硬依赖。"""
    hits = [_hit("a", "X_2025Q1", 1, 0,
                 "营业收入 50,600,957,885.78 45,775,517,043.29 10.54")]
    entries = trend_rows(hits, None)
    assert len(entries) == 1 and entries[0]["tag"] == "Q1"
    assert entries[0]["rows"][0]["raw"] == 50600957885.78
