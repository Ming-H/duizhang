"""evalx 新指标：chunk 级 P/R + citation_coverage（P1.3；原名
faithfulness，v0.7 后期改名——该指标测的是「答案数字句能否在证据池
找到出处」（引用覆盖率 ≥0.45 的 verify pass 占比），不是语义级
faithfulness，名不副实的名字在面试/评审里会被追问拆穿）。

操作性定义（关键词无全量 relevant 标注，写清近似边界）：
- 真值承载块 = doc ∈ expect_docs 且 text 含任一 expect_keywords
- P@8 = top-8 中真值承载块占比（检索噪声的下界度量）
- R@8 = top-8 内出现 ≥1 真值承载块记 1（命中率下界——同一数字常在
  叙述/表格多处出现，全量召回无标注支撑）
- citation_coverage = 非拒答题中 verify pass 占比 + 平均引用覆盖率
  （coverage 自 v0.6.3 起写入 GraphState）
"""

from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _hit(chunk_id, doc_id, text, rank):
    from tally.schemas import Chunk, Retrieved
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, page=1,
                                 seq=0, text=text), score=0.1, rank=rank)


def test_chunk_pr_counts_truth_bearing_chunks():
    from tally.evalx import chunk_pr
    case = {"q": "x", "expect_keywords": ["1,568,385,287"],
            "expect_doc": "中际旭创_2025Q1"}
    hits = ([_hit("t1", "中际旭创_2025Q1",
                  "扣除非经常性损益的净利润 1,568,385,287.19", 1),
             _hit("n1", "中际旭创_2025Q1", "无关叙述块。", 2)] +
            [_hit(f"n{i}", "中际旭创_2025FY", "年报别的期间内容。", i)
             for i in range(3, 9)])
    out = chunk_pr(case, {"hits": hits})
    assert out["p"] == pytest.approx(0.125)   # 8 块中 1 块真值承载
    assert out["r"] == 1.0                     # 命中即召回
    # 真值块缺席 → r=0
    miss = chunk_pr(case, {"hits": hits[1:]})
    assert miss["r"] == 0.0
    # 拒答题/无关键词题不计入
    assert chunk_pr({"q": "x", "expect_refused": True}, {"hits": hits}) is None
    assert chunk_pr({"q": "x"}, {"hits": hits}) is None


def test_summarize_reports_faithfulness_and_chunk_pr():
    from tally.evalx import summarize
    rows = [dict(q="a", top1="d", kw_hit=True, doc_hit=True, refused=False,
                 ok=True, expect_refused=False),
            dict(q="b", top1="", kw_hit=False, doc_hit=True, refused=True,
                 ok=True, expect_refused=True)]
    states = [dict(verdict="pass", coverage=0.9, hits=[]),
              dict(verdict="refused", coverage=0.0, hits=[])]
    s = summarize(rows, states, mode="heuristic", dataset="innolight")
    assert s["citation_coverage"] == pytest.approx(1.0)   # 非拒答 1/1 pass
    assert s["mean_coverage"] == pytest.approx(0.9)
    assert "chunk_p@8" in s and "chunk_r@8" in s
    assert s["refusal"] == 1.0                            # 陷阱题 1/1 拒
    # 无陷阱题（multiturn 类接力集）：refusal 口径不适用 → None（N/A），
    # 不打 0.0（会被未来 CI 门槛误判）
    s2 = summarize([rows[0]], [states[0]], mode="heuristic", dataset="multiturn")
    assert s2["refusal"] is None


def test_gate_failures_below_threshold():
    """CI 门槛（纯函数）：低于阈值返回失败说明，空列表=过闸。
    修前状态：cmd_eval 永远 exit 0，CI 三集 eval 只防崩溃不防退化——
    命中率跌到 0.3 CI 依然全绿（评估报告发现的头号名实差距）。"""
    from tally.evalx import gate_failures
    ok_summary = dict(retrieval_top3=1.0, answer_hit=1.0, refusal=1.0)
    assert gate_failures(ok_summary, min_retrieval=1.0, min_hit=1.0) == []
    bad = dict(retrieval_top3=0.9, answer_hit=0.3, refusal=1.0)
    fails = gate_failures(bad, min_retrieval=1.0, min_hit=1.0)
    assert len(fails) == 2                       # 两项各报一条
    assert any("retrieval" in f for f in fails)
    assert any("answer_hit" in f for f in fails)
    # refusal=None（无陷阱题集）不判——口径不适用，硬判会把 N/A 打成 0
    assert gate_failures(dict(retrieval_top3=1.0, answer_hit=1.0, refusal=None),
                         min_retrieval=1.0, min_hit=1.0, min_refusal=1.0) == []


def test_judge_numeric_keyword_full_match():
    """数字关键词只守左边界：防「3.6」以子串命中「13.68」、「36,000」
    命中「136,000」（修前 OR 子串语义可被页码/日期/邻近值巧合骗过）。
    右侧放行=数据集前缀截断约定（51.7 对 51.71、38,239 对
    38,239,935,640）——右边界一禁即误杀（innolight 实测 0.963，
    CI 门槛当场抓回，回归锚点）。"""
    from tally.evalx import judge

    def st(a):
        return {"answer": a, "hits": [], "citations": [], "verdict": "pass"}

    # 左粘连是假命中：3.6 不命中 13.68；36,000 不命中 136,000
    case = {"q": "x", "expect_keywords": ["3.6"]}
    assert judge(case, st("归母净利润 13.68 亿元"))["kw_hit"] is False
    assert judge(case, st("约 3.6 亿"))["kw_hit"] is True
    case2 = {"q": "x", "expect_keywords": ["36,000"]}
    assert judge(case2, st("营收 136,000 万元"))["kw_hit"] is False
    assert judge(case2, st("营收 36,000 万元"))["kw_hit"] is True
    # 右侧延续=前缀截断约定，必须命中（精度延续 + 千分位延续）
    assert judge({"q": "x", "expect_keywords": ["51.7"]},
                 st("净利润 51.71 亿元"))["kw_hit"] is True
    assert judge({"q": "x", "expect_keywords": ["38,239"]},
                 st("营业收入 38,239,935,640.67 元"))["kw_hit"] is True
    assert judge({"q": "x", "expect_keywords": ["1,568,385,287"]},
                 st("扣非净利润 1,568,385,287.19 元"))["kw_hit"] is True
    # 非数字关键词保持子串语义（公司名/指标词）
    case4 = {"q": "x", "expect_keywords": ["净利润"]}
    assert judge(case4, st("归母净利润为"))["kw_hit"] is True


def test_judge_forbid_keywords_vetoes():
    """forbid_keywords 反向断言：答案出现明确不该出现的值（问扣非却
    答了归母数）→ 该题判 miss——把单测层的负向断言能力上移到 eval 层。"""
    from tally.evalx import judge
    case = {"q": "扣非净利润多少", "expect_keywords": ["1,568,385,287"],
            "forbid_keywords": ["1,582,876,128"]}
    good = {"answer": "扣非净利润 1,568,385,287.19 元", "hits": [],
            "citations": [], "verdict": "pass"}
    bad = {"answer": "净利润 1,582,876,128.67 元", "hits": [],
           "citations": [], "verdict": "pass"}
    assert judge(case, good)["ok"] is True
    assert judge(case, bad)["ok"] is False       # forbid 命中一票否决
    # forbid 未命中时正常过（回归护栏）
    assert judge(case, good)["kw_hit"] is True
