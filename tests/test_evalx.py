"""evalx 新指标：chunk 级 P/R + faithfulness（P1.3）。

操作性定义（关键词无全量 relevant 标注，写清近似边界）：
- 真值承载块 = doc ∈ expect_docs 且 text 含任一 expect_keywords
- P@8 = top-8 中真值承载块占比（检索噪声的下界度量）
- R@8 = top-8 内出现 ≥1 真值承载块记 1（命中率下界——同一数字常在
  叙述/表格多处出现，全量召回无标注支撑）
- faithfulness = 非拒答题中 verify pass 占比 + 平均引用覆盖率
  （coverage 自 v0.6.3 起写入 GraphState）
"""

from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def _hit(chunk_id, doc_id, text, rank):
    from duizhang.schemas import Chunk, Retrieved
    return Retrieved(chunk=Chunk(chunk_id=chunk_id, doc_id=doc_id, page=1,
                                 seq=0, text=text), score=0.1, rank=rank)


def test_chunk_pr_counts_truth_bearing_chunks():
    from duizhang.evalx import chunk_pr
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
    from duizhang.evalx import summarize
    rows = [dict(q="a", top1="d", kw_hit=True, doc_hit=True, refused=False,
                 ok=True, expect_refused=False),
            dict(q="b", top1="", kw_hit=False, doc_hit=True, refused=True,
                 ok=True, expect_refused=True)]
    states = [dict(verdict="pass", coverage=0.9, hits=[]),
              dict(verdict="refused", coverage=0.0, hits=[])]
    s = summarize(rows, states, mode="heuristic", dataset="innolight")
    assert s["faithfulness_pass"] == pytest.approx(1.0)   # 非拒答 1/1 pass
    assert s["mean_coverage"] == pytest.approx(0.9)
    assert "chunk_p@8" in s and "chunk_r@8" in s
    assert s["refusal"] == 1.0                            # 陷阱题 1/1 拒
    # 无陷阱题（multiturn 类接力集）：refusal 口径不适用 → None（N/A），
    # 不打 0.0（会被未来 CI 门槛误判）
    s2 = summarize([rows[0]], [states[0]], mode="heuristic", dataset="multiturn")
    assert s2["refusal"] is None
