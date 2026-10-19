"""对账 duizhang — 核心数据模型。

命名「对账」：RAG 的每个答案都要能对账——逐句挂引用，引用必可回溯到
财报原文的页与段；对不上的答案宁可拒答，也不编。
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

# ── 语料侧 ──────────────────────────────────────────────


@dataclass
class Doc:
    """一份已入库文档（一份财报 PDF / 一篇分析 MD）。"""

    doc_id: str            # 稳定 id：公司_报告期 的 slug
    company: str           # 公司名（从文件名/文本提取）
    period: str            # 报告期，如 2026H1 / 2026Q1 / 2025FY
    kind: str              # 报告类型：年报/季报/业绩预告/分析
    source: str            # 原始路径
    n_pages: int = 0


@dataclass
class Chunk:
    """检索的最小单元。表格作为原子块，绝不拦腰切断。"""

    chunk_id: str          # doc_id#p{page}#{seq}
    doc_id: str
    page: int              # PDF 页码（MD 记 0）
    seq: int               # 页内序号
    text: str
    is_table: bool = False
    # 数字归一化注释：把「2,345,678,901.23 元」写成「23.46 亿」附加在块尾，
    # 让词法检索对「多少亿」类问题直接可命中（句读的基因）。
    num_notes: list[str] = field(default_factory=list)

    @property
    def citation(self) -> str:
        return f"[{self.doc_id} 第{self.page}页]" if self.page else f"[{self.doc_id}]"


# ── 检索/生成侧 ─────────────────────────────────────────


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    rank: int


Route = Literal["metric", "summary", "other"]


class GraphState(TypedDict, total=False):
    """LangGraph 全局状态：一条问答在图中的全部痕迹。

    设计原则：每个节点只读它需要的字段、只写它负责的字段——
    图的可解释性来自状态的窄接口。
    """

    # 输入
    query: str
    mode: str                          # heuristic | llm（auto 在 new_state 解析）
    # understand 节点（v0.5：LLM 查询理解，规则兜底）
    understanding: dict                # {intent, companies, period, reason}
    # route 节点
    route: Route
    route_reason: str
    # retrieve 节点
    hits: list[Retrieved]
    # grade 节点
    relevance: float                   # 命中块与 query 的相关度分
    sufficient: bool
    retries: int                       # 改写重试计数（≤1）
    # rewrite 节点
    effective_query: str               # 实际用于检索的 query（含改写）
    rewritten: bool
    # generate 节点（v0.6.1：页级证据，verify 闸门的证据池）
    evidence: list                     # 命中块 ±2 窗口的同页块集
    # generate / summarize / refuse 节点
    answer: str
    citations: list[str]
    # verify 节点
    verdict: Literal["pass", "degraded", "refused"]
    coverage: float                     # 引用覆盖率（faithfulness 报告指标用）
    trace: Annotated[list[str], operator.add]   # 执行轨迹（append 语义）


def new_state(query: str, mode: str = "auto") -> GraphState:
    """mode=auto：有 key 走 llm（LLM 是大脑），无 key 走 heuristic（兜底）。"""
    if mode == "auto":
        from duizhang import llm
        mode = "llm" if llm.llm_enabled() else "heuristic"
        trace = ["start", f"mode:{mode}"]
    else:
        trace = ["start"]
    return GraphState(
        query=query,
        effective_query=query,
        mode=mode,
        retries=0,
        rewritten=False,
        trace=trace,
    )
