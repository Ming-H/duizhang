"""Memory 三层（v0.7 Phase 3，DESIGN-v0.7 §7）。

L0 会话记忆：cli chat 把 history 传进 new_state——近 3 轮全量（q+答案
截断）+ 更早压缩为结构化轮摘要。继承只取 understanding 字段与指标词，
**永不继承自由文本**（公理 3：切断叙述的传播边）。

L1 事实台账：verified_facts 表（store.py），键寻址 (company, period,
metric, caliber)。**写闸** = verdict=pass ∧ 口径闸未触发；**读不豁免**
——命中拼出的答案仍过 verify（公理 2：缓存证据，不缓存结论）。合并
算子 noop/insert/conflict/restated/reingest 见 Store.write_fact；失效
= 语料指纹（probe 时校验，公理 6）。

L2 固定语义资产：tally/assets/semantic_assets.json——alias / 口径
词典等检索导航知识（非答案事实）；晋升闸 = 人工编辑 + eval 全绿。

开关：TALLY_MEMORY（默认开；off = v0.6.5 行为）。eval 默认隔离台账
（TALLY_EVAL_MEMORY=on 才放开）——防止把检索能力测成记忆命中。
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent / "assets" / "semantic_assets.json"

# 指标词典（键抽取用）：问法 → (metric 族, caliber 口径, 检索关键词)。
# 顺序敏感：扣非必须在净利润族之前（词面包含「净利润」）。
_METRIC_LEXICON: tuple[tuple[str, str, str, str], ...] = (
    # 问法片段, metric, caliber, 证据行关键词
    ("扣除非经常", "净利", "扣非", "扣除非经常性损益"),
    ("扣非", "净利", "扣非", "扣除非经常性损益"),
    ("营业总收入", "营收", "营业总收入", "营业总收入"),
    ("总收入", "营收", "营业总收入", "营业总收入"),
    ("营业收入", "营收", "营业收入", "营业收入"),
    ("营收", "营收", "营业收入", "营业收入"),
    ("归属", "净利", "归母", "归属于上市公司股东的净利润"),
    ("归母", "净利", "归母", "归属于上市公司股东的净利润"),
    ("净利", "净利", "归母", "净利润"),
    ("赚", "净利", "归母", "净利润"),
    ("利润", "净利", "归母", "净利润"),
    ("净资产", "资产", "净资产", "归属于上市公司股东的净资产"),
    ("总资产", "资产", "总资产", "总资产"),
)

# 问句里出现过任何指标词 → 无需继承上轮指标
_METRIC_HINTS = ("净利", "营收", "收入", "扣非", "资产", "利润", "赚",
                 "每股", "总资产", "净资产", "毛利率", "现金流")


def enabled() -> bool:
    """TALLY_MEMORY 总闸（默认开）。off = 退回 v0.6.5 无记忆行为。"""
    return os.environ.get("TALLY_MEMORY", "").strip().lower() != "off"


def metric_lexicon(text: str) -> dict | None:
    """问法/标签 → 指标键 {metric, caliber, key}；无命中返回 None。"""
    for pat, metric, caliber, key in _METRIC_LEXICON:
        if pat in text:
            return {"metric": metric, "caliber": caliber, "key": key}
    return None


def metric_families(text: str) -> set[str]:
    """问法命中的指标族集合（Planner 多跳信号用：≥2 族=多指标问题）。"""
    return {metric for pat, metric, _, _ in _METRIC_LEXICON if pat in text}


# caliber → 证据行术语（L1 命中重组答案时的显示词必须与证据行词面
# 一致，verify 的 token 重叠才对得上——显示「归母」而证据行是
# 「净利润」会被自己的闸门误杀）
CALIBER_LABEL = {
    "扣非": "扣除非经常性损益的净利润",
    "归母": "净利润",
    "营业收入": "营业收入",
    "营业总收入": "营业总收入",
    "净资产": "归属于上市公司股东的净资产",
    "总资产": "总资产",
}


# ── L2 固定语义资产 ──────────────────────────────────────


@lru_cache(maxsize=1)
def semantic_assets() -> dict:
    """alias / 口径词典（≤2KB，常驻）。文件缺失 → 空表（功能不依赖）。"""
    try:
        return json.loads(_ASSETS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def alias_of(name: str) -> str | None:
    """简称 → 全称（L2）。D20 的双向子串归一之外的显式别名表。"""
    return semantic_assets().get("alias", {}).get(name)


# ── L1 台账读写（薄封装：读不豁免/写有闸的纪律在 graph 出口）──


def _open_store(corpus):
    db = getattr(corpus, "db_path", "")
    if not db:
        return None
    from tally.store import Store
    return Store(db)          # SCHEMA 幂等：老库首次开自动建 verified_facts


def read_fact(corpus, key: dict) -> tuple[str, dict | None]:
    """探针 → (hit/stale/miss, 行)。stale 已就地 supersede(reingest)。"""
    st = _open_store(corpus)
    if st is None:
        return "miss", None
    try:
        return st.probe_fact(key["company"], key["period"],
                             key["metric"], key["caliber"])
    except Exception:
        return "miss", None
    finally:
        st.close()


def write_fact(corpus, fact: dict) -> str:
    """合并写入 → 算子名（trace 用）。失败静默降级为 skip——台账是
    加速层，写故障不该打断问答主路径。doc_fps 在此回填（当前
    manifest 指纹——读侧失效判定的锚）。"""
    st = _open_store(corpus)
    if st is None:
        return "skip"
    try:
        fps = st.current_fingerprints(fact.get("source_doc_ids") or [])
        fact.setdefault("doc_fps", fps)
        return st.write_fact(fact)
    except Exception:
        return "skip"
    finally:
        st.close()


def fact_citations(corpus, fact: dict) -> list:
    """台账行 citations [{doc_id, page, quote}] → 证据 Chunk 列表
    （quote 即当年 verify 过的证据行——缓存证据非结论的「证据」半边）。"""
    from tally.schemas import Chunk
    out = []
    for c in fact.get("citations") or []:
        page = int(c.get("page") or 0)
        text = str(c.get("quote") or "")
        if text:
            out.append(Chunk(chunk_id=f"mem#{c.get('doc_id', '')}#{page}",
                             doc_id=str(c.get("doc_id", "")), page=page,
                             seq=0, text=text))
    return out


# ── L0 会话记忆（cli chat）───────────────────────────────


def summarize_turn(query: str, st: dict) -> dict:
    """一轮问答 → 轮记录。全量轮带答案截断；append_turn 负责把更早的
    轮压缩掉自由文本（只留结构化字段）。years 进轮摘要——「归母呢」
    类裸追问自己不带年份，不存年份就没法继承（实测掉到别的年报）。"""
    u = st.get("understanding") or {}
    return {"q": query,
            "u": {k: u.get(k) for k in ("intent", "companies", "period", "years")},
            "metric": (st.get("fact_key") or {}).get("caliber", ""),
            "route": st.get("route", ""),
            "verdict": st.get("verdict", ""),
            "a": (st.get("answer") or "")[:400]}


def append_turn(history: list[dict], turn: dict) -> list[dict]:
    """追加并压缩：近 3 轮全量，更早只留结构化摘要（公理 1：事实不可
    压缩、叙述才可压缩——丢的是答案文本，不是理解字段）。窗口 10 轮。"""
    history.append(turn)
    for t in history[:-3]:
        t.pop("a", None)
    return history[-10:]


def inherit_priors(u: dict, history: list[dict],
                   query: str) -> tuple[dict, str, list[str]]:
    """缺省继承（L0）：公司/期间/指标词缺失时从最近一轮补。
    只在字段为空时补（query 提到新公司/新期间时规则层已抽出，不触发）。
    返回 (u, effective_query 追加词, trace 标记)。"""
    markers: list[str] = []
    eff_extra = ""
    prev = None
    for t in reversed(history):
        if t.get("u"):
            prev = t
            break
    if prev is None or not u:
        return u, eff_extra, markers
    pu = prev["u"]
    if not (u.get("companies") or []) and pu.get("companies"):
        u["companies"] = list(pu["companies"])
        markers.append("memory:inherit(company)")
    if not u.get("period") and pu.get("period"):
        u["period"] = pu["period"]
        markers.append("memory:inherit(period)")
    # 年份（v0.7 补面）：u.period 只是后缀（FY/H1…），年份在
    # doc_scope 由 query_years(query) 现抽——裸追问不带年份时会跨年
    # 漂移（「归母净利润呢」掉到上上年年报，实测）。query 自带年份
    # 时不继承。
    if not (u.get("years") or []) and pu.get("years"):
        u["years"] = list(pu["years"])
        markers.append("memory:inherit(year)")
    # 指标词：「那 Q3 呢？」——query 无任何指标词时用上一轮的口径词
    # 补检索（检索词面缺失会空手，理解先验同理）
    if not any(h in query for h in _METRIC_HINTS) and prev.get("metric"):
        eff_extra = prev["metric"]
        markers.append(f"memory:inherit(metric:{prev['metric']})")
    return u, eff_extra, markers
