# 对账 v0.6 设计：真 Agent 环——模型自主检索

| | |
|---|---|
| 状态 | **已实施并真机验收**（2026-10-16，qwen3.5:0.8b 本地） |
| 范围 | llm 模式下 metric/compare 从固定管线改为工具循环（模型自主决定查什么、何时收手）；heuristic 固定管线原样保留 |
| 前置 | `DESIGN-v0.5.md`（Agentic RAG 图与 §5 v0.5.1 本地化） |

## 1. 定位修正

v0.5 是「LLM 填决策点的固定图」（agentic RAG 模式），拓扑编译期写死——
按 Anthropic 定义不是自主 agent。v0.6 把检索阶段交给模型：单一 search
工具 + 循环，模型自己决定查什么、查几次、何时收手。**出口闸门不动**：
agent 收集的证据照样过 generate→verify，自主检索不豁免对账。

```
llm 模式：  understand → route ─metric/compare→ agent_research（工具循环）→ generate → verify
heuristic： understand → route ─metric→ retrieve / compare→ decompose → grade⇄rewrite → generate → verify
```

## 2. 决策记录（每条都是真机踩出来的）

### D9 单一工具 + 原生 tools 协议（防过度设计）
qwen3.5:0.8b 经 Ollama 原生 tools 实测可靠（正确选工具、结构化参数）。
刻意不建工具注册表：一个 `search(query, company, period)` 已覆盖
metric/compare 真实需要；read_doc/list_docs 是 YAGNI。步数硬上限 6
（业界惯例：agent 循环必须有预算）。

### D10 幂等缓存 + 重复收手
0.8b 实测死循环：同一调用原样重复 6 步烧满预算。修复：调用签名去重
缓存，重复调用回缓存结果并提示换角度；**整批全重复 = 模型没有新问题
要问 → 提前收手**（6步12次 → 2步2次）。

### D11 理解先验优先于工具参数
小模型三次实测事故定义了工具默认值语义：
- 不传 company（单跳题）→ 全库检索被千家同类内容淹没 → **从
  understanding 继承**；多公司 understanding → 工具内逐家检索交错合并
- 幻觉 period（「半年度」传成 FY）→ 年报块洪水 → **期间以理解为准、
  参数只补空白**——理解是模型对原问题的结构化决定（含规则回填），
  工具参数是检索战术，冲突时先验赢
- 抽 period 时灵时不灵 → 空值从规则确定性匹配回填（与公司名纠偏对称）

原则：廉价结构化理解做先验，模型自主性用在「迭代检索策略」上——
这正是 0.8b 级小模型能可靠承担的那部分自主性。

### D12 证据按文档轮转 + Agent 检索词写进 effective_query
pool 插入序会让先查的一方包揽前排（单边答案，decompose 同款教训）；
原始口语 query 与证据零重叠会误拒（「哪家赚得多」没有指标词）——
Agent 实际用的检索词必须写进 effective_query 供下游打分。

## 3. 验收（qwen3.5:0.8b 本地，全量语料）

| 数据集 | retrieval@3 | answer_hit | refusal |
|---|---|---|---|
| agent（Agent 环） | **1.00** | **1.00** | **1.00** |
| default（Agent 环） | 0.875 | 0.875 | 1.00 |
| agent（heuristic 固定管线，回归） | 1.00 | 1.00 | 1.00 |

单测 52 用例（新增工具循环六场景：循环/上限/回退/重复收手/轮转/
先验优先）。已记录的 0.8b 已知局限：比较方向偶发口误（数字与引用
全对、结论句说反）——token 重叠闸门管不住语义级关系错误，换强模型
消除；提示词已加「对比结论必须与所引数字的大小关系一致」缓解。

## 4. 面试叙事

「v0.5 是 agentic RAG（固定图+模型填决策点），v0.6 把检索交给模型
变成真 agent 环——单一工具、步数预算、幂等缓存；小模型的三类事故
（不传参/幻觉期间/死循环）各有一条针对性防线，且全部可追溯到实测
trace。出口闸门从头到尾没动过：自主检索不豁免对账。」
