# 对账 v0.5 设计：Agentic RAG——大模型主导检索策略

| | |
|---|---|
| 状态 | **已实施并真机验收**（2026-10-16，glm-4.5-flash） |
| 范围 | LLM 升级为默认大脑：查询理解节点 + 对比分解节点 + 智谱 provider；对账闸门保持确定性 |
| 前置文档 | `DESIGN-v0.4.md`（FTS5 检索层与 Store 收敛，含 §5 第四轮审查） |
| 实施计划 | `docs/plans/2026-10-16-agentic-rag-v0.5.md` |

## 1. 定位

**Corrective-RAG 架构的 Agentic RAG**：LLM 决定「怎么查」（理解/路由/评分/
改写/分解/生成），确定性代码决定「答案能不能出门」（覆盖率闸门、降级、拒答）。
规则模式（heuristic）保留为零 key 兜底与回归基线——同一张图，两种大脑。

```
START → understand（LLM 结构化理解：intent/companies/period，规则兜底）
      → route（纯函数分发 + 期间守卫）
         ├─ metric   → retrieve（doc_scope=公司+期间）→ grade ⇄ rewrite(≤1)
         │                                              ↓
         ├─ compare  → decompose（按公司拆子查询，交错合并）→ grade
         │                                              ↓
         ├─ summary  → summarize_doc                      generate（LLM/抽取式）
         └─ other    → refuse                              ↓
                                              verify_citations（确定性闸门）
```

## 2. 决策记录

### D1 understand：一次结构化抽取替代 route 内联调用
一次 `chat_json` 拿 {intent, companies, period, reason}：比 v0.4 的
route 分类多了公司与期间，直接驱动 `doc_scope`——v0.4 靠关键词子串猜
（第四轮审查三个 bug 的根源），LLM 理解 + 规则兜底替代。route 退化为
纯函数（省一次调用）。**备选与否决**：每节点独立 LLM 调用（贵一倍、
延迟翻倍）；全规则（回到 v0.4 脆弱性）。

### D2 decompose 只在 compare 意图触发（Adaptive RAG）
多跳分解是自适应的：简单问题走快路径（一次检索），≥2 公司 + 对比词
才分解。子查询**只留指标词**——公司/期间的限定已由 doc_scope 完成，
文本重复公司名只会让页眉块完美匹配挤掉数字块（实测海伦哲 top-4 全是
页眉）。交错合并保证双方证据在场。合并后走 grade→generate→verify，
**多跳不豁免对账闸门**。**备选与否决**：全量问题都分解（简单问题
多花 3-4 倍 token）；独立 planner agent（本任务规模不需要自由规划）。

### D3 闸门不动：自由规划、强制对账
LLM 再主导，出口仍是确定性 `verify_citations`。本轮真机联调对 verify
做了三处**口径**修正（语义不变严）：证据含数字归一化注释；token 两侧
过元词过滤（公司名/年份/期间不构成证据相关性）；只考核数字句且句界
含换行（框架句不考核——编造风险集中在数字句）。LLM 生成失败/无据
→ 降级抽取式而非拒答（与 verify 同一纪律）。

### D4 thinking 分层
结构化小任务（理解/评分/改写）关思考：枚举抽取不需要推理，省时省钱。
生成开思考：碎片化表格证据的数字关联（区分主体区间与「同比增加」
变动幅度）需要推理——flash 档关思考实测把变动区间误当指标值。
`DZ_LLM_THINKING=disabled|enabled` 可整体覆盖。仅智谱端点发送该
扩展参数（deepseek 等兼容端点不认识，绝不附加）。

### D5 provider 与密钥
默认智谱 `open.bigmodel.cn/api/paas/v4` + `glm-4.5-flash`（免费档
开箱可用；充值后 `.env` 改 `glm-4.6` 即提质，代码不动）。密钥只进
`.env`（gitignored），`_load_dotenv` 装载且环境变量优先；测试套件
经 conftest 全局封网（永不外呼）。**备选与否决**：密钥进代码/配置
模板（泄漏风险）；python-dotenv 依赖（12 行手写够用）。

### D6 评估：双数据集
default 集（10 题，CI 回归）不变；agent 集（8 题：3 对比 + 3 单跳/
摘要 + 2 陷阱）新增，judge 支持 `expect_docs` 列表（每个期望文档都
要在场——只中一家不算过）。llm 模式数字有生成方差，口径如实记录。

## 3. 验收（2026-10-16，全量语料 1,104 docs / 60.7 万块）

| 数据集 | 模式 | retrieval@3 | answer_hit | refusal |
|---|---|---|---|---|
| default | heuristic | 1.00 | 1.00 | 1.00 |
| default | llm（flash） | 1.00 | 0.75–0.875（方差） | 1.00 |
| agent | heuristic | 1.00 | 1.00 | 1.00 |
| agent | llm（flash） | **1.00** | **1.00** | **1.00** |

单测 46 用例（新增 understand/decompose/expect_docs/思考分层/.env/
coverage 口径六组）；ruff 全绿；conftest 封网。

## 4. 遗留与后续

- 全量库页眉剥离修复需 `ingest --rebuild` 完整迁移（当前已回填
  评估相关 5 文档；其余文档下次更新时自愈）
- glm-4.6 充值后重跑 agent 集预期 default 集方差收窄
- 向量检索/rerank 仍按 DESIGN-scale §6 触发条件排队
