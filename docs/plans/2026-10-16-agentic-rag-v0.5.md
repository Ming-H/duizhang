# v0.5 Agentic RAG 重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 duizhang 升级为 LLM 主导的 Agentic RAG（Corrective-RAG 模式）：LLM（智谱 GLM）做查询理解/路由/评分/改写/生成，新增多公司对比的查询分解；规则模式降级为兜底与回归基线；`verify_citations` 确定性闸门保持不变。

**Architecture:** 在现有 LangGraph 图的入口加 `understand` 节点（LLM 一次结构化抽取 intent/companies/period，规则回退），`route` 变纯函数分发并新增 `compare → decompose` 分支（按公司拆子查询、各自 doc_scope 检索、合并证据），后续 grade/generate/verify 复用既有路径。密钥经 `.env`（gitignored）装载，绝不入库。

**Tech Stack:** LangGraph StateGraph / SQLite FTS5 / openai SDK（OpenAI 兼容端点：`https://open.bigmodel.cn/api/paas/v4/`，模型 `glm-4.6`）/ pytest。

**安全红线：** API key 只存 `.env`（已加 .gitignore）；push 前 `git grep` 扫描新增提交历史中的 key 前缀；文档/README 只写变量名不写值。

---

### Task 1: 密钥装载与 zhipu provider 默认值（llm.py + .env + .gitignore）

**Files:** Modify `duizhang/llm.py`, `.gitignore`; Create `.env`（本地，不提交）

1. `.gitignore` 追加 `.env`（先做，防误提交）→ 写 `.env`（DZ_LLM_API_KEY / DZ_LLM_BASE_URL / DZ_LLM_MODEL=glm-4.6）
2. TDD：
   - test：`_load_dotenv(path)` 读 tmp .env → 环境获得键值且不覆盖已有环境变量
   - test：默认 base_url 含 `open.bigmodel.cn/api/paas/v4`、默认 model `glm-4.6`
   - test：`_parse_json` 剥 ```json 围栏、取首个 {...}
   - test：`_chat_kwargs()`（私有）对 bigmodel 端点附加 `extra_body={"thinking":{"type":"disabled"}}`，其它端点不附加；`DZ_LLM_THINKING=enabled` 时不附加
3. 实现：模块导入时 `_load_dotenv(Path(".env"))`（不存在则跳过）；`chat_json` 用 `_parse_json`；`chat_*` 组装 extra_body
4. 全套测试绿 → commit `feat(llm): 智谱 GLM 默认 provider + .env 装载 + JSON 容错解析`

### Task 2: auto 模式解析（schemas.py / cli.py / graph.ask）

**Files:** Modify `duizhang/schemas.py`（`new_state` 支持 auto）、`duizhang/graph.py:ask`、`duizhang/cli.py`（choices 加 auto 且默认 auto）

- test：无 key 时 `new_state("q","auto")` 解析为 heuristic；monkeypatch `llm_enabled→True` 时解析为 llm；trace 记录 `mode:heuristic(auto)`
- CLI `ask/eval --mode {auto,llm,heuristic}` 默认 auto
- commit `feat(cli): --mode auto——有 key 走 llm，无 key 走 heuristic`

### Task 3: understand 节点（LLM 结构化查询理解，规则回退）

**Files:** Modify `duizhang/graph.py`、`duizhang/schemas.py`（state 加 `understanding`）

- 规则版 `_rule_understand(query, corpus)`：公司=语料公司名全部命中（多公司支持，替代只回首个的 `_doc_filter`）、period=`period_hints` 首个、intent=既有摘要/指标词规则 + compare 判定（「对比/比较/vs/哪家」且公司≥2）
- LLM 版：一次 `chat_json` 抽取 `{"intent":"metric|summary|compare|other","companies":[...],"period":"H1|Q1|Q3|FY|","reason":"..."}`；异常 → 规则回退 + trace `understand:llm→fallback`
- test：多公司规则抽取（欧科亿+海伦哲 query → 两家 + compare intent）；LLM 返回驱动 understanding；LLM 抛错回退规则且 trace 留痕；图级：`START→understand` 先于 route 执行
- commit `feat(graph): understand 节点——LLM 结构化查询理解，规则回退`

### Task 4: route 纯函数化 + compare 分支 + retrieve 消费 understanding

**Files:** Modify `duizhang/graph.py`

- route：消费 `understanding["intent"]` 分发 metric/summary/compare/other；期间守卫（年份 vs 语料）保留；route 不再直接调 LLM（省一次调用，heuristic 关键词逻辑已上移 understand）
- retrieve：`doc_scope` 由 understanding 的 companies+period 构建（公司名单精确匹配 docs.company，子串匹配仍兜底 doc_id）；无公司则不过滤；scoped 空回退 raw（语义同 v0.4.1）
- 条件边：route → {retrieve, summarize_doc, decompose, refuse}
- test：route 按 intent 四路分发；retrieve 的 doc_scope 来自 understanding（FakeCorpus 记录参数断言）；无 understanding（旧调用兼容）回退规则抽取
- commit `feat(graph): route 消费 understanding；retrieve 按公司+期间限定 doc_scope`

### Task 5: decompose 多跳分解节点

**Files:** Modify `duizhang/graph.py`

- intent=compare：每公司子查询 `f"{company} {原query}"` → expand_query → corpus.search(doc_scope=该公司+期间, k=4) → 按公司交错合并 → `state["hits"]`；trace `decompose:A|B → N hits`
- 边：decompose → grade_docs（复用纠错环/生成/verify）
- test：FakeCorpus 两公司 → hits 覆盖两家且各 ≤4；某公司无命中 → 仍有另一家（不空手）
- commit `feat(graph): compare 查询分解——按公司 doc_scope 检索并合并证据`

### Task 6: eval 双数据集 + judge expect_docs

**Files:** Modify `duizhang/evalx.py`、`duizhang/cli.py`；Create `evals/dataset-agent.json`

- `run_eval(corpus, mode, dataset_path=None)`；CLI `eval --dataset {default,agent}`
- judge 支持 `expect_docs` 列表：每个期望 doc 都需在 hits 或 citations 中出现
- dataset-agent.json：2 道对比题（欧科亿vs海伦哲 2026Q1 营收；欧科亿vs日联科技 H1E）+ 3 道单跳/摘要 + 2 道拒答陷阱 + 1 道 compare 口语变体
- test：judge expect_docs 全中/缺一
- commit `feat(eval): agent 评估集与 expect_docs 判定`

### Task 7: 真实 GLM 冒烟 + agent 模式跑分（花钱，约 30–60 次调用）

- 单次 `chat_json` 连通性 → `ask --mode llm` 单跳 → 对比题 → `eval --mode llm --dataset agent`
- 数字记录进 README（标注「手动实测」与运行区间）；失败题逐条看 trace 定位提示词问题并迭代（≤2 轮）

### Task 8: 文档与收尾

- Create `docs/DESIGN-v0.5.md`（决策记录：understand 单次调用 vs route+filter 两次、thinking 默认关、多跳仅 compare、闸门不动）
- README：Agentic RAG 重定位 + 双模式架构图 + 双数据集数字 + 密钥安全说明；EXPLAINER 加「定位」节；CLAUDE.md 同步；`__version__ = "0.5.0"`
- commit `docs: v0.5 Agentic RAG 设计记录与重定位`
- **push 前**：`git log -p origin/main..main | grep -c <key前缀>` 必须为 0；`git status` 确认 .env 未跟踪 → push
