# 对账 duizhang 测试用例手册

> 版本：v0.6.2（2026-10-18）。工作库 = 中际旭创三年十期（data/中际旭创，
> 7,873 chunks）；LLM = 本地 Ollama qwen3.5:0.8b（默认 auto 模式）。
> 全部命令在项目根目录执行；带 `DZ_DB=` 的用临时库，不影响工作库。

## 0. 测试分层

| 层 | 命令 | 耗时 | 裁判作用 |
|---|---|---|---|
| 单元测试 | `pytest`（67 用例，conftest 封网永不调 LLM） | ~5s | 代码正确性 + 历史 bug 回归 |
| 评估集 | `eval`（双数据集 × 双模式，samples 库上跑） | ~2min | 端到端质量门槛（见 §3） |
| 手工验收 | `ask`（本文 §1 用例，看答案 + trace） | ~10s/题 | 真实体验与 trace 行为 |
| CI | push 自动（ruff + pytest + 30 秒冒烟） | ~45s | 最后防线 |

---

## 1. 手工验收用例（工作库：中际旭创）

### A. 跨期趋势（trend，v0.6.2 新增）

| 编号 | 问句 | 预期 | 关键检查点 |
|---|---|---|---|
| MC-A1 | `中际旭创最近几个季度营收的趋势，值得投资吗` | 六期骨架（2024Q3→2026H1）营收+净利+同比，每期真实引用，尾部免责声明 | trace 含 `route:trend → trend:6期 → generate:trend → verify:pass`；2026H1 营收 41,777,861,795.03（+182.49%）必须在场 |
| MC-A2 | `中际旭创最近几个季度利润的走势` | 同上（走势=trend 触发词） | 每期净利数字与引用一致；相邻重复数字已折叠（追溯调整双列） |
| MC-A3 | `欧科亿 2026 年半年度预计净利润多少` | **metric 单期作答**（无趋势词不得判 trend） | trace `route:metric`；答案含 36,000/42,000 区间或 ≈4.19亿 |

### B. 指标问答（metric + 过滤）

| 编号 | 问句 | 预期 | 关键检查点 |
|---|---|---|---|
| MC-B1 | `中际旭创 2026 年半年度营业收入多少` | 营业总收入 41,777,861,795.03（417.8 亿） | 年份+期间双过滤后证据只来自 2026H1 |
| MC-B2 | `中际旭创 2025 年第一季度净利润多少` | 归母净利 1,582,876,128.67（约 15.8 亿） | 证据**只**来自 中际旭创_2025Q1（年份维度回归用例） |
| MC-B3 | `中际旭创 2024 年年报营业收入多少` | FY2024 营收（约 236 页年报也能命中） | 不混入 2025FY/其他期间 |

### C. 拒答纪律（对账闸门）

| 编号 | 问句 | 预期 | 关键检查点 |
|---|---|---|---|
| MC-C1 | `中际旭创 2027 年净利润多少` | 拒答并列出现有十期 | trace `route:period-guard→refuse`；答案含「现有期间：2024FY…2026Q1」 |
| MC-C2 | `今天北京天气怎么样` | 礼貌拒答（领域外） | `refuse`，不硬答 |
| MC-C3 | `中际旭创 2025 年年报净利润变动的主要原因` | 诚实作答或诚实说未找到 | **不得编造原因**；引用真实（已知限制：MD&A 归因段可能漏检，见 §4） |

### D. 摘要（summary）

| 编号 | 问句 | 预期 | 关键检查点 |
|---|---|---|---|
| MC-D1 | `总结一下中际旭创 2025 年年报` | 选中 **2025FY**（非 2024FY），要点非空 | 分季度净利行（15.8→24.1→31.4→36.7 亿）应在场 |
| MC-D2 | `中际旭创2025年中报里管理层是怎么说的` | MD&A（管理层讨论与分析）叙述内容 | 800G/算力需求/1.6T 等业务叙述在场；表头/会计政策行被排除 |
| MC-D3 | `中际旭创15年中报里管理层是怎么说的` | **拒答**（15 年=2015，库里没有） | 答案含「现有期间」列表——口语二位年份解析 |

### E. 双模式兜底

| 编号 | 命令 | 预期 | 关键检查点 |
|---|---|---|---|
| MC-E1 | `python -m duizhang.cli ask "中际旭创 2026 半年报营收" --mode heuristic` | ~1 秒抽取式作答，零 LLM 依赖 | 断 Ollama（`brew services stop ollama`）也应正常 |
| MC-E2 | 停掉 Ollama 后 `ask "..."`（默认 auto） | understand/agent 全部 fallback 到规则，答案仍产出 | trace 出现 `understand:llm→fallback`、`agent:fallback`（正常降级，非故障） |

### F. 语料管理

| 编号 | 命令 | 预期 |
|---|---|---|
| MC-F1 | `python -m duizhang.cli ingest data/中际旭创` | `= 无变化 10 docs / 7873 chunks`（幂等） |
| MC-F2 | 往 `data/中际旭创/` 加新 PDF 后 ingest | `+N 新增`，秒级；多根共存（samples 与中际旭创互不清除） |

---

## 2. trace 判读速查

| 标记 | 含义 | 正常性 |
|---|---|---|
| `mode:llm` / `mode:heuristic` | auto 的解析结果 | 两种皆正常 |
| `understand:llm→fallback` | LLM 理解故障回退规则 | 偶发正常；频繁→查 Ollama |
| `agent:N步/M次search/…/self` | Agent 环自主检索并自行收手 | ✅ 核心行为 |
| `agent:…/重复收手` | 幂等缓存防死循环 | ✅ |
| `agent:fallback` | 工具层故障回退固定管线 | 兜底生效 |
| `trend:N期` | 跨期研究遍历了 N 个报告期 | 应为 6（十期语料） |
| `generate:llm-no-evidence→extractive` | LLM 判无据降级抽取式 | 闸门兜底，答案仍真实 |
| `verify:pass(0.x)` | 覆盖率过闸 | x≥0.45 |
| `verify:degraded→extractive` | 生成对不上账，降级 | 答案换成抽取式，仍可用 |
| `route:period-guard→refuse` | 期间守卫拒答 | 答案应含「现有期间」列表 |

---

## 3. 评估集（samples 库上跑，基准线）

```bash
rm -rf /tmp/dz_eval && mkdir /tmp/dz_eval
DZ_DB=/tmp/dz_eval/t.db python -m duizhang.cli ingest samples/
DZ_DB=/tmp/dz_eval/t.db python -m duizhang.cli eval --mode heuristic          # ①
DZ_DB=/tmp/dz_eval/t.db python -m duizhang.cli eval --mode heuristic --dataset agent   # ②
DZ_DB=/tmp/dz_eval/t.db python -m duizhang.cli eval --mode llm --dataset agent         # ③
DZ_DB=/tmp/dz_eval/t.db python -m duizhang.cli eval --mode llm               # ④
```

**基准线（任一低于即回归，禁提交）**：

| # | 组 | 基准 |
|---|---|---|
| ① | heuristic · default（10 题） | 1.00 / 1.00 / 1.00 |
| ② | heuristic · agent（8 题） | 1.00 / 1.00 / 1.00 |
| ③ | llm · agent | **1.00 / 1.00 / 1.00** |
| ④ | llm · default | 1.00 / 1.00 / 1.00 |
| ⑤ | llm · innolight（14 题真值集，工作库上跑） | 1.00 / 1.00 / 1.00 |

> 注：评估集基于 samples 三家公司；在 中际旭创 工作库上跑会因缺文档判 ✗，
> 属预期非 bug。陷阱题两类（2030 未来期间 / 天气）在任何模式下必须拒答。

---

## 4. 已知限制（测到这些不算 bug）

| 现象 | 原因 | 状态 |
|---|---|---|
| 归因题（变动原因）可能「未找到依据」 | MD&A 归因段未锚定检索 | 待优化（下轮） |
| 2026H1 净利行选中「会计准则差异」段 | 句子挑选器对跨块断词（「的净利润」）敏感 | 轻微 |
| llm 模式 default 集在 0.875–1.0 间波动 | 0.8b 生成方差（思考已关） | 如实记录 |
| trend 归纳段经常缺席 | 数字禁令由代码执行，0.8b 写不出无数字归纳 | 宁缺毋滥（设计决策） |
| 单题首次 ~10s | Ollama 模型加载；闲置 5 分钟自动卸载 | 正常 |

---

## 5. 历史回归对照（每个 bug 一条永久测试）

| 历史 bug（一轮实测事故） | 回归测试 |
|---|---|
| 幽灵旧块（REPLACE 换 rowid 留孤儿 FTS 行） | `test_store.py::test_readd_after_drop_no_ghost` |
| 「半年度」⊃「年度」FY 混入期间过滤 | `test_period_hints_exclusive` |
| 年报块把一季报挤出 top-8 | `test_period_filter_is_doc_level` |
| 问 2025Q1 混入 2024Q1/2026Q1（年份维度缺失） | `test_retrieve_year_scope` |
| 抽取式被自家 verify 误判 degraded（口径） | `test_coverage_counts_cited_chunks_beyond_rank4` |
| MD 文档引用永远过不了正则 | `test_verify_passes_for_md_doc_citations` |
| 页眉句靠公司名压过数字句 | `test_extractive_skips_boilerplate_header_sentences` |
| 长报告摘要选错年份/要点为空 | `test_summarize_keylines_from_retrieval_not_head` |
| Agent 死循环（同调用重复 6 步） | `test_agent_research_loop_bounded_and_fallback` 场景4 |
| 工具参数幻觉期间（半年度传成 FY） | 同上 场景7 |
| 「哪家赚得多」被 0.8b 判成合法枚举 metric | `test_understand_period_override_and_trend_necessity` |
| 「第一季度」被稳定抽成 H1（提示词加长劣化） | 同上 ① |
| 「预计…多少」被判 trend（无趋势词） | 同上 ② |
| 跨页表格/半句上下文丢失（v0.6.1 页级证据） | `test_page_evidence_window` / `test_generate_stores_page_evidence_for_verify` |
| 追溯调整双列重复打印 | `test_trend_skeleton_collapses_adjust_columns` |

> 新增修复必配回归测试（TDD 纪律）：先写失败测试，亲眼看它失败，再修。

---

## 6. 快速回归流程（改代码后的标准动作）

```bash
pytest                                                          # ① 67 用例
（§3 四连评估）                                                  # ② 基准不破
python -m duizhang.cli ask "中际旭创最近几个季度营收的趋势，值得投资吗"   # ③ trend 冒烟
python -m duizhang.cli ask "中际旭创 2027 年净利润多少"            # ④ 守卫冒烟
git push                                                        # ⑤ CI 兜底
```
