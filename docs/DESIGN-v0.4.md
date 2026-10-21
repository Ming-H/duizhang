# 对账 v0.4 设计：FTS5 检索层与写入层收敛

| | |
|---|---|
| 状态 | **已实施并验收**（2026-10-11，验收记录见 commit 5874147） |
| 范围 | 检索层替换（内存倒排 → SQLite FTS5）+ 写入层收敛（消灭双路径） |
| 前置文档 | `DESIGN-scale.md`（容量推导）、`DESIGN-incremental.md`（增量设计） |
| 硬约束 | 本机 M1/8GB/47GB；语料维持现状 ~1,103 份；**最小改动集**原则 |

## 1. 现状盘点（2026-10-11 全量代码通读）

**能力清单**（全部有测试或线上验证）：
- 摄取：文件名三通道元数据 / 表格感知分块 / 页眉剥离 / 数字归一化 /
  manifest 指纹增量 / 孤儿清理 / 幂等跳过
- 检索：双路 BM25（自研内存倒排 + numpy）RRF 融合，60.7 万块 9ms
- 问答图：LangGraph 八节点（route 三分类+期间守卫 / 期间硬过滤 /
  有界纠错环 / 双模式生成 / 引用覆盖率对账 / 域词守卫摘要 / 拒答）
- 评估：10 题内置集三指标；pytest 14 用例

**本次通读发现的缺陷**（v0.4 一并处理）：

| # | 缺陷 | 位置 | 性质 |
|---|---|---|---|
| D-1 | `read_pdf(path)` 被调用两次（每份 PDF 解析两遍） | ingest.py:206 | 冗余 IO |
| D-2 | watch 模式首轮 `one_pass()` 执行两遍 | cli.py:141-146 | 逻辑错误 |
| D-3 | `--rebuild` 时数据写两遍（增量 INSERT 后 `save()` 再全量 executemany） | cli.py:136-137 | 冗余写 |
| D-4 | 写入路径双轨：cli 直写 SQL 与 `Corpus.save` 并存 | cli.py / index.py | 幽灵块 bug 根源 |
| D-5 | `__import__("json")` 内联导入、两处函数内局部 import | cli.py:108 / graph.py:71,195 | 代码卫生 |
| D-6 | 全内存驻留：`Corpus.load` 读全部 60.7 万 chunk（1.8GB），检索层容量与内存耦合 | index.py | 架构约束（见 DESIGN-scale） |

## 2. 目标 / 非目标

**目标**
1. 检索层换 SQLite FTS5 磁盘倒排：容量与内存解耦（本机可承载全 A 股 5 年
   文本层的架构路径，见 DESIGN-scale 推导），内存常驻 ~2GB 内
2. 写入层收敛为单一模块（Store）：docs/chunks/fts/manifest 四表唯一写入口，
   消灭 D-3/D-4；增量摄取从「加载全库+重建索引（3 分钟）」变为纯 SQL 增量（秒级）
3. 修复 D-1/D-2/D-5
4. A/B 验证后**同批删除**旧路径（内存 BM25 + pickle 缓存层），主线单一实现

**非目标**（明确不做，及理由）
- schema 迁移（chunk_id TEXT 主键改整型）：收益 27MB/210MB，需全量重灌，
  风险收益比不成立。FTS5 表直接对齐 `chunks.rowid`（隐式主键），零迁移
- 分年分库：服务 3,000 万块规模的决策，当前 60 万块单库远未触顶
- 摄取并行化：全量重建 20 分钟可接受，日常路径是增量（秒级）
- ES / 向量检索：见 DESIGN-scale §6 升级触发条件，均未触发

## 3. 决策记录

### D1 检索层：FTS5 双表，RRF 语义平移

```
fts_w(content_w)         tokenize=unicode61   ← 摄取时 jieba 预分词、空格分隔
fts_c(content_c)         tokenize=trigram     ← 原文前 200 字（与旧字级轨道 cap 一致）
两表 rowid := chunks.rowid（对齐 → 删除走 rowid 快路径）
```

- 查询：`SELECT c.* FROM fts_w JOIN chunks ON chunks.rowid=fts_w.rowid
  WHERE fts_w MATCH ? ORDER BY bm25(fts_w) LIMIT k*3`，两路各取 top-k*3，
  Python 侧 RRF(k=60) 融合——**融合语义与 v0.3 完全一致，仅打分引擎更换**
- MATCH 表达式：jieba 切词后 `"tok1" OR "tok2"`（词组引号包裹，防语法注入）
- 公司/期间过滤：`JOIN docs` + WHERE 下推（替代 numpy 布尔掩码）
- **备选与否决**：ES（单机 JVM 负资产）/ 向量（增益未证明）/ 自研磁盘倒排
  （重复造 FTS5）——完整对比见 DESIGN-scale §3

### D2 Corpus 瘦身：数据按需取

`Corpus` 不再持有 `chunks` 全量列表。构成：
- `docs` 元数据全量（千级，期间守卫/过滤必需）
- 检索时按 rowid 批量取 Chunk（单次 JOIN 已含文本）
- `summarize_doc` 的 `[c for c in corpus.chunks ...]` 改为按 doc_id 的 SQL 查询

效果：启动 <5s（原全量加载+建索引 117s 或 pickle 热启 5s）、内存 ~2GB。

### D3 写入层收敛：Store 模块

```
tally/store.py（新）
  ensure_schema()          四表 DDL（含两个 fts 虚拟表）
  doc_ids() -> set
  add_doc(doc, chunks)     四表事务写入（fts 增量插入，段合并自动）
  drop_doc(doc_id)         四表删除（fts 按 chunks.rowid 对齐删）
  manifest bootstrap/diff 所需读写
```
`cli.ingest` 只做「扫描→diff→调 Store」；`Corpus.save` 与 pickle 层删除。
幂等跳过保留（零变更零 SQL 写）。

### D4 迁移与回滚

1. 新增 `fts.py`（检索）+ `store.py`（写入）；`index.py` 保留 `word_tokens/
   char_bigrams/expand 依赖`，`Corpus` 重写为瘦身版——**同一次提交内完成替换**
2. 一次性迁移：`cli` 新增 `index` 子命令，为现有 60.7 万 chunks 建 fts（~2 分钟）
3. A/B：~~`TALLY_BACKEND=memory|fts` 环境变量临时并存~~ **实施偏差**：
   实际未引入双后端开关——以 v0.3 已记录的基线指标（样例集/全库 evalx）
   作为对比基准，比并存两套实现更符合反冗余原则，且删除时点提前为零
4. 回滚：git revert 单提交即回 v0.3（迁移只增表不删列，v0.3 代码可直接读旧表）

### D5 验收标准

| 项 | 标准 |
|---|---|
| 检索质量 | evalx 样例集三指标不低于 v0.3 基线（retrieval@3 ≥0.875 / answer_hit ≥1.0 / refusal =1.0） |
| 检索延迟 | 全库 60.7 万块 ≤100ms（v0.3 实测 9ms，FTS5 允许放宽但需实测记录） |
| 内存常驻 | ≤2.5GB（活动监视器实测） |
| 启动时间 | ≤5s |
| 增量摄取 | 单份新 PDF → 可检索 ≤5s（原 ~3 分钟） |
| 冗余清除 | SparseBM25 / save_index / IDX 常量 / D-1~D-5 全部消失于代码库 |
| 测试 | pytest 全绿，新增 Store 与 FTS 用例 ≥6 个 |

**A/B 双后端删除时点**：因实施偏差（见 §3 步骤 3）从未引入开关，
主线自始单一实现。

## 4. 风险

| 风险 | 缓解 |
|---|---|
| trigram 索引膨胀（磁盘） | content_c 截 200 字；实测入库后检查 fts 磁盘增量，超预算则降级纯词法（中文 jieba 已覆盖主路径，DESIGN-scale §6） |
| FTS5 bm25 排序与自研版有差异 | evalx 三指标做裁判（验收项 1），不做排序逐位对齐 |
| sqlite 单写者并发 | 单进程 CLI 设计内无并发写；文档声明 |

## 5. 第四轮审查修复（2026-10-14）

全量代码通读 + 752MB 生产库实测发现五处逻辑缺陷与一处性能缺陷，全部按
TDD（先失败测试后修复）处理；修复前后 eval 对比与回归测试见 tests/。

| # | 缺陷 | 根因 | 修复 | 实测证据 |
|---|---|---|---|---|
| R-1 | 期间硬过滤被子串污染：「半年度」包含「年度」，FY 混进 hints，年报块淹没目标期间 | `period_hints` 顺序 if 无互斥 | 互斥匹配（从最具体期间开始）+ 补「一/二/三/四季度」口语映射（二季度→H1、四季度→FY） | 问 2026 半年度命中分布 2025FY:5/2026H1E:2 → 全部 H1E |
| R-2 | 期间过滤是命中级后过滤：大文档把目标期间块挤出 top-8，exact 落空回退未过滤结果 = 拿年报答一季报 | top-8 窗口内做块级过滤 | 升级为文档级 `doc_scope`（fts.search 新参数），该公司有对应期间文档即限定集合重查 | 欧科亿 Q1 同比题 top1 2025FY → 2026Q1 |
| R-3 | 自家抽取式答案被自家 verify 误判 degraded：extractive 从 hits[:8] 抽句、coverage 只认 hits[:4] | 证据池与抽取范围口径不一致 | 证据池 = 答案中实际引用的块（citation 串匹配），无引用回退 hits[:4]（LLM 路径不变） | trace `verify:degraded(0.37)→extractive` → `verify:pass(0.941)` |
| R-4 | MD/TXT 块（page=0）引用 `[doc_id]` 永远过不了 `has_citation` 正则 | 正则要求「第N页」或 `_\d{4}` 后缀 | hits 兜底：citation 串直接比对 | `[永和股份_MD]` False → True |
| R-5 | 页眉句（公司名+报告标题）靠 query 公司名得分，事实数字句落选；数字注释固定取 top1 块 | 元词表不含公司名；jieba 把公司名切碎（海伦哲→海伦+哲） | 元词过滤补公司名（整串子串 + 字符集覆盖碎片）；注释跟随被选中句所在块 | 海伦哲题答案 4 句页眉 → 数字句 |
| R-6 | `chunks(doc_id)` 无索引：drop_doc/chunks_of 全表扫 60.7 万行 | SCHEMA 缺索引 | `CREATE INDEX IF NOT EXISTS idx_chunks_doc`（存量库下次 Store 打开自动补建）+ WAL + ANALYZE | 31.8ms → <0.1ms；建索引 0.16s/~15MB |

工程化同步：多根摄取（ingest B 不再清掉 A 的文档，删除/孤儿判定限本根）、
LLM 降级 trace 留痕（`route:llm→fallback` 等）、evalx 口径修正（retrieval@3
docstring、逐题 ✓ 含 doc_hit、图编译一次）、裸 `pytest` 可跑（pyproject
pythonpath，删除 sys.path hack）、ruff + GitHub Actions CI（lint + 测试 +
README 30 秒流程冒烟）。

补录实测：content_c 截 200 字影响 14.2%（86,104/607,502 块的去空白后长度
超 200），词轨（全文）兜底——维持 §4 的截断决策，数字记录在案。

**修复前后（heuristic，全量语料 1,104 docs / 607,502 chunks）**：
retrieval@3 0.875 → **1.00**，answer hit 0.625 → **1.00**，refusal 1.00。
评估集第 6 题期望词补 `644,051,473`（corpus 版财报无 113.49 字样，表格
两年数可推出；samples 版两词皆可命中）。
