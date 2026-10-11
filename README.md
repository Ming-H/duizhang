# 对账 duizhang 🧾

> **RAG 的每个答案都要能对账。**

带证据链纠错环的财报问答引擎：LangGraph 编排 Corrective-RAG，逐句挂引用、引用必可回溯到财报原文的页与段；对不上的答案宁可拒答，也不编。

```
2,345,678,901.23 元  ──句读──▶  ≈23.46 亿      （数字归一化，句读 judou 的基因）
   任意一个答案    ──对账──▶  [公司_报告期 第N页] （引用覆盖率不达标自动降级）
```

## 30 秒跑起来

```bash
pip install -r requirements.txt
python -m duizhang.cli ingest samples/     # 摄取样例财报（3 家公司 5 份文件）
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli eval                # 内置评估集，一条命令复现
```

全量语料：`python -m duizhang.cli ingest data/corpus`（manifest 指纹增量，
新增/修改/删除自适应，幂等运行秒级返回）。

零 API key、零模型下载。接 LLM 只需三个环境变量（图结构不变）：

```bash
export DZ_LLM_API_KEY=sk-...
export DZ_LLM_BASE_URL=https://api.deepseek.com/v1   # 任意 OpenAI 兼容端点
export DZ_LLM_MODEL=deepseek-chat
python -m duizhang.cli ask "海伦哲营收变动原因" --mode llm
```

## 这张图在做什么

```
START → route ──metric──→ retrieve → grade_docs ──insufficient(retries<1)──→ rewrite_query ─┐
         │       │                        │                                   （改写后重检索）
         │       └─summary─→ summarize_doc │                                          ↑
         │                            ↑    ↓ sufficient                               │
         └──other──→ refuse           └─ generate ──→ verify_citations ──fail──→ 抽取式降级
                                    （heuristic 抽取式 / llm grounded 生成）
```

四个值得看代码的决策点：

| 决策 | 为什么 |
|------|--------|
| **Corrective-RAG 纠错环** | `grade_docs → rewrite_query → retrieve` 的**有界**重试（≤1，条件边实现）。检索不足先改写重查一次，还不行才拒答——无界重试是状态机的反模式。 |
| **期间守卫** | 问「2030 年净利润」直接拒：库里没有该期间的报告，对不上账就不硬答。年份来自文档元数据，不靠模型判断。 |
| **期间感知重排** | 问「半年度」却拿一季报的数字回答，是财报 RAG 最常见的静默错误。检索后按 query 期间对命中块重排。 |
| **verify 降级而非失败** | LLM 生成的答案引用覆盖率低于阈值 → 自动降级为抽取式作答（永远有真凭实据），而不是把错误答案交给用户。 |

## 设计原则

- **heuristic 是一等公民**：默认模式零 key 零模型下载，路由/评分/改写/生成全部规则实现，演示永不翻车。`--mode llm` 把四个决策点换成模型调用，**图不变**——模型每代都在变，编排假设要能活过任何一版实现。
- **元数据白捡**：巨潮财报命名 `公司_报告期类型.pdf` 自带公司与期间，解析文件名而非让模型猜。
- **表格感知分块**：财报的核心事实住在表格里，纯字符切分等于销毁证据。数字密集短行聚成原子块，永不跨块。
- **数字归一化**：`41,914.72 万元` 注释成 `≈4.19亿` 附在块尾，让词法检索对「多少亿」类问法直接命中（纯程序，零幻觉）。
- **检索：SQLite FTS5 双轨 + RRF**（v0.4）：词轨（jieba 预分词 + unicode61）
  与字轨（trigram）bm25 打分、RRF 融合；磁盘倒排使容量与内存解耦——
  60.7 万块实测：稳态 39ms / 内存 0.03GB / 启动 0.05s / 增量摄取 0.8s。
  容量推导与备选取舍见 `docs/DESIGN-scale.md`、讲解见 `docs/EXPLAINER.md`。
- **评估内建**：改任何代码，`python -m duizhang.cli eval` 验证三指标不退化。

## 评估（heuristic 模式，内置 10 题集）

| 指标 | 结果 |
|---|---|
| retrieval@3 | 0.875 |
| answer hit | 1.00 |
| refusal（应拒答拒答率） | 1.00 |

含两类陷阱题：未来期间（2030）与领域外（天气）——都必须拒答。

## 用自己的财报

```bash
# 任意目录：PDF 按巨潮命名「公司_2025年年度报告.pdf」；也吃 md
python -m duizhang.cli ingest /path/to/财报目录
```

## 目录

```
duizhang/
├── schemas.py      # GraphState（窄接口状态）+ Doc/Chunk/Citation
├── source.py       # 语料扫描 + manifest 三态 diff（增量账本）
├── ingest.py       # 文件名元数据 / 表格感知分块 / 页眉剥离 / 数字归一化 / 查询扩展
├── store.py        # 四表唯一写入口（事务 + fts rowid 对齐，幽灵块防线）
├── index.py        # 分词原语 + 瘦身 Corpus（docs 全量、chunk 按需）
├── fts.py          # FTS5 双轨检索（unicode61 词轨 + trigram 字轨）RRF 融合
├── graph.py        # LangGraph 状态机（route/retrieve/grade/rewrite/generate/verify）
├── answer.py       # 抽取式生成 + 引用覆盖率校验（对账闸门）
├── llm.py          # 可选 LLM 层（OpenAI 兼容；heuristic 模式不加载）
├── evalx.py        # 评估 runner
└── cli.py          # ingest / index / ask / eval / docs
tests/              # 19 用例：分块/路由/守卫/纠错环/Store 幽灵块回归/增量三态
samples/            # 3 家公司 5 份真实财报（巨潮）
docs/               # 设计文档 ×3 + EXPLAINER 讲解 + 流程动效图
```

## Roadmap

- [ ] 向量检索可选后端（bge-small-zh，HF mirror），与词法双路 RRF
- [ ] 跨期对比查询（「2025 vs 2026 毛利率变化」→ 结构化指标抽取）
- [ ] rerank 层（LLM 或 cross-encoder 可切换）
- [ ] Web UI（对账视图：答案句 ↔ 原文高亮）

## License

MIT
