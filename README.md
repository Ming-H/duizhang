# 对账 duizhang 🧾

[![CI](https://github.com/Ming-H/duizhang/actions/workflows/ci.yml/badge.svg)](https://github.com/Ming-H/duizhang/actions/workflows/ci.yml)

> **RAG 的每个答案都要能对账。**

带证据链纠错环的 **Agentic RAG** 财报问答引擎：LLM（本地 qwen3.5:0.8b，
零 key 零费用零外呼）
决定「怎么查」——查询理解、路由、检索质量评判、query 改写、**多公司对比分解**、
grounded 生成；确定性代码决定「答案能不能出门」——引用覆盖率闸门不过就降级或
拒答。逐句挂引用、引用必可回溯到财报原文的页与段；对不上的答案宁可拒答，也不编。

```
2,345,678,901.23 元  ──句读──▶  ≈23.46 亿      （数字归一化，句读 judou 的基因）
   任意一个答案    ──对账──▶  [公司_报告期 第N页] （引用覆盖率不达标自动降级）
  LLM 主导检索策略     ──闸门──▶  verify_citations（确定性，模型无权跳过）
```

## 30 秒跑起来

```bash
pip install -r requirements.txt
python -m duizhang.cli ingest samples/     # 摄取样例财报（3 家公司 5 份文件）
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli eval                # 内置评估集，一条命令复现
```

**LLM 大脑默认就在本地**：装好 [Ollama](https://ollama.com)（≥0.40）并
`ollama pull qwen3.5:0.8b`，零 key 零配置直接可用（本地端点免鉴权）：

```bash
ollama pull qwen3.5:0.8b
python -m duizhang.cli ask "对比欧科亿和海伦哲 2026 一季度谁赚得多"   # 多跳对比
python -m duizhang.cli eval --mode llm --dataset agent
```

没装 Ollama 也能跑（`--mode auto`：本地不可用自动走规则模式）。要接远端，
把配置写进 `.env`（已 gitignore，绝不进 git）——任意 OpenAI 兼容端点
（deepseek/智谱 v4/gpt）或 Anthropic 兼容端点均可：

```bash
cat >> .env <<'EOF'
DZ_LLM_API_KEY=你的key
DZ_LLM_BASE_URL=https://api.deepseek.com/v1
DZ_LLM_MODEL=deepseek-chat
EOF
```

全量语料：`python -m duizhang.cli ingest data/corpus`（manifest 指纹增量，
新增/修改/删除自适应，幂等运行秒级返回；**多根摄取互不干扰**——
ingest `samples/` 后再 ingest `data/corpus`，两个根的文档共存，
删除/孤儿清理只对本次扫描的根负责）。

## 这张图在做什么（v0.6：真 Agent 环）

```
START → understand ──→ route ──metric/compare(llm)──→ 【agent_research 工具循环】──→ generate ──→ verify_citations
      （LLM 结构化理解：    │                          模型自主决定查什么/查几次/何时收手      （确定性闸门，不豁免）
       intent/公司/期间，    │                          单一 search 工具；步数≤6；幂等缓存；        ↓ fail
       规则兜底）           │                          理解先验优先于工具参数                抽取式降级/拒答
                           ├─metric/compare(heuristic)→ retrieve/decompose → grade ⇄ rewrite(≤1) ↗
                           ├─summary─→ summarize_doc
                           └─other──→ refuse
```

五个值得看代码的决策点：

| 决策 | 为什么 |
|------|--------|
| **understand 查询理解** | LLM 一次抽取 intent/公司/期间，直接驱动路由与 `doc_scope` 检索限定——v0.4 的关键词子串猜测是三轮 bug 的根源，理解交给模型、规则只做兜底。 |
| **compare 多跳分解** | 「对比 A 和 B」按公司拆子查询、各自公司+期间 doc_scope 检索、交错合并——两家证据都在场，且**多跳不豁免对账闸门**。 |
| **真 Agent 环（v0.6）** | llm 模式下检索交给模型：单一 search 工具 + 循环，模型自主决定查什么、何时收手；步数硬上限 + 幂等缓存防小模型死循环，理解先验优先于工具参数防幻觉参数——自主检索照样过对账闸门。 |
| **Corrective-RAG 纠错环** | `grade_docs → rewrite_query → retrieve` 的**有界**重试（≤1，条件边实现）。检索不足先改写重查一次，还不行才拒答——无界重试是状态机的反模式。 |
| **期间守卫 + 文档级期间过滤** | 问「2030 年净利润」直接拒（年份来自元数据不靠模型）；问「半年度」时限定该公司对应期间的文档集合重查——年报大文档块会淹没目标期间（v0.4 实测教训）。 |
| **自由规划、强制对账** | LLM 判无据 → 降级抽取式而非拒答；覆盖率闸门只考核数字句（编造风险所在）、token 过元词过滤；闸门是确定性代码，模型无权跳过。 |

## 设计原则

- **LLM 是大脑，本地是默认**（v0.5.1）：默认本地 Ollama qwen3.5:0.8b
  （零 key 零费用零外呼），`--mode auto` 本地不可用自动退规则——同一张
  图，两种大脑；远端任意 OpenAI/Anthropic 兼容端点 `.env` 三行可换。
  模型每代都在变，编排假设要能活过任何一版实现。
- **自由规划、强制对账**：LLM 决定怎么查；答案出口的覆盖率闸门是
  确定性代码，模型无权跳过。
- **元数据白捡**：巨潮财报命名 `公司_报告期类型.pdf` 自带公司与期间，解析文件名而非让模型猜。
- **表格感知分块**：财报的核心事实住在表格里，纯字符切分等于销毁证据。数字密集短行聚成原子块，永不跨块。
- **数字归一化**：`41,914.72 万元` 注释成 `≈4.19亿` 附在块尾，让词法检索对「多少亿」类问法直接命中（纯程序，零幻觉）。
- **检索：SQLite FTS5 双轨 + RRF**（v0.4）：词轨（jieba 预分词 + unicode61）
  与字轨（trigram）bm25 打分、RRF 融合；磁盘倒排使容量与内存解耦——
  60.7 万块实测：稳态 39ms / 内存 0.03GB / 启动 0.05s / 增量摄取 0.8s。
  容量推导与备选取舍见 `docs/DESIGN-scale.md`、讲解见 `docs/EXPLAINER.md`。
- **评估内建**：改任何代码，`python -m duizhang.cli eval` 验证三指标不退化。

## 评估（全量语料 1,104 docs / 60.7 万块）

| 数据集 | 模式 | retrieval@3 | answer hit | refusal |
|---|---|---|---|---|
| default（10 题，CI 回归） | heuristic | 1.00 | 1.00 | 1.00 |
| default | llm（qwen3.5:0.8b 本地，Agent 环） | 0.875 | 0.875 | 1.00 |
| agent（3 对比 + 3 单跳/摘要 + 2 陷阱） | heuristic | 1.00 | 1.00 | 1.00 |
| agent | **llm（qwen3.5:0.8b 本地，Agent 环）** | **1.00** | **1.00** | **1.00** |
| agent | llm（glm-5.3 远端，对照） | 1.00 | 1.00 | 1.00 |

**架构比模型更保命**：0.8b 本地小模型的生成质量不足时，verify 闸门
自动降级为抽取式作答——agent 集满分的含金量在「理解+检索+闸门」，
不在生成模型。glm-5.3（远端对照）走 GLM Coding Plan 的 Anthropic
兼容端点；heuristic 模式可复现无方差。陷阱题两类：未来期间（2030）与领域外（天气）——都必须拒答。
多跳对比题示例：`对比欧科亿和海伦哲 2026 一季度谁赚得多` → 分解检索两家
Q1 净利润证据 → 对比作答。历史修复记录见 `docs/DESIGN-v0.4.md` §5、
`docs/DESIGN-v0.5.md`。

单元测试（52 用例：分块/理解/守卫/纠错环/分解/Store 幽灵块/多根摄取/
对账口径/思考分层/.env，conftest 封网永不外呼）：

```bash
pip install -r requirements-dev.txt
pytest
```

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
├── fts.py          # FTS5 双轨检索（unicode61 词轨 + trigram 字轨）RRF 融合 + doc_scope
├── graph.py        # LangGraph 状态机（understand/route/retrieve/decompose/grade/rewrite/generate/verify）
├── answer.py       # 抽取式生成 + 引用覆盖率校验（对账闸门）
├── llm.py          # LLM 层：智谱 GLM 默认，.env 装载，思考分层，任意兼容端点可换
├── evalx.py        # 评估 runner（default/agent 双数据集）
└── cli.py          # ingest / index / ask / eval / docs
tests/              # 52 用例（conftest 封网：永不外呼 LLM）
evals/              # dataset.json（10 题）+ dataset-agent.json（8 题含多跳对比）
samples/            # 3 家公司 5 份真实财报（巨潮）
docs/               # 设计文档 ×4 + EXPLAINER 讲解 + 流程动效图
.env                # 密钥（gitignored，永不进 git）
```

## 密钥安全

- 默认本地 Ollama **无需任何密钥**；接远端时 key 只放 `.env`
  （已 gitignore），`_load_dotenv` 装载且环境变量优先
- 密钥不进代码、不进文档、不进示例命令、不进 git 历史
- 测试套件 conftest 全局封网（与机器上是否装 Ollama 无关），CI 无密钥全绿

## Roadmap

- [ ] 向量检索可选后端（bge-small-zh，HF mirror），与词法双路 RRF
- [x] ~~多公司对比（v0.5 compare 分解）~~
- [ ] 跨期对比（「2025 vs 2026 毛利率变化」→ 按期间分解）
- [ ] rerank 层（LLM 或 cross-encoder 可切换）
- [ ] Web UI（对账视图：答案句 ↔ 原文高亮）

## License

MIT
