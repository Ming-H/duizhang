# 对账 duizhang 🧾

[![CI](https://github.com/Ming-H/duizhang/actions/workflows/ci.yml/badge.svg)](https://github.com/Ming-H/duizhang/actions/workflows/ci.yml)

> **RAG 的每个答案都要能对账。**

**封闭世界约束下的事实 Agent**（v0.7：四支柱齐备）：LLM（本地 Ollama，
零 key 零费用零外呼；模型档位自选 `DZ_LLM_MODEL`——小档适配多数
机器，大档叙述更佳）决定「怎么查、怎么讲」——查询理解、多跳规划、
自主迭代检索、多轮记忆继承、grounded 生成；确定性代码决定「答案
能不能出门」——引用覆盖率闸门不过就降级或拒答。逐句挂引用、引用
必可回溯到财报原文的页与段；对不上的答案宁可拒答，也不编。

**封闭世界（closed-world）契约**：财报是专属内部资料——答案里的
每个事实只能来自内部 RAG 检索的证据，LLM 参数知识只做语言组织、
不产出事实；工具白名单只含内部动作（search / metrics_table /
derive），永无浏览器、外部 API、模型生成代码；写入长期记忆的只有
verify:pass 的结构化事实。封闭世界让「对账」成为可判定的问题——
开放世界里 verify 永远证伪不了「模型碰巧知道」。

```
2,345,678,901.23 元  ──句读──▶  ≈23.46 亿      （数字归一化，句读 judou 的基因）
   任意一个答案    ──对账──▶  [公司_报告期 第N页] （引用覆盖率不达标自动降级）
  LLM 主导检索策略     ──闸门──▶  verify_citations（确定性，模型无权跳过）
  事实唯一来源 = 内部 RAG 证据；参数知识/外部工具/生成代码 一律不出事实
```

## 30 秒跑起来

```bash
pip install -r requirements.txt
python -m duizhang.cli ingest samples/     # 摄取样例财报（3 家公司 5 份文件）
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli eval                # 内置评估集，一条命令复现
```

**LLM 大脑默认就在本地**：装好 [Ollama](https://ollama.com)（≥0.40）并
`ollama pull` 一个模型，零 key 零配置直接可用（本地端点免鉴权）。档位
自选——下面以 `qwen3.5:0.8b` 为例（多数机器跑得动；机器资源充足时
`DZ_LLM_MODEL` 换更大档，叙述质量更好，闸门兜底不变）：

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

## 这张图在做什么（v0.7：封闭世界事实 Agent）

```
START → understand ──→ memory_probe ─→ route ──多跳信号──→ 【plan 规划】─→ 【agent_research ReAct 工具循环】─→ generate ──→ verify_citations
      （LLM 结构化理解：   （L1 台账探针：    │            ≥3公司/跨年/≥2指标族      子目标白名单校验(只删不造)      （确定性闸门，不豁免）
       intent/公司/期间，   命中仍要过 verify） │                                    search/metrics_table/derive       ↓ fail
       规则兜底）                            ├─metric/compare(heuristic)→ retrieve/decompose → grade ⇄ rewrite(≤1) ↗  抽取式降级/拒答
                                            ├─trend─→ trend_research（确定性期间遍历 + Q2=H1−Q1 / Q4=FY−Q3 差分派生）
                                            └─other─→ refuse
```

**Agent 四支柱（v0.7 补齐，DESIGN-v0.7 §1 盘点 → §4 落地）**：

| 支柱 | 落地 | 边界（不破坏对账底线） |
|------|------|------|
| **Planning 规划** | `plan` 节点（llm 模式）：≥3 公司 / 跨年 / ≥2 指标族的多跳问题先出结构化子目标清单，逐个执行；子目标检查点（数字在场 ∧ 指标词命中）逐个验证 | 校验器**只删不造**：白名单外动作丢弃、规约去重、上限封顶；缺口=部分答案不硬凑 |
| **Memory 记忆** | 三层：L0 会话（近 3 轮全量+更早压缩为结构化摘要，公司/期间/年份/指标词继承，**永不继承自由文本**）/ L1 事实台账（verified_facts 表，键=公司+期间+指标+口径，合并算子 noop/insert/conflict/restated/reingest）/ L2 固定语义资产（alias 词典） | 写闸=verify:pass ∧ 口径闸未触发；**读不豁免**——台账命中拼出的答案照样过 verify；语料指纹变更即失效 |
| **Tool Use 工具** | 注册表三工具：`search`（检索）/ `metrics_table`（取某期主要会计数据行）/ `derive`（单季差分 Q2=H1−Q1、Q4=FY−Q3，原币精度） | 白名单只含内部动作；**禁止模型心算**——差分是确定性算子，出口重算操作数必须真在场 |
| **ReAct 回路** | 步级四元组进观测：`thought/action/obs/gate`（含耗时），`-v` 逐行、`--json` 结构化可重放 | 步数≤6 + 幂等缓存 + 整批重复即收手；理解先验覆盖工具参数防幻觉参数 |

**长程可靠性（为什么拉长链之前先装闸门）**：单步 95% 可靠的任务 20 步只剩
~36% 成功——该改的不是单步精度，是传播结构：

```
P(错误答案) ≤ Σₖ eₖ·gₖ    eₖ=第k步出错率（模型决定）  gₖ=错误穿过该步闸门的存活率（代码决定）
e=5%、g=2%、20 步 → ≤2%（朴素模型 36% 失败），且失败形态=降级/拒答（安全失败）
```

四个杠杆：能确定化的环节不给模型（trend 确定性遍历）；verify-as-you-go
（子目标检查点把拦截推进到步级）；结构化传递（字段复制不文本重释——
GraphState/subgoals/台账都是传播率设计）；并行分支不乘法（compare 按公司
分解、各自过闸）。`eval --json` 的 steps 出口给出 waste_rate（dup+empty
占比）作 eₖ 的可测代理。相关性警告：闸门与模型犯同一类错时上界失效——
heuristic/llm 双脑回归就是相关性探测，不是冗余。

值得看代码的决策点：

| 决策 | 为什么 |
|------|--------|
| **understand 查询理解** | LLM 一次抽取 intent/公司/期间，直接驱动路由与 `doc_scope` 检索限定——v0.4 的关键词子串猜测是三轮 bug 的根源，理解交给模型、规则只做兜底。 |
| **compare 多跳分解** | 「对比 A 和 B」按公司拆子查询、各自公司+期间 doc_scope 检索、交错合并——两家证据都在场，且**多跳不豁免对账闸门**。 |
| **真 Agent 环（v0.6→v0.7）** | llm 模式下检索交给模型：三工具白名单 + ReAct 循环，模型自主决定查什么、何时收手；步数硬上限 + 幂等缓存防小模型死循环，理解先验优先于工具参数防幻觉参数——自主检索照样过对账闸门。 |
| **Corrective-RAG 纠错环** | `grade_docs → rewrite_query → retrieve` 的**有界**重试（≤1，条件边实现）。检索不足先改写重查一次，还不行才拒答——无界重试是状态机的反模式。 |
| **trend 双闸 + 裁判自测（v0.6.5）** | 趋势叙述层 LLM 默认在环（数字全部来自确定性骨架），三层兜底：数字闸（叙述数字必须 ∈ 骨架）→ 方向闸（方向词必须与同比符号相容）→ 结构化裁判（faithful/on_topic/sensitive，**非对称信任**：只能否决不能解锁；敏感句只剥不重写）。任一违例退骨架直出=保守失败。 |
| **derive 差分派生（v0.7）** | 「Q2/Q4 净利」原文不披露单季——用累计数差分（Q2=H1−Q1、Q4=FY−Q3，操作数同为年初至期末累计、口径对齐），原币精度算术，出口重算 |a−b−value|≤0.01 且操作数必须在证据池。**禁止模型心算**（封闭世界：算术也是事实）。 |
| **期间守卫 + 文档级期间过滤** | 问「2030 年净利润」直接拒（年份来自元数据不靠模型）；问「半年度」时限定该公司对应期间的文档集合重查——年报大文档块会淹没目标期间（v0.4 实测教训）。 |
| **自由规划、强制对账** | LLM 判无据 → 降级抽取式而非拒答；覆盖率闸门只考核数字句（编造风险所在）、token 过元词过滤；闸门是确定性代码，模型无权跳过。 |

## 设计原则

- **LLM 是大脑，本地是默认**（v0.5.1）：默认本地 Ollama
  （零 key 零费用零外呼；默认小档见 `llm.DEFAULT_MODEL`，`DZ_LLM_MODEL`
  按机器条件自选），`--mode auto` 本地不可用自动退规则——同一张
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
| default | llm（本地默认档，Agent 环） | 0.875 | 0.875 | 1.00 |
| agent（3 对比 + 3 单跳/摘要 + 2 陷阱） | heuristic | 1.00 | 1.00 | 1.00 |
| agent | **llm（本地默认档，Agent 环）** | **1.00** | **1.00** | **1.00** |
| agent | llm（glm-5.3 远端，对照） | 1.00 | 1.00 | 1.00 |
| innolight（31 题真值：事实/趋势/陷阱/鲁棒） | heuristic | 1.00 | 1.00 | 1.00 |
| innolight | **llm（本地默认档，Agent 环）** | 1.00 | 0.963（每股收益，§9 样本 C） | 1.00 |
| **maotai（9 题真值：简称/口径/断词泛化）** | **heuristic** | **1.00** | **1.00** | **1.00** |
| **maotai** | **llm（本地默认档，Agent 环）** | **1.00** | **1.00** | **1.00** |
| **quarters（8 题：单季派生/直披露/期间守卫）** | **heuristic** | **1.00** | **1.00** | 1.00 |
| **quarters** | **llm（本地 9b，Planner/derive）** | **1.00** | **1.00** | 1.00 |
| **multihop（8 题：跨年双值/三公司/多指标族）** | **heuristic** | **1.00** | **1.00** | 1.00 |
| multihop | llm（本地 9b，Planner） | 0.857 | **1.00** | 1.00 |
| **multiturn（10 轮接力：继承/口径切换/代词指代）** | **heuristic** | **1.00** | **1.00** | — |
| **multiturn** | **llm（本地 9b，L0 记忆）** | **1.00** | **1.00** | — |

> llm 行为**默认档**（`DZ_LLM_MODEL` 未设时）的实测值——换更大档
> 分数会上移（innolight 实测满分），heuristic 行与档位无关、可复现。

**语料分工**：`samples/`（3 家公司 5 份财报，演示与 CI 回归）；
`data/中际旭创/`（10 docs / 7,873 chunks，**随 git 入库 26MB**——
真值评估集的承重墙：每题答案都是人工核对过的原文数字，脱离这份
语料 innolight 集不可复现；CI 冒烟常驻承重）；
`data/贵州茅台/`（10 docs / 6,518 chunks，10MB 随 git 入库——
2024–2026 与中际旭创同期结构，第二家公司语料就位、真值题待补）；
`data/corpus/`（全量 1,104 docs，本机自备不入 git）。多语料可同库
共存（多根摄取），上表 innolight 行即 samples+中际旭创 同库跑出。

**架构比模型更保命**：本地小模型（默认档）的生成质量不足时，verify 闸门
自动降级为抽取式作答——agent 集满分的含金量在「理解+检索+闸门」，
不在生成模型。glm-5.3（远端对照）走 GLM Coding Plan 的 Anthropic
兼容端点；heuristic 模式可复现无方差。**多公司泛化（v0.6.4）**：简称
归一（茅台→贵州茅台）/行级抽取/口径数值闸（D20–D22，DESIGN-v0.6
§10）——第二家公司（贵州茅台 10 份）即插即测，9/9 双脑全绿，「任意
丢一家 A 股公司的财报进来就能用」。陷阱题两类：未来期间（2030）与领域外（天气）——都必须拒答。
评估新增 faithfulness（pass 率/平均覆盖率 1.00/0.94）与 chunk 级 P/R@8
（0.148/0.926，P 的差距=前排噪声靶子）；v0.7 再加 **steps 步级观测**
（calls/new/dup/empty + waste_rate=eₖ 代理）与三个新集：**quarters**
（单季差分派生——Q4=FY−Q3累、Q2=H1−Q1，真值与年报分季度表交叉验证
到分位）、**multihop**（跨年双值/三公司/多指标族——Planner 的靶子集）、
**multiturn**（10 轮接力链——L0 继承/口径切换/代词指代，eval 默认隔离
L1 台账防「把检索能力测成记忆命中」，`DZ_EVAL_MEMORY=on` 可放开）。
多跳对比题示例：`对比欧科亿和海伦哲 2026 一季度谁赚得多` → 分解检索两家
Q1 净利润证据 → 对比作答。历史修复记录见 `docs/DESIGN-v0.4.md` §5、
`docs/DESIGN-v0.5.md`、`docs/DESIGN-v0.6.md` §7（innolight 真值驱动
四组修复：裸年份→FY/引用格式解耦/口径行抽取/前置封顶）。

单元测试（130 用例：分块/理解/守卫/纠错环/分解/Store 幽灵块/多根摄取/
对账口径/引用漂移/表块拼接/口径守卫/前置封顶/裸年份/拒收闸/
思考分层/.env/向量可选路（缺依赖退双轨）/评测指标/简称归一/
行级抽取/口径数值闸/配对降权/分层证据/v0.7 四支柱（步级观测/
Planner 校验/记忆合并算子与写闸读闸/derive 差分与年份隔离），
conftest 封网永不外呼）：

```bash
pip install -r requirements-dev.txt
pytest
```

## 用自己的财报

```bash
# 任意目录：PDF 按巨潮命名「公司_2025年年度报告.pdf」；md/txt 主干
# 必须是公司名（公司_分析.md）——解析不出公司名的文件 ⚠ 拒收，不静默入库
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
├── graph.py        # LangGraph 状态机（understand/memory_probe/route/plan/agent_research/trend/…/verify）
├── memory.py       # Memory 三层（L0 会话压缩继承 / L1 台账读写 / L2 语义资产）
├── answer.py       # 抽取式生成 + 引用覆盖率校验（对账闸门）
├── llm.py          # LLM 层：智谱 GLM 默认，.env 装载，思考分层，任意兼容端点可换
├── evalx.py        # 评估 runner（judge 纯函数；default/agent/innolight 三数据集）
└── cli.py          # ingest / index / ask / eval / docs
tests/              # 130 用例（conftest 封网：永不外呼 LLM；test_v07.py=v0.7 四支柱）
evals/              # default 10 + agent 8 + innolight 31 + maotai 9 + quarters 8 + multihop 8 + multiturn 10 轮
samples/            # 3 家公司 5 份真实财报（巨潮）
docs/               # 设计文档 ×7（v0.4–v0.7/incremental/scale）+ EXPLAINER + TEST-CASES + RAG-EVAL-CASES
.env                # 密钥（gitignored，永不进 git）
```

## 密钥安全

- 默认本地 Ollama **无需任何密钥**；接远端时 key 只放 `.env`
  （已 gitignore），`_load_dotenv` 装载且环境变量优先
- 密钥不进代码、不进文档、不进示例命令、不进 git 历史
- 测试套件 conftest 全局封网（与机器上是否装 Ollama 无关），CI 无密钥全绿

## Roadmap

- [x] ~~向量检索可选后端（bge-small-zh @ fastembed），与词法双路 RRF~~
  （消融结论 DESIGN-v0.6 §8.3：数字/口径题净负贡献，默认关，DZ_VEC=on opt-in）
- [x] ~~多公司对比（v0.5 compare 分解）~~
- [x] ~~trend 双闸 + 裁判自测（v0.6.5：LLM 默认在环，保守失败兜底）~~
- [x] ~~**v0.7 Agent 化升级**（`docs/DESIGN-v0.7-agent.md`，封闭世界契约下
  的四支柱 + 长程可靠性设计）：ReAct 步级观测（thought/action/obs/gate
  四元组 + waste_rate 度量）→ Planner-Executor 多跳规划（子目标检查点 +
  只删不造校验器）→ 三层记忆（L0 会话压缩继承 / L1 verify 门控事实台账
  / L2 固定语义资产）→ 工具注册表（search / metrics_table /
  derive 差分算子）~~（quarters/multihop/multiturn 三集验收）
- [x] ~~跨期对比（「2025 vs 2026」类跨年双值题）~~（multihop 集 + Planner）
- [ ] rerank 层（LLM 或 cross-encoder 可切换）
- [ ] Web UI（对账视图：答案句 ↔ 原文高亮）

## License

MIT
