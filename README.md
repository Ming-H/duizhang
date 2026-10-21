# 对账 duizhang 🧾

[![CI](https://github.com/Ming-H/duizhang/actions/workflows/ci.yml/badge.svg)](https://github.com/Ming-H/duizhang/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **RAG 的每个答案都要能对账。**

**封闭世界约束下的事实 Agent**（v0.7：四支柱齐备）：LLM 决定「怎么查、怎么讲」——查询理解、多跳规划、自主迭代检索、多轮记忆继承、grounded 生成；确定性代码决定「答案能不能出门」——引用覆盖率闸门不过就降级或拒答。逐句挂引用、引用必可回溯到财报原文的页与段；对不上的答案宁可拒答，也不编。

> **零 key 是默认，不是上限。** LLM 通道双轨：本地 Ollama **任意档位**模型（零 key / 零费用 / 零外呼，出厂默认小档位）或远端 API key（任意 OpenAI / Anthropic 兼容端点，`.env` 三行切换）。档位只改善「怎么讲」，不改变「能不能出门」——数字正确性由确定性闸门兜底，小档生成垮掉时自动降级抽取式，答案仍然逐句对账。**可靠性由架构保证，不靠模型堆料。**

新读者从这里开始：**[docs/INTRODUCTION.md](docs/INTRODUCTION.md) — 项目导览**。

## 目录

- [核心特性](#核心特性)
- [30 秒跑起来](#30-秒跑起来)
- [LLM 配置：三档通道](#llm-配置三档通道)
- [CLI 一览](#cli-一览)
- [架构](#架构)
- [评测](#评测)
- [项目结构](#项目结构)
- [文档索引](#文档索引)
- [FAQ](#faq)
- [Roadmap](#roadmap)

## 核心特性

| 特性 | 一句话 | 实测锚点 |
|---|---|---|
| **对账闸门** | 答案数字句必须能在证据池里回溯到原文页段，覆盖率 < 阈值自动降级抽取式或拒答 | verify 从不豁免任何路径（Agent/台账/差分） |
| **封闭世界契约** | 事实只来自内部 RAG 证据；工具白名单只有内部动作（search / metrics_table / derive），永无浏览器/外部 API/模型生成代码；参数知识只组织语言不产出事实 | 封闭世界让「对账」可判定——开放世界 verify 永远证伪不了「模型碰巧知道」 |
| **Agent 四支柱** | Planning（子目标清单+只删不造校验）/ Memory（L0 会话继承+L1 事实台账+L2 语义资产）/ Tool Use（三工具注册表）/ ReAct（步级四元组观测） | DESIGN-v0.7 §4 |
| **口径守卫体系** | 六对指标配对降权 + 口径数值闸（D22）：问扣非不得拿归母、问总收入不得拿营业收入、问总资产不得拿货币资金冒充 | gate≠correctness 边界的第一处收口（DESIGN-v0.6 §10） |
| **多公司泛化** | 简称归一（茅台→贵州茅台）、行级抽取、断词修复——任意丢一家 A 股公司财报进语料即插即用 | 贵州茅台 10 份零调参 9/9 双脑全绿 |
| **确定性优先** | trend 走确定性期间遍历；单季数 Q2=H1−Q1 / Q4=FY−Q3 是原币精度差分算子+出口重算，**禁止模型心算** | 算术也是事实（封闭世界） |
| **双大脑** | LLM 模式与零 key 规则模式（heuristic）同一张图；LLM 故障逐点退规则并留 trace 标记 | heuristic 是可复现的回归基线 |
| **评测内建** | 7 数据集 / 84 题；retrieval@3 · answer hit · refusal · citation_coverage · chunk P/R@8 · steps waste_rate | `python -m duizhang.cli eval` 一条命令复现 |
| **观测性三档** | 产品态（pass 静默）/ `-v` 诊断态（trace 每节点带耗时，stderr）/ `--json` 机器态（整态+行级 rows） | 修前/修后对比可脚本化 |

## 30 秒跑起来

```bash
pip install -r requirements.txt            # 精确复现已测环境改用 requirements.lock.txt
python -m duizhang.cli ingest samples/     # 摄取样例财报（3 家公司 5 份文件）
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli eval                # 内置评估集，一条命令复现
```

规则模式（heuristic）零依赖零 key 即可用——上面三条命令不需要任何 LLM 配置。要上 LLM 大脑见下节。

## LLM 配置：三档通道

所有配置写 `.env`（已 gitignore，绝不进 git）。通道由 `DZ_LLM_BASE_URL` 自动识别，**换脑不改代码**：

| 档位 | 端点 | 需要 | 说明 |
|---|---|---|---|
| 0（默认） | `http://localhost:11434/v1` | 装 [Ollama](https://ollama.com) ≥0.40 + pull 模型 | 零 key 零费用零外呼；原生 `/api/chat` 支持 think 参数（默认关，开约 ×5 延迟） |
| 1 | 任意 OpenAI 兼容端点 | `DZ_LLM_API_KEY` + `BASE_URL` | deepseek / 智谱 `/api/paas/v4` / GLM Coding Plan `/api/coding/paas/v4` / gpt 等 |
| 2 | Anthropic 兼容端点 | `DZ_LLM_API_KEY` + `BASE_URL` | 如 `https://open.bigmodel.cn/api/anthropic`；thinking 计入 max_tokens → 预算自动 ×8 |

```bash
cp .env.example .env    # 按注释填三行即可切档
```

```bash
# 档位 1 示例：智谱 Coding Plan（本仓实测通道）
DZ_LLM_API_KEY=你的key
DZ_LLM_BASE_URL=https://open.bigmodel.cn/api/coding/paas/v4
DZ_LLM_MODEL=glm-5.3
```

**通道能力差异**：ReAct 工具循环（agent_research）目前**仅本地档位生效**（Ollama 原生 tools）；远端档位 metric/compare 自动走固定管线兜底——答案同闸（verify 不豁免），但步级观测/幂等缓存不参与（DESIGN-v0.7 §8 实测记录）。

**完整环境变量参考**：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DZ_LLM_API_KEY` | 空 | 只存 `.env`；本地端点免 key |
| `DZ_LLM_BASE_URL` | localhost:11434 | 通道识别（本地 / OpenAI 兼容 / Anthropic 兼容） |
| `DZ_LLM_MODEL` | `qwen3.5:0.8b` | 编排假设不绑档位——换大档叙述上移，闸门兜底不变 |
| `DZ_LLM_MAX_TOKENS` | 按档位 | 生成 token 预算 |
| `DZ_LLM_THINKING` | `disabled` | 仅影响生成；结构化调用永远关 |
| `DZ_DB` | `./duizhang.db` | SQLite 库路径 |
| `DZ_TREND_ANALYSIS` | 未设=叙述开 | 三态：默认数字闸+方向闸双闸兜底；`enabled` 加深度投资分析；`off` 骨架直出（逃生门） |
| `DZ_SELF_CHECK` | 开 | LLM 裁判非对称信任：只能否决不能解锁；`off` 省一次调用 |
| `DZ_MEMORY` | 开 | L1 事实台账总闸；`off` 退无记忆行为 |
| `DZ_EVAL_MEMORY` | 隔离 | eval 内默认隔离台账，`on` 显式放开（防止把检索能力测成记忆命中） |
| `DZ_VEC` | 关 | 向量第三路 RRF（需 `requirements-vec.txt` + `cli embed`）；消融实测数字/口径题净负，opt-in 实验 |

## CLI 一览

```bash
python -m duizhang.cli ingest samples/            # 增量摄取（manifest 指纹 diff，幂等）
python -m duizhang.cli ingest data/corpus --watch # 轮询模式：每 60s 增量同步
python -m duizhang.cli ask "对比欧科亿和海伦哲 2026 一季度谁赚得多"   # 多跳对比
python -m duizhang.cli ask "茅台 2026 半年度扣非净利润多少" -v        # 诊断态：trace 每节点耗时
python -m duizhang.cli ask "…" --json            # 机器态：整态 JSON
python -m duizhang.cli chat                      # 多轮对话（L0 记忆：追问继承公司/期间/指标/年份）
python -m duizhang.cli docs                      # 已入库文档清单
python -m duizhang.cli index                     # 为存量 chunks 重建 FTS
python -m duizhang.cli embed                     # 可选向量第三路（需 requirements-vec.txt）
python -m duizhang.cli eval [--mode m] [--dataset d] [--json]
```

`-v` 诊断态示例（stderr，产品态 pass 时静默）：

```
trace: start(358ms) → understand:rule(0ms) → route:metric(80ms) → retrieve:14 hits(1ms)
       → grade:2.0 ok(5ms) → generate:extractive(4ms) → verify:pass(0.967)  总耗时 0.45s
```

## 架构

```
START → understand ──→ memory_probe ─→ route ──多跳信号──→ 【plan 规划】─→ 【agent_research ReAct 工具循环】─→ generate ──→ verify_citations
      （LLM 结构化理解：   （L1 台账探针：    │            ≥3公司/跨年/≥2指标族      子目标白名单校验(只删不造)      search/metrics_table/derive       ↓ fail
       intent/公司/期间，   命中仍要过 verify） │                                    search/metrics_table/derive       （确定性闸门，不豁免）
       规则兜底）                            ├─metric/compare(heuristic)→ retrieve/decompose → grade ⇄ rewrite(≤1) ↗  抽取式降级/拒答
                                            ├─trend─→ trend_research（确定性期间遍历 + Q2=H1−Q1 / Q4=FY−Q3 差分派生）
                                            └─other─→ refuse
```

**Agent 四支柱与边界**：

| 支柱 | 落地 | 边界（不破坏对账底线） |
|------|------|------|
| **Planning** | `plan` 节点：≥3 公司/跨年/≥2 指标族的多跳问题先出结构化子目标清单，逐个执行；检查点（数字在场 ∧ 指标词命中）逐个验证 | 校验器**只删不造**：白名单外动作丢弃、去重、封顶；缺口=部分答案不硬凑 |
| **Memory** | L0 会话（近 3 轮全量+更早压缩为结构化摘要，**永不继承自由文本**）/ L1 事实台账（键=公司+期间+指标+口径，五算子合并）/ L2 语义资产（alias 词典） | 写闸=verify:pass ∧ 口径闸未触发；**读不豁免**——台账命中拼的答案照样过 verify；语料指纹变更即失效 |
| **Tool Use** | 注册表三工具：`search` / `metrics_table` / `derive`（单季差分，原币精度） | 白名单只含内部动作；**禁止模型心算**——出口重算，操作数必须真在场 |
| **ReAct** | 步级四元组 `thought/action/obs/gate` 进观测，`-v` 逐行、`--json` 可重放 | 步数≤6 + 幂等缓存 + 整批重复即收手；理解先验覆盖工具参数防幻觉参数 |

**长程可靠性**（为什么拉长任务链之前先装闸门）——单步 95% 可靠的任务 20 步只剩 ~36% 成功，该改的是传播结构：

```
P(错误答案) ≤ Σₖ eₖ·gₖ    eₖ=第k步出错率（模型决定）  gₖ=错误穿过该步闸门的存活率（代码决定）
e=5%、g=2%、20 步 → ≤2%（朴素模型 36% 失败），且失败形态=降级/拒答（安全失败）
```

四个杠杆：能确定化的环节不给模型（trend 遍历/差分算子）；verify-as-you-go（子目标检查点把拦截推进到步级）；结构化传递（字段复制不文本重释）；并行分支不乘法（compare 按公司分解、各自过闸）。`eval --json` 的 `steps.waste_rate` 是 eₖ 的可测代理。

**检索**：SQLite FTS5 双轨（jieba 词轨 + trigram 字轨）RRF 融合（k=60），磁盘倒排使容量与内存解耦——60.7 万块实测：稳态 39ms / 内存 0.03GB / 启动 0.05s / 增量摄取 0.8s（`docs/DESIGN-scale.md`）。可选向量第三路（bge-small-zh @ fastembed）默认关：数字/口径题上词法已饱和，向量净负贡献（消融表 DESIGN-v0.6 §8.3）。

**值得看代码的决策点**：

| 决策 | 为什么 |
|------|--------|
| **understand 查询理解** | LLM 一次抽取 intent/公司/期间驱动路由与 `doc_scope`——v0.4 的关键词子串猜测是三轮 bug 的根源 |
| **compare 多跳分解** | 按公司拆子查询、各自 doc_scope 检索、交错合并——两家证据都在场，且多跳不豁免闸门 |
| **Corrective-RAG 纠错环** | `grade → rewrite → retrieve` 的**有界**重试（≤1）——无界重试是状态机的反模式 |
| **trend 双闸 + 裁判自测** | 数字闸（叙述数字 ∈ 骨架）→ 方向闸（方向词与同比符号相容，拦「+90.05% 说成大幅回落」）→ 结构化裁判（非对称信任）；任一违例退骨架=保守失败 |
| **derive 差分派生** | 「Q2/Q4 净利」原文不披露——累计数差分（Q2=H1−Q1、Q4=FY−Q3），原币精度，出口重算；**封闭世界里算术也是事实** |
| **期间守卫** | 问「2030 年净利润」直接拒——年份来自文档元数据不靠模型；期间过滤互斥匹配防 FY 泄漏 |
| **多公司泛化** | 简称双向子串归一（D20）+ 行级抽取（D21）+ 口径数值闸（D22）——「任意丢一家 A 股公司进来就能用」 |

## 评测

**指标定义**：

| 指标 | 定义 |
|---|---|
| retrieval@3 | 期望文档进入 top-3 检索 |
| answer hit | 答案命中人工核对的原文关键词（真值集逐题 SQL 对库核对） |
| refusal | 陷阱题（未来期间/语料外/领域外）是否正确拒答；无陷阱集记 N/A |
| citation_coverage | 非拒答题中 verify pass 占比 / 平均引用覆盖率（原名 faithfulness——测的是引用对账，不是语义忠实度，v0.7 改名） |
| chunk P/R@8 | 真值承载块（doc∈期望 ∧ 含关键词）在 top-8 的占比与命中下界 |
| steps | ReAct 步级观汇总：calls/new/dup/empty + `waste_rate=(dup+empty)/calls` |

**结果**（双语料同库 24 docs / 14,701 chunks；heuristic 与档位无关可复现）：

| 数据集 | 模式 | retrieval@3 | answer hit | refusal |
|---|---|---|---|---|
| default（10 题，CI 回归） | heuristic | 1.00 | 1.00 | 1.00 |
| agent（8 题：多跳对比+陷阱） | heuristic / llm | 1.00 | 1.00 | 1.00 |
| innolight（31 题真值） | heuristic | 1.00 | 1.00 | 1.00 |
| innolight | llm | 1.00 | 0.963（每股收益=已知边界 §9 样本 C） | 1.00 |
| **maotai（9 题真值：简称/口径/断词泛化）** | heuristic / llm | **1.00** | **1.00** | **1.00** |
| **quarters（8 题：单季差分/期间守卫）** | heuristic / llm | **1.00** | **1.00** | 1.00 |
| **multihop（8 题：跨年双值/三公司/多指标族）** | heuristic | 1.00 | 1.00 | 1.00 |
| **multiturn（10 轮接力：继承/口径切换/代词指代）** | heuristic / llm | **1.00** | **1.00** | N/A |

**语料分工**：`samples/`（3 家公司 5 份财报，演示与 CI 回归）；`data/中际旭创/`（10 docs / 7,873 chunks，**26MB 随 git 入库**——31 题真值集的承重墙）；`data/贵州茅台/`（10 docs / 6,518 chunks，10MB 随 git 入库——多公司泛化的磨刀石）；`data/corpus/`（全量 1,104 docs，本机自备不入 git）。多根摄取同库共存，删除/孤儿清理只对本次扫描根负责。

**架构比模型更保命**：小模型生成不足时 verify 闸门自动降级抽取式——满分的含金量在「理解+检索+闸门」，不在生成模型。CI 每次推送跑 120 用例 + README 流程冒烟 + 真值三集冒烟（innolight 31 + quarters 8 + multihop 8 = 47 题）。

## 项目结构

```
duizhang/
├── schemas.py      # GraphState（窄接口状态）+ Doc/Chunk/Citation/Retrieved
├── source.py       # 语料扫描 + manifest 三态 diff（增量账本）
├── ingest.py       # 文件名元数据 / 表格感知分块 / 页眉剥离 / 数字归一化 / 查询扩展
├── store.py        # 六表唯一写入口（docs/chunks/fts_w/fts_c/manifest/verified_facts）
├── index.py        # 分词原语 + 瘦身 Corpus（docs 全量、chunk 按需）
├── fts.py          # FTS5 双轨检索 + RRF 融合 + doc_scope（可选向量第三路）
├── graph.py        # LangGraph 状态机（understand/memory_probe/plan/agent/trend/generate/verify）
├── memory.py       # Memory 三层（L0 会话 / L1 事实台账 / L2 语义资产）
├── answer.py       # 抽取式生成 + 口径配对降权 + 引用覆盖率校验（对账闸门）
├── llm.py          # LLM 层：三档通道自动识别 / .env 装载 / 思考分层
├── evalx.py        # 评估 runner（judge 纯函数；7 数据集；steps/citation_coverage/chunk P/R）
└── cli.py          # ingest / ask / chat / docs / index / embed / eval（-v/--json 三档观测）
tests/              # 120 用例（conftest 封网：永不外呼 LLM）
evals/              # 7 数据集 58 题（default/agent/innolight/maotai/quarters/multihop/multiturn）
samples/            # 3 家公司 5 份真实财报（巨潮）
data/               # 中际旭创 26MB + 贵州茅台 10MB（随 git）；corpus 全量本机自备
docs/               # 设计文档 ×5 + 导览 + 测试手册 + 评估集设计（见下表）
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/INTRODUCTION.md](docs/INTRODUCTION.md) | **新读者导览**：定位、问题、架构白话版、阅读路径 |
| [docs/DESIGN-v0.7-agent.md](docs/DESIGN-v0.7-agent.md) | Agent 化升级：四支柱、封闭世界公理、长程可靠性 §3、Phase 1-4 落地 |
| [docs/DESIGN-v0.6.md](docs/DESIGN-v0.6.md) | Agent 环与小模型护栏（D9–D22）：页级证据、trend 层、真值驱动修复、口径数值闸、多公司泛化、已知边界账本 §9 |
| [docs/DESIGN-v0.5.md](docs/DESIGN-v0.5.md) | Agentic 地基（D1–D6）：理解/分解/纠错环、LLM 通道实测（D7） |
| [docs/DESIGN-v0.4.md](docs/DESIGN-v0.4.md) | 检索/写入架构：FTS5 双轨、Store 唯一写入口、§5 = round-4 评审 |
| [docs/DESIGN-incremental.md](docs/DESIGN-incremental.md) | 增量摄取：指纹 diff 与多根共存 |
| [docs/DESIGN-scale.md](docs/DESIGN-scale.md) | 容量推导：60.7 万块实测与扩展路径 |
| [docs/EXPLAINER.md](docs/EXPLAINER.md) | 全流程讲解：一条 query 的完整旅程 |
| [docs/TEST-CASES.md](docs/TEST-CASES.md) | 测试手册：四层测试 / 验收用例 / trace 阅读 |
| [docs/RAG-EVAL-CASES.md](docs/RAG-EVAL-CASES.md) | 评估集设计方法论 |
| docs/plans/ | 实施计划存档 |

设计文档记录 *why*——动手前先查，改架构后更新。

## FAQ

**没装 Ollama / 不想用本地模型，能用吗？**
能。三选一：规则模式（零 LLM，`--mode heuristic`）；任意 OpenAI 兼容远端；Anthropic 兼容远端。`.env` 三行切换，见[上表](#llm-配置三档通道)。

**怎么换模型档位？**
改 `DZ_LLM_MODEL` 一行。编排假设不绑档位——小档靠闸门兜底，大档叙述上移；实测记录见 DESIGN-v0.5 D7 与各评估表。

**怎么用自己的财报？**
```bash
python -m duizhang.cli ingest /path/to/财报目录
```
PDF 按巨潮命名「公司_2025年年度报告.pdf」；md/txt 主干必须是公司名。解析不出公司名的文件 ⚠ 拒收，不静默入库。

**怎么加评估题？**
`evals/*.json` 丢一道 `{q, expect_keywords, expect_doc}`（陷阱题加 `expect_refused: true`），`--dataset 文件名` 即跑。真值必须人工对库核对——没题的语料是死重。

**为什么问「2030 年净利润」会被拒？**
期间守卫：年份来自文档元数据（不靠模型），语料里没有该期间就拒——对账纪律优先于有用性。

**测试会外呼 LLM 吗？**
不会。conftest 全局剥 key 封网，与机器上装没装 Ollama、配没配远端无关；CI 无密钥全绿。

## Roadmap

- [x] ~~多公司泛化（v0.6.4：D20 简称/D21 行级/D22 口径闸）~~
- [x] ~~Agent 四支柱（v0.7：Planner/Memory/Tool/ReAct）~~
- [ ] 每股收益断词标签修复（§9 样本 C，innolight llm 30/31 的最后一题）
- [ ] 跨期对比（「2025 vs 2026 毛利率变化」→ 按期间分解）
- [ ] rerank 层（LLM 或 cross-encoder 可切换）
- [ ] 向量轨语义问法实验（D19 留的口子：「为什么/怎么样」类是否受益）
- [ ] Web UI（对账视图：答案句 ↔ 原文高亮）

## 密钥安全

- key 只放 `.env`（已 gitignore），`_load_dotenv` 装载且环境变量优先；不进代码、不进文档、不进示例命令、不进 git 历史
- 测试套件 conftest 全局封网；CI 无密钥全绿

## License

代码：[MIT](LICENSE) · Copyright © 2026 Ming-H。

语料版权另论：`data/` 下的财报 PDF 来自巨潮资讯网公开披露，版权归原作者与披露主体，随库携带仅用于本仓库的可复现评测（CI 真值冒烟）；对外分发本仓库前应移除语料或另行取得授权。
