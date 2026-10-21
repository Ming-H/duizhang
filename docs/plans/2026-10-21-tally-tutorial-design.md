# Tally 项目重命名 + 《动手学 Agent RAG》教程设计（2026-10-21 定案）

> 本文档是命名、教程全案与重命名决策的唯一权威记录。brainstorming 过程
> 已收敛，此处只记定案与理由，防会话丢失。

## 一、命名定案

### 项目名：Tally（repo: `tally`）

- **词源即卖点**：中世纪符木（tally stick）刻痕后劈成两半，两半对得上
  才算数——人类最早的「对账/凭证」。答案与原文对上刻痕才允许出门。
- **README hero**：`Tally 🪵 — 答案与原文对上刻痕，才允许出门。财报
  事实 Agent：LLM 组织语言，确定性闸门拥有事实。`
- 中文语境可表述为「刻痕」，原项目名「对账 duizhang」退役（会计黑话、
  外行无感）；**「对账」作为概念词保留**（对账闸门/对账思想），只换品牌。

### 教程名：《动手学 Agent RAG》

- 无副标题（副标题两版均被否，越解释越丑）；系列惯例即定位。
- 副行（非副标题）：`基于开源项目 Tally 的证据式问答系统教程——
  13 章 · 13 个破坏性实验 · 40 道真值题`。
- 备选存档（均否决）：从 Demo 到不敢编造 / 只认刻痕的证据系统 /
    Agent RAG 实战 / 造一个不说谎的 AI / 工程实录 / 言之有据 /
    十三课 / 答案的底气 / 无征不信。

## 二、教程定位（brainstorming 收敛结果）

| 决策点 | 定案 | 理由 |
|---|---|---|
| 构建路径 | **讲解型代码导读**，无 mini 系统 | 教程只贴关键片段（文件名+函数名锚定，行号辅助）；完整代码读者 clone tally 仓库运行 |
| 目标读者 | **有 RAG 经验的进阶者** | 基础知识点（什么是 embedding 等）降为决策驱动讲法：不讲定义，讲在哪撞墙 |
| LLM 通道 | 讲解现场分析代码，运行用 **API key**（OpenAI 兼容） | 读者都有 key；本地 Ollama 作零成本附录 |
| 语料 | **中际旭创 + 贵州茅台 20 份双语料** | 40 道人工核对真值题直接复用；CI 锁死实验可复现性；版权保留「下载脚本替代」备注 |
| 载体 | **GitHub 开源仓库（唯一内容源）+ 自动构建静态站**；个人网站只做链接橱窗 | 可复现/面试证据/传播/免费；双源必漂移 |
| 构建工具 | mkdocs-material | Python 栈一致、中文教程生态成熟、零 node |

## 三、全书结构：四部 13 章

```
第一部 知识入库——PDF 怎么变成可检索的证据（ingest.py / store.py / fts.py）
  1. 开篇：为什么你的 RAG Demo 上不了线（对账思想，tally 30 秒跑起来）
  2. PDF → 大模型可读的内容：文件名元数据契约 / 页眉剥离 / 扫描版拒收
  3. 切分：表格感知分块——毁掉证据的 N 种方式
  4. 存储选型：SQLite FTS5 vs 向量库 vs ES（容量实测决策法）
  5. 检索：双轨 FTS + RRF，以及「要不要 embedding」的正确打开方式

第二部 求证闸门——答案凭什么出门（graph.py 前半 / answer.py）
  6. 查询理解与路由：LLM 当大脑、规则当防线（八守卫的诞生）
  7. 生成与引用：grounded generation
  8. 对账闸门：verify / 降级 / 拒答（全书灵魂章）
  9. 口径守卫：问扣非为什么不能答归母（五层防线）

第三部 Agent 化——四支柱（graph.py 后半 / memory.py）
  10. ReAct 工具环：白名单 / 步级观测 / 幂等缓存 / 重复收手
  11. Planner 与 Memory：只删不造 / 三层记忆 / 读不豁免

第四部 诚实——评测与边界（evalx.py / DESIGN §9）
  12. 评测即测试：真值集 / 陷阱题 / 反向断言 / CI 门槛
  13. 边界账本：你的系统管不住什么（诚实收尾 + 可靠性公理）
```

**每章五段式模板**：
1. 失败先行（真实事故或破坏实验的结果截图/输出）
2. 决策与被否备选（引 DESIGN 文档 D 编号）
3. tally 关键片段精讲（10-40 行，函数名锚定，注释里的事故故事重点展开）
4. 动手：跑/改/看（命令 + 预期输出 + 指标变化）
5. 习题与延伸（fork 上小改动 + DESIGN 链接）

## 四、13 个破坏性实验（教学签名）

| 章 | 实验 | 观察 |
|---|---|---|
| 1 | 跑起来 + 问 2030 年净利 | 期间守卫拒答 |
| 2 | 日期开头文件名 ingest / 空白 PDF | ⚠ 拒收 / 0 块闸门 |
| 3 | `_is_table_line` 恒 False 跑 eval | answer_hit 崩塌 |
| 4 | `.schema` + EXPLAIN QUERY PLAN + 容量表 | 为什么不上 ES |
| 5 | `TALLY_VEC=on` + embed 重跑真值集 | 复现 D19 消融 1.00→0.83-0.93 |
| 6 | `-v` 看八守卫 trace；monkeypatch 假 LLM | 每条守卫对应真实事故 |
| 7 | `--json` 看引用；改坏引用格式 | verify 抓漂移 |
| 8 | COVERAGE_THRESHOLD 调 0.0/1.0 | 阈值是工程不是玄学 |
| 9 | 问扣非看不含归母数；删配对降权重问 | 六对降权必要性 |
| 10 | `-v` 读 agent_steps；MAX_AGENT_STEPS=2 | 步级观测；预算砍半退化* |
| 11 | multihop/multiturn；TALLY_MEMORY=off | 「未覆盖」红线；L0 断链 |
| 12 | 改坏 expect_keyword + --min-hit 1.0 | 门槛通电，CI 真的会红 |
| 13 | 亲跑 §9 三个越界样本 | gate ≠ correctness |

\* ch10 诚实标注：远端通道 ReAct 未接（DESIGN-v0.7 §8），实验用白盒
回放测试替代真跑，留作读者 v1.1 挑战题（远端 function calling）。

## 五、仓库与联动

```
../hands-on-agent-rag（教程，唯一内容源）
├── docs/ch01..ch13.md          ← 章节源
├── experiments/chXX.sh         ← 实验脚本化
├── mkdocs.yml                  ← 静态站配置（GitHub Pages/Vercel）
└── .github/workflows/drift.yml ← 防漂移：定期跑全部实验，tally 升 tag
                                    后验证教程输出没变——教程用自己证明自己
tally（代码）⇄ 教程互链；引用锚点=文件名+函数名（抗漂移），行号辅助
```

**发布节奏**：仓库先推 3 章试水（ch1/5/8，差异化最强的三张牌）→
掘金/知乎连载 → 反馈好出 mkdocs PDF/平台小册。教程优先级永远让位于
求职窗口。

## 六、重命名决策（duizhang → tally）

- 包目录 `duizhang/` → `tally/`（git mv 保历史）；全仓 import、
  `python -m duizhang.cli`、docs、CI、run_tests.sh 同步。
- 环境变量 `DZ_*` → `TALLY_*`（11 个：LLM_API_KEY/BASE_URL/MODEL/
  MAX_TOKENS/THINKING、TREND_ANALYSIS、SELF_CHECK、VEC、MEMORY、
  EVAL_MEMORY、DB）；本地 .env 静默迁移；默认 DB 路径 duizhang.db →
  tally.db（本地文件一并改名）。
- 历史 DESIGN 文档里的 DZ_ 提及**一并改名**（面向未来读者的可复现性
  优先于历史保真；v0.7 文档加一行重命名注记）。
- 「对账」概念词全保留；GitHub 仓库 rename 与本地文件夹 rename 留给
  用户手动执行（避免破坏进行中会话的 cwd 与记忆路径）。
