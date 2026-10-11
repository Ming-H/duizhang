# 对账 v0.3 设计：全量语料 + 自适应增量摄取

> **时效说明**：增量/排除/研报通道等核心设计仍有效；其中「cli 直写 SQL」的
> 写入路径已在 v0.4 收敛至 `store.py` 单一写入口（见 DESIGN-v0.4.md D-3）。


## 1. 需求

1. 语料范围：`finance-analysis` 全库，**排除 `0 分析结果`**（自产分析，另一条产品线）；
   含行业财报（31 行业/770 公司）、`00 研报集合`（券商研报）、各公司 MD/TXT。
2. **自适应增长**：新增/修改/删除财报后，一条命令增量同步，已入库的不重复处理。
3. 研报是异质文档：命名 `YYYYMMDD-券商-标题.pdf`，无公司/报告期结构，需要独立元数据通道。

## 2. 业界做法调研（选型依据）

| 模式 | 出处 | 取舍 |
|---|---|---|
| **Manifest 指纹增量**（path → mtime+size 指纹，diff 出增/改/删） | Airbyte/LlamaIndex 目录连接器、dbt 增量模型 | ✅ 采用。最简单可靠，无守护进程依赖 |
| **内容寻址去重**（content hash 定身份） | 句读 cninfo 导入、git blob | ✅ 文件名解析不出 doc_id 时兜底（研报标题哈希） |
| **冷热分层**：存储增量追加，索引版本化缓存、**惰性重建** | 搜索引擎 epoch-based indexing、ES refresh | ✅ 采用。sqlite 增量落库；BM25 倒排 pickle 缓存 + 指纹校验，不匹配才重建（60万块 117s） |
| 实时 watcher（watchdog/inotify 常驻） | LlamaIndex watch、Haystack | ⚠️ 不引入常驻依赖；给 `--watch` 轮询模式（60s，零依赖），日常用 cron/手动 |
| 向量库 UPSERT 全量重算 | Chroma/MilvUs 常见误用 | ❌ 明确不做：N 文档改动重算全库是反模式 |

**设计原则**（延续 v0.1/v0.2）：零守护进程、零新必装依赖（watchdog 可选）、索引可随时从 sqlite 重建（pickle 只是缓存，删了也能跑）。

## 3. 数据模型变更

```sql
-- 新增 manifest 表：语料 ↔ 文档 的增量账本
CREATE TABLE manifest (
  path TEXT PRIMARY KEY,      -- 语料绝对路径
  doc_id TEXT,                -- 关联文档
  fingerprint TEXT,           -- "{mtime_ns}-{size}"，轻量指纹（内容级去重交给文件名/doc_id）
  ingested_at TEXT
);
-- chunks/docs 不变：doc_id 主键，删除文档即级联删块
```

## 4. 模块设计

### 4.1 `duizhang/source.py`（新）：语料扫描与 diff

- `scan(root, exclude=("0 分析结果",))` → `{path: fingerprint}`，后缀白名单 `pdf/md/txt`，
  跳过 `.DS_Store/csv/png/html`（数据工件非正文）。
- `diff(manifest, scanned)` → `to_add / to_update(指纹变) / to_delete(manifest 有、磁盘无)`。

### 4.2 `ingest.py`：研报元数据通道

财报命名走现行 `_PERIOD_PAT`；**研报**新增模式：

```
(\d{8})-(券商)-(标题).pdf → Doc(kind="研报", period=日期, company=券商)
doc_id = 研报_{broker}_{date}        （同日同券商撞车时拼标题 hash6）
```

TXT 一等支持（`美迪西2026中报.txt` 类裸文本：kind=文本，company 取文件名主干）。

### 4.3 `index.py`：索引缓存（pickle，惰性重建）

- `Corpus.save_index(path.pkl)`：倒排 postings + idf + dl 序列化，头带指纹
  `sha1(chunk_ids)`。
- `Corpus.load(db, idx=None)`：pickle 指纹匹配 → 直接热启（<5s）；不匹配/缺失 → 重建（117s）并回写。
- 语义：**pickle 永远只是缓存**，删除后行为不变只是慢一次。

### 4.4 `cli.py`：增量成为默认

```
python -m duizhang.cli ingest data/corpus     # 增量：diff → 增/改/删 → 落库 → 索引缓存失效
python -m duizhang.cli ingest --rebuild       # 全量重建（破坏性，先清库）
python -m duizhang.cli ingest --watch         # 轮询 60s 循环增量（零依赖）
```

删除路径：`to_delete` 的 doc 从 docs/chunks/manifest 三表清除。

## 5. 故障与边界

| 场景 | 行为 |
|---|---|
| 坏 PDF（加密/扫描） | 逐文件 try/except 跳过并计数，不阻塞批次 |
| 同名同指纹重复运行 | manifest 命中 → 跳过（幂等） |
| 文件改名 | 旧 path 进 to_delete，新 path 进 to_add（doc_id 不变则块级天然接续） |
| 摄取中途 ^C | sqlite 单事务提交前不落 manifest → 下次自动重做该文件 |
| 研报标题含特殊字符 | doc_id 只取 券商+日期，标题进 Doc 元数据不进 id |

## 6. 测试计划

- 研报文件名解析（日期/券商/标题三元组）
- diff 三态（tmp 目录造 add/update/delete）
- 排除目录不入库
- 幂等：连续两次 ingest，第二次 0 新增
- 索引缓存：指纹不匹配自动重建
- 回归：pytest 原有 10 用例全绿

## 7. Roadmap 对齐

向量第三路（bge + RRF）接口不变；增量管线对向量库同样适用（embed 新块 upsert）。
