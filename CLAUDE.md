# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

对账 duizhang — a Chinese financial-report **Agentic RAG** QA engine ("every answer must be reconcilable against the source"). Since v0.5.1 the LLM defaults to a LOCAL Ollama model (qwen3.5:0.8b, zero key/cost/egress): it drives query understanding, routing, retrieval grading, rewriting, multi-company compare decomposition, and grounded generation. Deterministic code owns the exit gate (`verify_citations`): degrade or refuse, never fabricate. The rule mode (`heuristic`) remains as the zero-key fallback and regression baseline — same graph, two brains.

All code comments, docstrings, docs, and commit messages are in Chinese — follow that convention.

## Commands

```bash
source venv/bin/activate                # Python 3.12; deps already installed

# Ingest & query (auto = llm if key in .env, else heuristic)
python -m duizhang.cli ingest samples/  # incremental ingest (fingerprint diff; idempotent)
python -m duizhang.cli ingest samples/ --watch   # poll mode: resync every 60s
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli ask "对比欧科亿和海伦哲 2026 一季度谁赚得多"   # compare → decompose
python -m duizhang.cli ask "…" -v          # 诊断态：trace 每节点带耗时（stderr）
python -m duizhang.cli ask "…" --json      # 机器态：整态 JSON
python -m duizhang.cli eval --json         # 机器态：summary+行级 rows（对比脚本用）
python -m duizhang.cli eval                          # default dataset (10 单跳; CI regression)
python -m duizhang.cli eval --mode llm --dataset agent     # agent dataset (multi-hop; local 0.8b ok)
python -m duizhang.cli eval --dataset innolight      # 中际旭创真值 31 题（事实/趋势/陷阱/鲁棒；CI 冒烟）
python -m duizhang.cli eval --dataset maotai         # 贵州茅台真值 9 题（简称/口径/断词泛化）
python -m duizhang.cli docs             # list ingested docs
python -m duizhang.cli index            # (re)build FTS for existing chunks
python -m duizhang.cli embed            # 可选向量第三路：嵌入存量 chunks（需 requirements-vec.txt）

# Tests / lint
pytest                                  # 93 cases; conftest blocks all real LLM calls
pytest tests/test_store.py::test_readd_after_drop_no_ghost   # one test
ruff check .                            # lint (CI enforces)
```

Corpus layout: `samples/` = small demo set; `data/corpus/` = full corpus (1,104 docs / 60.7万 chunks); `data/中际旭创/` = ground-truth corpus for the innolight dataset (10 docs / 7,873 chunks, **26MB committed to git** — CI ingests it and runs the 31-题 truth smoke; the innolight eval is not reproducible without this corpus); `data/贵州茅台/` = second-company corpus (10 docs / 6,518 chunks, ~10MB committed, same 2024–2026 period structure — anchors the 9-题 maotai truth set, the multi-company generalization baton). Roots coexist in one DB (multi-root ingest). Eval logic lives in `evalx.py` (`judge()` is a pure function; drop custom datasets into `evals/*.json`).

After changing retrieval/graph/answer code, run `python -m duizhang.cli eval --mode heuristic` (deterministic baseline — plain `eval` defaults to `auto`, which silently goes llm when Ollama is up and adds 0.8b variance) and verify the three headline metrics don't regress (full corpus: 1.00/1.00/1.00); `evalx.py` additionally reports faithfulness (pass/coverage) and chunk-level P/R@8 — check those too on retrieval changes. Bug fixes follow TDD — failing test first, watch it fail, then fix (DESIGN-v0.4.md §5 and DESIGN-v0.5.md record two such rounds).

Secrets: `DZ_LLM_API_KEY` lives ONLY in `.env` (gitignored; `_load_dotenv` loads it, env vars take precedence). Never write the key into code, docs, sample commands, or git history. Default channel is LOCAL Ollama via its NATIVE `/api/chat` (`llm._local_chat`): the OpenAI-compat `/v1` endpoint ignores the `think` param, `/no_think` soft switch is dead in Qwen3.5, and thinking starves max_tokens leaving content empty (all three measured — DESIGN-v0.5.md D7). Local endpoints need no key (`llm_enabled` is True for localhost). `think` defaults OFF locally (~×5 latency when on). Remote options via `.env`: any OpenAI-compatible endpoint, or Anthropic-compatible (`_use_anthropic`; thinking counts toward max_tokens → budget ×8). Requires Ollama ≥ 0.40. Other env: `DZ_DB`, `DZ_LLM_MODEL`, `DZ_LLM_MAX_TOKENS`, `DZ_LLM_THINKING`, `DZ_TREND_ANALYSIS` (trend 意图的投资分析层， default OFF — the gate guarantees numbers come from tables, not that 「大幅回落」isn't narrated backwards by 0.8b), `DZ_VEC` (vector third track, default OFF — see Retrieval).

## Architecture

Two planes, one SQLite file (`docs` / `chunks` / `fts_w` / `fts_c` / `manifest`, plus optional `vecs` / `vecmeta`):

**Offline ingest** — `source.py` scans a directory into `{path: mtime_ns-size fingerprint}` and three-way diffs against the `manifest` table (add/update/delete + orphan cleanup scoped to the scan root). `ingest.py` extracts metadata from the filename (巨潮 convention → company/period/kind — never from model guessing; broker-report naming `日期-券商-公司` also parses; md/txt with a CJK company-name stem ingest as ANALYSIS, **anything else is rejected with ⚠** — a digit-stem like `20261009_…分析.md` used to become garbage doc-id `20261009_MD`), strips repeated header/footer lines (frequency-based, digits stripped in normalization — headers share lines with page numbers), chunks with table-awareness, and appends number-normalization notes (`41,914.72 万元` → `≈4.19亿`) so lexical search hits 「多少亿」-style queries.

**Online QA graph** (`graph.py`) — LangGraph state machine over `GraphState` (narrow-interface TypedDict; `trace` has append semantics):

```
START → understand ─→ route ─metric/compare(llm)──→ agent_research（工具循环：单一 search 工具，
       （LLM 抽取          │                          模型自主迭代检索，步数≤6，幂等缓存+重复收手，
        intent/公司/期间，   │─metric/compare(heuristic)→ retrieve/decompose→grade⇄rewrite  理解先验优先）
        规则兜底）          │─trend（双模式共用）──→ trend_research（确定性遍历最近 6 期，
                           │─summary→ summarize_doc    「主要会计数据」锚定，时间升序）
                           │                          ↓ generate → verify_citations（确定性闸门）
                           └─other──→ refuse
```

**trend 意图**（v0.6.2）— `trend_research` deterministically walks the company's reporting periods (≤6, chronological — the 0.8b agent left periods out / looped when asked to do it itself), then `trend_skeleton` parses each period into structured rows （营收/归母净利 + 同比 + 引用） and the LLM only narrates (direct multi-period generation by 0.8b contradicted itself — measured). No fact rows → refuse. Two guards: query must contain a trend word or intent downgrades to metric (0.8b hallucinated trend for 「预计…多少」); rule layer overrides when the LLM misjudges trend as summary. Multi-period evidence still exits through generate→verify.

**Retrieval** (`fts.py` + `index.py`) — dual-track FTS5 + RRF (k=60): word track (jieba + unicode61) and char track (trigram, whitespace-stripped, ≤200 chars — truncates 14.2% of chunks by design, word track covers full text). An optional third vector track (`embed.py`: bge-small-zh-v1.5 @ fastembed, same RRF) is **default OFF** — opt in via `DZ_VEC=on` + `pip install -r requirements-vec.txt` + `cli embed`; the 31-题 truth ablation (DESIGN-v0.6 §8.3, D19) found lexical dual-track already saturated on numeric/caliber tasks while embeddings pull semantically-near-but-wrong-caliber chunks (net negative). Missing dep / model mismatch / un-embedded DB → `ranking()` returns None → silent dual-track fallback (CI runs without fastembed). `understanding` (from the understand node) drives `doc_scope`: exact company match + period restriction at the doc level (period *type* + *year* — `query_years` extracts 20xx and colloquial two-digit years 「15年」→2015, must be followed by 年 to avoid amount false-positives). `expand_query` maps colloquial terms to report terminology (营收→营业收入, 赚→净利润, 原因→主要系). Generation context is assembled by `page_evidence` (small-to-big, DESIGN-v0.6 §5.3: dedup top-hit pages → ±2 sibling-chunk window → hard cap); metric intent additionally fronts the 「主要会计数据」 table page (truth rows lose lexical fights to number-less header chunks — measured).

### Invariants — violate these and you reintroduce past bugs

1. **`Store` (`store.py`) is the ONLY writer to the six tables** (vecs/vecmeta included — `add_vecs`/`set_vec_model` are the only vector writers; changing the vecmeta model wipes all stale vecs). fts rowids align to `chunks.rowid`; re-add = drop-then-insert (never `INSERT OR REPLACE` on chunks), and drop_doc clears the doc's vecs rows. `chunks(doc_id)` is indexed in SCHEMA (legacy DBs auto-migrate on next Store open). `tests/test_store.py` is the permanent regression guard.
2. **One graph, two brains.** Every LLM decision point falls back to rules on failure **with a trace marker** (`understand:llm→fallback` etc.). `--mode auto` resolves at `new_state`: llm if key present, else heuristic. Tests never make real LLM calls (conftest strips the key).
3. **The corrective loop is bounded** (`MAX_RETRIES = 1` heuristic; `MAX_AGENT_STEPS = 6` for the agent loop). llm-mode metric/compare route into `agent_research` (single search tool; the model iterates autonomously); trend never enters the loop (deterministic walk, both modes). Guardrails the loop needs with small models (all measured, DESIGN-v0.6.md): idempotent call cache + early stop on all-repeat batches; understanding priors OVERRIDE tool args for period (0.8b hallucinates FY) and backfill company/period when omitted; agent search terms are written into `effective_query`; evidence pool emits round-robin by doc. Agent-collected evidence still goes through generate→verify — autonomy never exempts the gate.
4. **Refusal over fabrication.** Gates: period guard in `route` (year from doc metadata, not the model); doc-level period filter (`period_hints` is mutually exclusive — "半年度" contains "年度", substring-ordered ifs let FY leak, round-4 bug R-1; bare year + no period word → FY since D18); `verify_citations` (evidence pool = page-level set **∪ hits** — page windows only cover hits[:4] while extractive quotes hits[:8]; citation matching is whitespace-normalized since D13 — small-model citation formats drift 「第 8 页」; cited chunks incl. num_notes, meta-token-filtered both sides, numeric sentences only, newline-aware splitting; < 0.45 → degrade to extractive; LLM no-evidence → extractive, not refuse). Metric caliber is guarded at **five layers** since D16/D20–D22: `stitch_adjacent` re-joins same-page label/number table chunks (runs in retrieve AND agent paths); trend `pick()` is LINE-level (splitlines, first big number after the indicator word, label-context caliber guard = 20 chars before the indicator, rank-fallback when the best line fails parse) and extractive scoring both refuse 扣非 rows unless the query asks 扣; **paired-caliber demotions ×5** in extractive (asking X demotes sibling-Y sentences that share label-prefix tokens: 扣非↔归母 / 净利润↔净资产 / 总收入↔营业收入 / 总资产↔净利润族 / 总资产↔营收族); and `caliber_mismatch` (D22) — an LLM-mode numeric gate: asking a named caliber (扣非/总收入/净资产/总资产) requires the answer's big numbers to appear in that caliber's labeled rows (source = page set ∪ hits, next-line pickup for split label/value rows, segment-cut for 营业总收入) else degrade to extractive — the first closing of the gate≠correctness boundary (§9). `ans.SPLIT_FIX` broken-label repairs are shared by extractive / trend pick / generate evidence assembly — keep the three in sync. Fronting (`front_key_table`, cap 8 ∪ metric-word-matching tail chunks) runs in BOTH retrieve and agent paths; extractive window is hits[:12]; `page_evidence` is page-stratified (first-hit ±2 window per page — the union-window variant measured net-negative on 0.8b, context dilution).
5. **SQLite PRAGMAs are per-connection** (`fts._apply_pragmas` every search; Store sets WAL + synchronous=NORMAL; `store.analyze()` after changed ingests).
6. **Multi-root ingest**: deletion/orphan cleanup scoped to the scan root (`under_root` in `cli.py`). `--rebuild` still wipes everything.
7. **Thinking is tiered** (`llm._thinking_off`): structured calls off (cheap enums), generation on (number association on fragmented table evidence needs it; flash without thinking quotes increase-ranges as metric values). The `extra_body.thinking` param is sent ONLY to bigmodel endpoints.

## Docs

`docs/DESIGN-v0.6.md` is the authority on the agent loop & small-model guardrails (decision records D9–D12), the trend layer (§6), the innolight-truth fixes of v0.6.3 (§7: D13/D14/D16/D18), and the P1 batch (§8: CI truth smoke, eval expansion 19→31 + faithfulness/chunk-P/R metrics, vector ablation D19); §9 is the known-boundary ledger (gate ≠ correctness — read before touching `verify_citations`; sample C = per-share broken-label, unfixed); §10 records the multi-company generalization batch (D20 short-alias normalization, D21 line-level trend extraction, D22 caliber numeric gate, paired demotions, front-in-retrieve, stratified evidence — all measured on the Moutai corpus); `DESIGN-v0.5.md` on the agentic foundations (D1–D6); `DESIGN-v0.4.md` on retrieval/write architecture (§5 = round-4 review), `DESIGN-incremental.md` on incremental ingest, `DESIGN-scale.md` capacity, `EXPLAINER.md` narrated walkthrough. `TEST-CASES.md` is the test-case manual (four test layers / acceptance cases / trace reading); `RAG-EVAL-CASES.md` covers eval-set design. Design docs record *why* — check before re-litigating, update when architecture changes. `docs/plans/` holds implementation plans.
