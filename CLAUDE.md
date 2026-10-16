# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

对账 duizhang — a Chinese financial-report **Agentic RAG** QA engine ("every answer must be reconcilable against the source"). Since v0.5 the LLM (zhipu GLM by default) is the default brain: it drives query understanding, routing, retrieval grading, rewriting, multi-company compare decomposition, and grounded generation. Deterministic code owns the exit gate (`verify_citations`): degrade or refuse, never fabricate. The rule mode (`heuristic`) remains as the zero-key fallback and regression baseline — same graph, two brains.

All code comments, docstrings, docs, and commit messages are in Chinese — follow that convention.

## Commands

```bash
source venv/bin/activate                # Python 3.12; deps already installed

# Ingest & query (auto = llm if key in .env, else heuristic)
python -m duizhang.cli ingest samples/  # incremental ingest (fingerprint diff; idempotent)
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli ask "对比欧科亿和海伦哲 2026 一季度谁赚得多"   # compare → decompose
python -m duizhang.cli eval                          # default dataset (CI regression)
python -m duizhang.cli eval --mode llm --dataset agent   # agent dataset (multi-hop; costs API calls)
python -m duizhang.cli docs             # list ingested docs
python -m duizhang.cli index            # (re)build FTS for existing chunks

# Tests / lint
pytest                                  # 49 cases; conftest blocks all real LLM calls
pytest tests/test_store.py::test_readd_after_drop_no_ghost   # one test
ruff check .                            # lint (CI enforces)
```

After changing retrieval/graph/answer code, run `python -m duizhang.cli eval` (heuristic) and verify the three metrics don't regress (full corpus: 1.00/1.00/1.00). Bug fixes follow TDD — failing test first, watch it fail, then fix (DESIGN-v0.4.md §5 and DESIGN-v0.5.md record two such rounds).

Secrets: `DZ_LLM_API_KEY` lives ONLY in `.env` (gitignored; `_load_dotenv` loads it, env vars take precedence). Never write the key into code, docs, sample commands, or git history. Two provider channels: v4 pay-per-use (`/api/paas/v4`, default glm-4.5-flash free tier) and GLM Coding Plan (`/api/anthropic` — plan quota is ONLY valid there; `llm._use_anthropic()` auto-switches protocol; glm-5.3 always thinks and thinking counts toward max_tokens, so the anthropic path multiplies the budget ×8). Other env: `DZ_DB`, `DZ_LLM_MODEL`, `DZ_LLM_MAX_TOKENS`, `DZ_LLM_THINKING`.

## Architecture

Two planes, one SQLite file (`docs` / `chunks` / `fts_w` / `fts_c` / `manifest`):

**Offline ingest** — `source.py` scans a directory into `{path: mtime_ns-size fingerprint}` and three-way diffs against the `manifest` table (add/update/delete + orphan cleanup scoped to the scan root). `ingest.py` extracts metadata from the filename (巨潮 convention → company/period/kind — never from model guessing), strips repeated header/footer lines (frequency-based, digits stripped in normalization — headers share lines with page numbers), chunks with table-awareness, and appends number-normalization notes (`41,914.72 万元` → `≈4.19亿`) so lexical search hits 「多少亿」-style queries.

**Online QA graph** (`graph.py`) — LangGraph state machine over `GraphState` (narrow-interface TypedDict; `trace` has append semantics):

```
START → understand ─→ route ─metric─→ retrieve → grade_docs ⇄(≤1) rewrite_query
       （LLM 抽取          │─compare→ decompose（按公司拆解、交错合并）→ grade
        intent/公司/期间，   │─summary→ summarize_doc                      ↓ generate
        规则兜底）          └─other──→ refuse                    verify_citations（确定性闸门）
```

**Retrieval** (`fts.py` + `index.py`) — dual-track FTS5 + RRF (k=60): word track (jieba + unicode61) and char track (trigram, whitespace-stripped, ≤200 chars — truncates 14.2% of chunks by design, word track covers full text). `understanding` (from the understand node) drives `doc_scope`: exact company match + period restriction at the doc level. `expand_query` maps colloquial terms to report terminology (营收→营业收入, 赚→净利润, 原因→主要系).

### Invariants — violate these and you reintroduce past bugs

1. **`Store` (`store.py`) is the ONLY writer to the four tables.** fts rowids align to `chunks.rowid`; re-add = drop-then-insert (never `INSERT OR REPLACE` on chunks). `chunks(doc_id)` is indexed in SCHEMA (legacy DBs auto-migrate on next Store open). `tests/test_store.py` is the permanent regression guard.
2. **One graph, two brains.** Every LLM decision point falls back to rules on failure **with a trace marker** (`understand:llm→fallback` etc.). `--mode auto` resolves at `new_state`: llm if key present, else heuristic. Tests never make real LLM calls (conftest strips the key).
3. **The corrective loop is bounded** (`MAX_RETRIES = 1`). compare/decompose joins the same grade→generate→verify path — multi-hop is never exempt from the reconciliation gate.
4. **Refusal over fabrication.** Gates: period guard in `route` (year from doc metadata, not the model); doc-level period filter (`period_hints` is mutually exclusive — "半年度" contains "年度", substring-ordered ifs let FY leak, round-4 bug R-1); `verify_citations` (evidence pool = cited chunks incl. num_notes, meta-token-filtered both sides, numeric sentences only, newline-aware splitting; < 0.45 → degrade to extractive; LLM no-evidence → extractive, not refuse).
5. **SQLite PRAGMAs are per-connection** (`fts._apply_pragmas` every search; Store sets WAL + synchronous=NORMAL; `store.analyze()` after changed ingests).
6. **Multi-root ingest**: deletion/orphan cleanup scoped to the scan root (`under_root` in `cli.py`). `--rebuild` still wipes everything.
7. **Thinking is tiered** (`llm._thinking_off`): structured calls off (cheap enums), generation on (number association on fragmented table evidence needs it; flash without thinking quotes increase-ranges as metric values). The `extra_body.thinking` param is sent ONLY to bigmodel endpoints.

## Docs

`docs/DESIGN-v0.5.md` is the authority on the agentic layer (decision records D1–D6); `DESIGN-v0.4.md` on retrieval/write architecture (§5 = round-4 review), `DESIGN-scale.md` capacity, `EXPLAINER.md` narrated walkthrough. Design docs record *why* — check before re-litigating, update when architecture changes. `docs/plans/` holds implementation plans.
