# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

对账 duizhang — a Chinese financial-report RAG QA engine ("every answer must be reconcilable against the source"). LangGraph orchestrates a Corrective-RAG loop; SQLite FTS5 provides lexical retrieval; every answer sentence carries a `[公司_报告期 第N页]` citation. When evidence doesn't check out, the system degrades or refuses rather than fabricates.

All code comments, docstrings, docs, and commit messages are in Chinese — follow that convention.

## Commands

```bash
source venv/bin/activate                # Python 3.12; deps already installed

# Ingest & query (zero API key needed — heuristic mode is the default and first-class)
python -m duizhang.cli ingest samples/  # incremental ingest (fingerprint diff; idempotent)
python -m duizhang.cli ask "欧科亿 2026 年半年度预计净利润多少"
python -m duizhang.cli eval             # built-in eval: retrieval@3 / answer hit / refusal
python -m duizhang.cli docs             # list ingested docs
python -m duizhang.cli index            # (re)build FTS for existing chunks (migration/corruption)

# Tests
pytest                                  # 30 cases (pyproject sets pythonpath; bare pytest works)
pytest tests/test_store.py              # one file
pytest tests/test_store.py::test_readd_after_drop_no_ghost   # one test
ruff check .                            # lint (CI enforces)
```

After changing retrieval/graph/answer code, run `python -m duizhang.cli eval` and verify the three metrics don't regress (full corpus & samples: 1.00 / 1.00 / 1.00; see README's dual-scope table). `tests/` fixtures build a real DB from `samples/` through the production path (Store → Corpus), not mocks. Bug fixes follow TDD — write the failing test first, watch it fail, then fix (see DESIGN-v0.4.md §5 for the round-4 review that established this).

Environment variables:
- `DZ_DB` — DB path (default `duizhang.db`, a 752MB gitignored local corpus)
- `DZ_LLM_API_KEY` / `DZ_LLM_BASE_URL` / `DZ_LLM_MODEL` — optional OpenAI-compatible LLM layer; `ask "..." --mode llm` swaps 4 decision points to model calls. `DZ_LLM_MAX_TOKENS` overrides budget (reasoning models eat the default).

## Architecture

Two planes, one SQLite file (`docs` / `chunks` / `fts_w` / `fts_c` / `manifest`):

**Offline ingest** — `source.py` scans a directory into `{path: mtime_ns-size fingerprint}` and three-way diffs against the `manifest` table (add/update/delete + orphan cleanup). `ingest.py` extracts metadata from the filename (巨潮 convention `公司_2026年半年度业绩预告.pdf` → company/period/kind — never from model guessing), strips repeated header/footer lines, chunks with table-awareness (short digit-dense lines cluster into atomic table chunks, never split mid-table), and appends number-normalization notes (`41,914.72 万元` → `≈4.19亿`) so lexical search hits 「多少亿」-style queries.

**Online QA graph** (`graph.py`) — LangGraph state machine over `GraphState` (a narrow-interface TypedDict in `schemas.py`: each node reads only it needs, writes only what it owns; `trace` uses `operator.add` for append semantics):

```
route ─metric→ retrieve → grade_docs ─insufficient & retries<1→ rewrite_query → retrieve
  │summary→ summarize_doc                    │sufficient
  └other→ refuse                             ↓
                                      generate → verify_citations ─fail→ extractive degrade / refuse
```

**Retrieval** (`fts.py` + `index.py`) — dual-track FTS5 fused with RRF (k=60): word track (`fts_w`, jieba pre-tokenized + unicode61 tokenizer) and char track (`fts_c`, trigram over whitespace-stripped text, ≤200 chars — measured to truncate 14.2% of chunks, word track covers full text; decision recorded in DESIGN-v0.4.md §4/§5). `Corpus` is a thin facade: docs metadata in memory, chunks fetched on demand via SQL. Query-side: `expand_query` maps colloquial metric names to report terminology (营收→营业收入) and adds attribution synonyms (原因→主要系) — without this, causation questions never recall. Period hard-filter is **doc-level** (`doc_scope` in `fts.search`): when the queried company has a doc of the queried period, retrieval is restricted to those doc_ids — a hit-level post-filter lets large annual-report chunks crowd the target period out of top-8 (round-4 bug R-2).

### Invariants — violate these and you reintroduce past bugs

1. **`Store` (`store.py`) is the ONLY writer to the four tables.** v0.4 collapsed the previous dual write path that produced "ghost chunks" (stale FTS hits). `fts_w`/`fts_c` rowids are aligned to `chunks.rowid`; re-adding a doc is drop-then-insert, because `INSERT OR REPLACE` on chunks assigns new rowids while orphaning old FTS rows. `chunks(doc_id)` has an index in SCHEMA (`IF NOT EXISTS` — legacy DBs auto-migrate on next Store open). `tests/test_store.py` is the permanent regression guard.
2. **Heuristic and LLM modes share one graph.** LLM calls wrap every decision point in try/except and fall back to rules **with a trace marker** (`route:llm→fallback` etc.) — silent in behavior, visible in trace. LLM generation falls back to extractive. The demo must never break for lack of an API key. When adding a decision point, keep this dual-mode pattern.
3. **The corrective loop is bounded** (`MAX_RETRIES = 1` via conditional edges). Unbounded retry is a state-machine antipattern.
4. **Refusal over fabrication.** Gates: period guard in `route` (year not in corpus → refuse, decided from doc metadata not the model), doc-level period filter in `retrieve` (`period_hints` is mutually-exclusive matching — "半年度" contains "年度", substring-ordered ifs let FY leak into H1 queries, round-4 bug R-1), `verify_citations` in `answer.py` (evidence pool = chunks actually cited in the answer, falling back to `hits[:4]`; coverage < 0.45 → degrade to extractive; empty evidence → refuse), and summarize's domain-word guard. Preserve them when touching the graph.
5. **SQLite PRAGMAs are per-connection** (`fts._apply_pragmas` runs on every search call) — don't hoist them to a one-time global. The Store write connection additionally sets WAL + `synchronous=NORMAL`; `store.analyze()` is called after ingests that changed anything.
6. **Multi-root ingest**: deletion and orphan cleanup are scoped to the current scan root (`under_root` in `cli.py`) — ingesting root B must never drop root A's docs. `ingest --rebuild` still wipes everything.

## Docs

`docs/DESIGN-v0.4.md` is the authority on the current retrieval/write architecture (decision records with alternatives and rejection reasons); `DESIGN-scale.md` capacity derivation, `DESIGN-incremental.md` manifest design (partially superseded), `EXPLAINER.md` the narrated walkthrough. Design docs record *why* — check them before re-litigating a decision; update them when architecture changes.
