# Memory-Layer Benchmark Runbook

Harnesses validating the memory layer. Read `/MEMORY_LAYER.md` first.

## Results (updated 2026-08-03)

- **LongMemEval-oracle FULL 500: 89.4% (447/500)** — assistant 96, user 94, pref 93, KU 91,
  multi-session 88, temporal 83. Pre-registered prediction was 76% (band 70–80), so **+13.4,
  above the band**: the n=60 tuning generalized. Report + provenance:
  [`LME500_RESULTS.md`](LME500_RESULTS.md); raw rows in `longmemeval_results.jsonl`.
- **LongMemEval-oracle n=60 (10/type): 90%** — the full-500 lands within 1 point, so the small
  sample was not an artifact. Zep's recipe in this same harness: 58%.
- **LoCoMo n=50 (2 convs, stratified): 78%** — multi-hop 83, temporal 82, single-hop 82,
  adversarial 73, open-domain 60. No per-question tuning.
- **LoCoMo-1986 still NOT run.** Prediction ≈74% (69–78). Before starting: prune per-question
  graphs (see `MEMORY_LAYER.md` §4 — FalkorDB OOM) and keep the machine on AC.

**Reading `*_results.jsonl`:** append-only; authoritative = the **last NON-error row per
question id**. A naive last-wins read (counting error rows) under-reports — an error must never
erase a paid answer. Questions answered against an incomplete graph score low and unfairly:
audit graph completeness before trusting a weak per-type number (temporal went 80→83% purely
by re-ingesting 13 partial graphs and re-answering).

## Setup

- FalkorDB: `docker run -d --name graphiti-ambient-demo -p 6379:6379 falkordb/falkordb:latest`
- `OPENAI_API_KEY` in repo-root `.env`. Datasets: `longmemeval_oracle.json`
  (HuggingFace xiaowu0162/longmemeval-cleaned) and `locomo/data/locomo10.json`
  (github snap-research/locomo) placed next to the harness.
- Run from `server/`: `uv run python evals/longmemeval_harness.py` (env-config below).
- **Keep the Mac awake** — AC power + lid open. `caffeinate -i` does NOT stop clamshell or
  battery maintenance sleep; sleep leaves the process alive on dead sockets, so a stall looks
  exactly like a slow run. For long runs use the supervisor, which restarts on stall:

  ```bash
  PER_TYPE=200 CHUNK_TURNS=2 CONCURRENCY=2 USE_ONTOLOGY=0 TOP_K=40 \
    ./evals/run_supervised.sh longmemeval 500
  ```

  Args are `<longmemeval|locomo> <target answer count>`; `STALL_SECS` (default 900) is how
  long without a new answer counts as stalled. Restarting is always safe — resume works both
  per-answer and per-episode (see below).

## LongMemEval

```bash
OPENAI_API_KEY=... ARM=ours PER_TYPE=10 CHUNK_TURNS=2 CONCURRENCY=2 \
  SKIP_INGEST=0 USE_ONTOLOGY=0 TOP_K=40 READER_MODEL=gpt-4.1 \
  uv run python evals/longmemeval_harness.py
```

Knobs: `PER_TYPE` (10=60q, 200=full 500) · `CHUNK_TURNS=2` turn-pair episodes (**the** structural
fix — 0 = whole-session, scores much lower) · `SKIP_INGEST=1` QA-only on existing graphs ·
`INGEST_ONLY=1` · `QTYPE_FILTER` / `QID_FILTER` for micro-tests (~$0.30) · `ARM=zep` runs
Graphiti's own recipe (cross-encoder + their context string) as the baseline arm.

Results → `longmemeval_results.out` (report) + `longmemeval_results.jsonl` (one line per
answer, written as each completes). Both ingest **and QA are resumable**: re-running skips
already-answered questions (errored answers re-run; `ARM` is tagged per line so ours/zep runs
never mix; `QID_FILTER` bypasses resume for fix→retest) and ingests only **genuinely missing
episodes** — a graph left partial by a killed run is topped up, not mistaken for complete.
No manual `GRAPH.DELETE` needed. Delete the jsonl for a fresh scoring run.

## LoCoMo

```bash
OPENAI_API_KEY=... N_CONV=2 QA_PER_CONV=25 CHUNK_TURNS=2 CONCURRENCY=2 \
  uv run python evals/locomo_harness.py
```

Full run: `N_CONV=10 QA_PER_CONV=999` (~$60–80, ~15–18h at 30k TPM). Results →
`locomo_results.out` + `locomo_results.jsonl` (appended per answer). QA is **resumable**:
re-running skips already-answered questions (errored ones re-run) and fully-answered
conversations entirely. Delete the jsonl for a fresh scoring run.

## recall_diag.py — use this first

Reader-free evidence-recall (did retrieval surface the gold `has_answer` turns?). ~Free
(embeddings only) — the right metric for retrieval iteration; never burn reader money tuning
retrieval. Current: semantic-episode 86% vs edge-ref 79%.

## Cost discipline (learned the hard way)

Reader+judge (gpt-4.1 × N) is the cost; ingest is mini. Micro-test pattern: fix → rerun ONLY the
failing qids (`QID_FILTER`, ~$0.30) → verify → next. Decouple ingest from QA. One LLM job at a
time (30k TPM). Never trust a run with ERRs in it — errors score as wrong and fake regressions.
