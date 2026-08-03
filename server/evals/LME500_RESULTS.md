# LongMemEval-500 (oracle) — full-size run, COMPLETE

Config: `ARM=ours PER_TYPE=200 CHUNK_TURNS=2 CONCURRENCY=2 USE_ONTOLOGY=0 TOP_K=40 READER_MODEL=gpt-4.1`
Substrate: FalkorDB, one graph per question. Run 2026-07-29 → 2026-08-03.

## Result: **447/500 = 89.4%** — all 500 questions answered

| question type | correct | n | acc |
|---|---|---|---|
| knowledge-update | 71 | 78 | 91% |
| multi-session | 117 | 133 | 88% |
| single-session-assistant | 54 | 56 | 96% |
| single-session-preference | 28 | 30 | 93% |
| single-session-user | 66 | 70 | 94% |
| temporal-reasoning | 111 | 133 | 83% |

## vs the pre-registered prediction

Registered 2026-07-14, BEFORE running: **76%** (band 70–80), reasoning that the full set is
53% temporal+multi-session (our weakest types) and that the n=60 had adaptive tuning.

Actual: **89.4%** — **+13.4 points**, above the band.

Read: the fix stack generalized rather than overfitting. The prediction's *reasoning* was
still sound — temporal-reasoning IS the weakest type (83%) and multi-session second-weakest
(88%) — but both held far better at scale than feared. The n=60 result (90%) was not a
small-sample artifact: the full-set number is within 1 point of it.

Reference measured in THIS harness: Zep's own recipe = 58% (n=60).
Published elsewhere: Zep 71.2% on LongMemEval-**S** (a different, harder split — do not conflate).

## Provenance / how to read the raw data

- `longmemeval_results.jsonl` is append-only. Authoritative read = **last NON-error row per**
  **`question_id`** (`load_done()` semantics). A later error row must never erase a paid answer;
  a naive last-wins read under-reports by ~19 questions.
- Questions whose graphs ingest left incomplete were re-ingested and re-answered in a repair
  pass; those later rows supersede. Temporal went 80% → 83% once its 13 partial-graph
  questions were answered against complete memory.
- The run survived repeated Mac sleeps and two FalkorDB OOM kills. No answer was ever lost —
  per-answer durability held throughout. Operational post-mortem: `MEMORY_LAYER.md` §4.
