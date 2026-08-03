# LongMemEval-500 (oracle) — full-size run

Config: `ARM=ours PER_TYPE=200 CHUNK_TURNS=2 CONCURRENCY=2 USE_ONTOLOGY=0 TOP_K=40 READER_MODEL=gpt-4.1`
Substrate: FalkorDB, one graph per question. Run 2026-07-29 → 2026-08-02.

## Result: **442/499 = 88.6%** (499/500 answered)

| question type | correct | n | acc | of total |
|---|---|---|---|---|
| knowledge-update | 71 | 78 | 91% | 78 |
| multi-session | 117 | 133 | 88% | 133 |
| single-session-assistant | 54 | 56 | 96% | 56 |
| single-session-preference | 28 | 30 | 93% | 30 |
| single-session-user | 66 | 70 | 94% | 70 |
| temporal-reasoning | 106 | 132 | 80% | 133 |

**Pre-registered prediction (2026-07-14, before running): 76% (band 70–80).**
Actual 88.6% — **+12.6 points vs prediction**, above the band.
The n=60 tuning generalized rather than overfitting.

Reference points measured in THIS harness: Zep's own recipe = 58% (n=60).
Published elsewhere: Zep 71.2% on LongMemEval-**S** (not oracle — do not conflate).

## Caveats

- Raw rows live in `longmemeval_results.jsonl` (append-only; the last NON-error row per
  question_id is authoritative — a later error must never erase a paid answer).
- Some temporal questions were answered against graphs that ingest left incomplete;
  a repair pass (top-up ingest + re-answer) supersedes those rows. Scoring a question
  against partial memory understates it, so the temporal number is a floor until the
  repair completes.
- Run was repeatedly interrupted by Mac sleep and by FalkorDB OOM kills; see
  `MEMORY_LAYER.md` §4. No answer was ever lost — per-answer durability held throughout.
