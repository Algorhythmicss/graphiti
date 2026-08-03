# Memory Layer — Master Handoff Doc

> **Read this first.** This file lets a fresh contributor (human or model) pick up the entire
> project in one pass. Companion: `server/evals/README.md` (benchmark runbook).
> Last updated: 2026-08-03.

## 1. Mission

This repo (a fork of getzep/graphiti) hosts a **general-purpose memory-context layer** for AI
assistants and agents: ingest conversations/emails/messages/transcripts → temporally-aware
knowledge graph → serve **reactive** (query→context) and **proactive** (rolling
window→briefing-or-SILENCE) retrieval, with trust (per-fact confidence, validity windows,
supersession), legibility (why-chains, citations), and a feedback loop (outcome-labeled decision
traces). Target: a genuinely SOTA, widely-usable memory layer.

**Flagship use case: Kyra** (trykyra.io) — a proactive cross-app communication assistant ("one
interface for everything that needs your attention"; iOS/macOS first, minimal glasses+ring
later). Kyra drives requirements (the COMMITMENT/TASK/MEETING ontology, salience gating,
message-grain ingestion) but nothing in the layer is Kyra-specific — any agent needing durable,
trustworthy memory can consume the same endpoints.

A separate repo `/Users/mac/memory-engine` is a research prototype: **port its ideas, never its
code** (it deliberately keeps heavy research baggage).

## 2. Where everything lives

- **Fork remote**: `fork` = github.com/Algorhythmicss/graphiti (push here). `origin` =
  getzep/graphiti (**NEVER push**). PR #1 (merged): the whole layer. PR #2 (open): Phases 5–7.
- `server/graph_service/ambient.py` — proactive assembly: two-level salience gate
  (block-level "should I speak at all" + per-fact floor), relevance-first fill, content near-dup
  guard, soft per-entity cap, token budget, `effective_confidence` (persisted w/ decay, pinned,
  TTL), citations with why-chains.
- `server/graph_service/context_assembly.py` — reactive `/get-context`: three-channel context
  (SUPERSEDED-tagged validity-window facts [authoritative] + entity profiles [advisory] +
  semantically-ranked raw episodes), retrieve-ALL facts for counting queries.
- `server/graph_service/consolidation.py` — Phase 5 job: TTL sweep + stale-summary regeneration
  from open facts, **withholding the old summary** from the LLM. `POST /consolidate/{group_id}`.
- `server/graph_service/decision_trace.py` — Phase 6: DecisionTrace ledger per ambient call,
  `POST /ambient-outcome` (engaged/dismissed/ignored), 4h proactive suppression.
- `server/graph_service/ontology.py` — evidence-driven: nouns = entity types
  (Person/Organization/Project); **actions = EDGE types** (COMMITMENT/TASK/MEETING/WORKS_AT/
  PREFERENCE) with structured attrs (due_date, status, direction). Types are a SIGNAL never a
  dedup key.
- `graphiti_core/utils/confidence.py` — (rating, uncertainty) + corroborate/contest/read-time
  decay. Wired into `edge_operations.py` (corroborate on re-assertion, contest on contradiction).
  Six trust columns persisted on EntityEdge across Neo4j/Falkor/Kuzu.
- `graphiti_core/prompts/extract_edges.py` rule 6 — **in-passing/subordinate-clause facts MUST be
  extracted** (measured fix: "my snake plant, which I got from my sister" class).
- `server/evals/` — LongMemEval + LoCoMo harnesses + reader-free `recall_diag.py`.

## 3. Benchmark journey + honest numbers

`8% → 57% → 67% plateau → root-caused fix stack → 90%` on LongMemEval-oracle (n=60, 10/type),
**LoCoMo 78%** (n=50, 2 convs, ZERO per-question tuning — the cleaner number).
References measured in OUR harness: Zep's own recipe = 58%. Published: Zep 71.2% (on
LongMemEval-**S**, not oracle — don't conflate), Mem0 ~66–68% LoCoMo.

**The fix stack (each traced to root cause; see git log):**
1. **Chunked/message-grain ingestion** (CHUNK_TURNS=2) — THE structural fix: session-level
   extraction drops enumerable items. multi-session 20→86%, temporal 40→100% non-error.
   The server already ingests one message/episode — the product grain is correct.
2. **In-passing extraction rule** (core prompt) — subordinate-clause facts.
3. **SUPERSEDED tags** on closed-window facts — **marking beats instructing** (three instruction
   variants failed; the explicit tag fixed knowledge-update stale-value picks). Graphiti's
   contradiction-invalidation was already setting invalid_at correctly.
4. **Facts-authoritative-over-profiles** — unversioned node summaries leak stale values (the
   consolidation job is the write-side fix).
5. Retrieve-ALL facts for counting (ranking is wrong for enumeration); split evidence char budget
   (flat truncation cut answers); multi-query event decomposition; pref-mode reader
   (recommend from stated interests, never abstain) + alignment-style preference judging;
   date-math + latest-supersedes reader rules; FINAL-line parsing.

**LME-500 RESULT (2026-08-03): 447/500 = 89.4%** — all 500 answered. Per type: assistant 96,
user 94, preference 93, knowledge-update 91, multi-session 88, temporal-reasoning 83. Full
report + provenance: `server/evals/LME500_RESULTS.md`.

Pre-registered 2026-07-14 (BEFORE running): **76%** (70–80), reasoning the full set is 53%
temporal+multi-session and the 60 had adaptive tuning. **Actual +13.4 above prediction, outside
the band.** The tuning generalized — the n=60 90% was not a small-sample artifact (full set is
within 1 point). The prediction's *reasoning* still held qualitatively: temporal (83) and
multi-session (88) ARE the two weakest types, just far stronger at scale than feared.

**LoCoMo-1986 still pending**, prediction ≈ **74%** (69–78) registered on the same date. Costs:
~$60–80/~15–18h (30k-TPM gpt-4.1 bound). Harness is ready and now crash/sleep-resilient —
before starting, PRUNE per-question graphs (see §4 OOM) and keep the machine on AC.

## 4. Hard-won infra gotchas (will bite you)

- **FalkorDB is multi-graph**: each `group_id` = its own graph. Search/reads on an unscoped
  driver hit an empty default graph → **reader answers from nothing** (this faked our first 8%).
  Use `Graphiti(FalkorDriver(database=group_id))` for BOTH ingest and QA; ALSO
  `build_indices_and_constraints()` explicitly (background index build races QA). Neo4j
  (production) is single-graph — unaffected.
- **FalkorDB OOM is the #1 killer of a full-size run** (cost us ~2 days on LME-500, 2026-07-31).
  One graph per question means 500 graphs in ONE instance; at ~500 graphs it held ~1.5GB and
  forked a ~314MB RDB snapshot every 300s, and the fork's copy-on-write spike OOM-killed the
  container (`OOMKilled=true`, exit 137) inside Docker Desktop's ~3.8GB VM. **The failure does
  not look like OOM** — before the kill, memory pressure presents as whole-run *hangs* (frozen
  heartbeat, no answers), so you chase phantom deadlocks in your own code. Always check
  `docker inspect -f '{{.State.OOMKilled}}'` FIRST when a run freezes.
  **Fix: prune per-question graphs once their answers are recorded** — this is step one and by
  far the biggest lever: deleting 452 answered graphs took Redis 1.51G → 71M and the container
  2G → 661M. Do NOT reach for `CONFIG SET save ""`: it removes the fork spike but also stops
  persistence, so the next kill loses every episode since the last save (cost us ~2h of ingest).
  Prune first, keep saves ON, and give the container an explicit memory limit. Graphs DO survive
  the kill via RDB reload — restart the container, never rebuild it (there is no volume mount, so
  `docker rm` loses everything).
- **Mac sleep kills runs**: `caffeinate -i` stops idle-sleep only — NOT lid-close (clamshell) and
  not battery maintenance sleep. Keep it on AC with the lid open. Sleep = dead sockets =
  cascades of timeouts, partial ingests. Ingest resumes per-episode, so just re-run.
- **Never let a stall auto-restart on answer count alone.** A healthy run goes quiet for 30+ min
  (ingest + a 75s×12 retry backoff). Supervising on answers killed healthy runs and *livelocked*:
  nothing could finish inside the window, so it restarted forever with zero progress. Use the
  harness heartbeat (`*_heartbeat` mtime, written on every unit of work) and restart only when
  BOTH it and the answer count are frozen. Kill the process GROUP (`set -m` + `kill -- -$pid`) —
  killing the `caffeinate → uv → python` wrapper orphans the real worker, and two harnesses then
  race on the same graphs.
- **FIXED in core (74fa00b): `FalkorDriver` used to leave `socket_timeout=None`** → a read blocked
  FOREVER on a dead socket, and the awaiting task could not even be cancelled (`asyncio.wait_for`
  issues the cancel, then waits on that same socket). It now defaults to socket_timeout=300s,
  connect=10s, keepalive, health_check_interval=30 — all constructor args, `socket_timeout=None`
  restores the old behaviour, and a caller-supplied `falkor_db` instance is left untouched. This
  was a production-path bug, not just an eval one. If you write another driver, check this.
- **When a long run freezes, split ingest from QA** (`INGEST_ONLY=1` then `SKIP_INGEST=1`) and run
  ONE question in isolation (`QID_FILTER`) before believing the questions are at fault — the 49
  "stuck" temporal questions each answered correctly in ~2 min once run alone.
- **OpenAI org limits**: gpt-4.1 TPM = 30k → ~2-3 reader calls/min at 10k-token contexts;
  concurrency 1–2 + retry (12 tries, 75s cap — must outlast a full TPM window). Mini is fine at
  concurrency 3.
- Attribute-extraction on gpt-4.1-mini occasionally emits ~78k-char degenerate JSON → retry/skip
  (harness handles). `resolve_extracted_edge` **wipes edge.attributes** for unmatched relation
  types — never store trust/provenance data in attributes; first-class columns only.
- Two edge-rehydration pop-lists must stay in lockstep: `edges.py:get_entity_edge_from_record` and
  `driver/record_parsers.py` (plus `bulk_utils.py` builds its own edge_data — 3 write paths!).

## 5. Open ends (ranked next work)

1. **Full-500 LongMemEval + full LoCoMo** — say-go-and-fire; harnesses ready. Incremental
   result writing DONE (2026-07-29): both harnesses append per-answer `.jsonl` and resume on
   restart (errored answers re-run; arm-tagged; QID_FILTER bypasses resume). Only blocker:
   credits.
2. **App integration** — nothing consumes `/get-context`, `/get-ambient-context`,
   `/ambient-outcome` yet. The Phase-6 learning loop needs real outcomes. Includes cross-app
   scope re-filter + behavioral preference inference (needs app signals: reply latency, etc.).
3. **Usefulness boost** — OFF by design until outcome data accrues (clamp [0.3, 1.5],
   reactive-only; injection-recency is a SUPPRESSION signal proactively).
4. **Salience calibration** — floors (block 0.40 / fact 0.22) calibrated on one demo corpus with
   text-embedding-3-small; retune per embedding model. Named-criteria factorization
   (urgency/commitment/actionable/novelty) designed, not built.
5. **Self-entity binding** — deterministic uuid (`self_uuid_for_group`) wired as exclusion key;
   ingest binding + extraction carve-out (first-person pronoun ban conflicts) NOT done.
6. **Verbatim-quote provenance** (supporting_quote → char offsets) — schema fields exist on
   Citation, extraction not wired.
7. **Identity/who's-who differentiator** — EmoryNLP Friends coref test (same-name collisions);
   the moat competitors can't measure. See memory `generalization-dataset-choice`.
8. **Known extraction tails**: relative-clause counting items still occasionally dropped;
   node summaries regenerate only via consolidation (run it on a cron).

## 6. Likely failure points (pre-registered)

- Temporal at scale (133 q) — widest question variety, thinnest verified sample (n=10).
- Multi-session counting semantics ambiguity ("pick up OR return" = 2 or 3?).
- SUPERSEDED depends on graphiti invalidation firing — inconsistent for implicit updates.
- Salience floors on a different embedding model or non-self-centric graph.
- FalkorDB-only quirks (NUL-strip, ISO-string dates) vs the Neo4j production path — the eval
  substrate ≠ prod substrate.

## 7. Session memory

Claude sessions also persist notes at `~/.claude/projects/-Users-mac-graphiti-local/memory/`
(`kyra-memory-layer.md`, `benchmark-findings.md`, `memory-systems-learnings.md`,
`generalization-dataset-choice.md`) — richer narrative + the Mem0/Letta/HydraDB adopt/skip list.
This file is the canonical repo-resident summary.
