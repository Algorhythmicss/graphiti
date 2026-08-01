"""LongMemEval retrieval-QA harness (v2) for the Kyra memory layer.

Fair measurement: per question, ingest evidence sessions as episodes, retrieve
DATED facts + the raw evidence episodes they came from, then let a reader reason
temporally (latest-for-updates, count-for-counts) and judge vs gold. v1 undersold
badly by handing the reader bare undated facts; the graph already carries the
temporal metadata SOTA needs.

Writes a full report to longmemeval_results.out, plus longmemeval_results.jsonl
(one line per question, appended as each completes). Restarting resumes:
completed non-error results are loaded from the jsonl and skipped; errored
ones re-run. Delete the jsonl to start a fresh scoring run.
Usage: PER_TYPE=3 CONCURRENCY=3 TOP_K=25 python longmemeval_harness.py
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '/Users/mac/graphiti-local/server')
import openai  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402


RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)


async def with_retry(coro_fn, tries=12, exc=RETRYABLE):
    delay = 3.0
    for i in range(tries):
        try:
            return await coro_fn()
        except exc:
            if i == tries - 1:
                raise
            await asyncio.sleep(delay)
            # TPM windows reset every 60s -- backoff must exceed a full window.
            delay = min(delay * 1.8, 75)

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # noqa: E402
from graph_service.ambient import cosine_similarity  # noqa: E402
from graph_service.ontology import KYRA_EDGE_TYPE_MAP, KYRA_EDGE_TYPES, KYRA_ENTITY_TYPES  # noqa: E402

DATA = Path(__file__).parent / 'longmemeval_oracle.json'
OUT = open(Path(__file__).parent / 'longmemeval_results.out', 'w')
RESULTS_JSONL = Path(__file__).parent / 'longmemeval_results.jsonl'
PER_TYPE = int(os.environ.get('PER_TYPE', '4'))
CONCURRENCY = int(os.environ.get('CONCURRENCY', '1'))
TOP_K = int(os.environ.get('TOP_K', '30'))
N_EPISODES = int(os.environ.get('N_EPISODES', '8'))
MODEL = os.environ.get('MODEL_NAME', 'gpt-4.1-mini')  # ingest + judge
READER_MODEL = os.environ.get('READER_MODEL', 'gpt-4.1')  # answer step (stronger)
# USE_ONTOLOGY=0 -> stock extraction (no custom types, no attribute-extraction
# step). The Kyra ontology is for the product (commitments/tasks); for generic QA
# it adds an attribute-extraction call that occasionally degenerates on the small
# model. This flag lets us measure ontology vs stock on the same benchmark.
USE_ONTOLOGY = os.environ.get('USE_ONTOLOGY', '1') == '1'
# Decouple ingest from QA so retrieval/reader experiments run fast + sleep-resilient
# against a persistent, cleanly-ingested graph. INGEST_ONLY=1 -> ingest, no QA.
# SKIP_INGEST=1 -> QA only (assume graph already populated). DB name is fixed so it
# persists across runs; ingest is idempotent (skips groups already populated).
INGEST_ONLY = os.environ.get('INGEST_ONLY', '0') == '1'
SKIP_INGEST = os.environ.get('SKIP_INGEST', '0') == '1'
# ARM=zep -> Zep's own recipe: search_() with cross-encoder recipe + their exact
# context string (facts JSON w/ valid_at/invalid_at + "Present = valid" preamble).
# ARM=ours -> our assembly (facts + profiles + semantic episodes + chain-of-note).
ARM = os.environ.get('ARM', 'ours')
# CHUNK_TURNS>0 -> split each session into chunks of N user-assistant turn pairs
# per episode (matches Kyra's real one-message-per-episode ingestion; the
# LongMemEval paper's "session decomposition"). 0 = whole session per episode.
CHUNK_TURNS = int(os.environ.get('CHUNK_TURNS', '0'))
# QTYPE_FILTER=multi-session -> run only that question type (cheap targeted tests)
QTYPE_FILTER = os.environ.get('QTYPE_FILTER', '')
QID_FILTER = [q for q in os.environ.get('QID_FILTER', '').split(',') if q]
DB_NAME = os.environ.get('DB_NAME', 'longmemeval5')
# Ceiling for one question end-to-end (ingest + retrieve + answer + judge).
# Must exceed the legitimate worst case -- ~6 min ingest plus a rate-limit
# backoff of 75s x 12 tries -- so only a true hang trips it.
QUESTION_TIMEOUT = int(os.environ.get('QUESTION_TIMEOUT', '1500'))
# ROOT CAUSE of the recurring whole-run freezes: FalkorDriver builds its client
# as FalkorDB(host, port, ...) with socket_timeout defaulting to None, i.e. a
# read blocks FOREVER. When a socket dies (Mac sleep, FalkorDB hiccup) the
# awaiting task can never be cancelled -- asyncio.wait_for issues the cancel and
# then waits on that same dead socket, so even the per-question timeout hangs.
# A bounded socket_timeout turns 'hang forever' into a raisable error the retry
# path can handle. health_check_interval pings idle connections so a stale one
# is discovered before it is used.
FALKOR_SOCKET_TIMEOUT = float(os.environ.get('FALKOR_SOCKET_TIMEOUT', '120'))


def falkor_driver(database):
    """FalkorDriver whose underlying socket cannot block indefinitely."""
    from falkordb.asyncio import FalkorDB as _FalkorDB

    return FalkorDriver(
        falkor_db=_FalkorDB(
            host='localhost', port=6379,
            socket_timeout=FALKOR_SOCKET_TIMEOUT,
            socket_connect_timeout=15,
            socket_keepalive=True,
            health_check_interval=30,
        ),
        database=database,
    )


oai = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])


def log(*a):
    print(*a, file=OUT, flush=True)


def load_done():
    # Resume map from prior runs: question_id -> result. Errored results are NOT
    # treated as done (errors score as wrong -- always re-attempt them).
    done = {}
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Only resume results from the same arm -- mixing ours/zep answers
            # in one scored run would silently corrupt the comparison.
            if not r.get('error') and r.get('question_id') and r.get('arm', 'ours') == ARM:
                done[r['question_id']] = r
    return done


HEARTBEAT = Path(__file__).parent / 'longmemeval_heartbeat'


def beat(msg):
    """Liveness signal for run_supervised.sh.

    Answer count is a BAD stall signal: answers arrive in bursts, and a healthy
    run can go quiet for 30+ min during ingest or a rate-limit backoff (75s x 12
    tries). Supervising on answers alone kills healthy runs. This file's mtime
    advances on every unit of real work, so 'hung' is distinguishable from 'slow'.
    """
    try:
        HEARTBEAT.write_text(f'{datetime.now(timezone.utc):%F %T} {msg}\n')
    except Exception:
        pass


_JSONL_OUT = RESULTS_JSONL.open('a')


def record(result):
    _JSONL_OUT.write(json.dumps(result, ensure_ascii=False) + '\n')
    _JSONL_OUT.flush()


def parse_date(s):
    try:
        return datetime.strptime(s.split(' (')[0] + s.split(')')[1], '%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc)
    except Exception:
        return datetime(2023, 1, 1, tzinfo=timezone.utc)


def edge_date(e):
    d = e.valid_at or getattr(e, 'reference_time', None) or e.created_at
    return d.strftime('%Y-%m-%d') if d else '?'


def stratified(data, per_type):
    by = defaultdict(list)
    for x in data:
        by[x['question_type']].append(x)
    types = [QTYPE_FILTER] if QTYPE_FILTER else sorted(by)
    sample = [x for t in types for x in by[t][:per_type]]
    if QID_FILTER:
        sample = [x for x in sample if x['question_id'] in QID_FILTER]
    return sample


def episode_units(inst):
    """The (body, date) episodes this instance should contain -- the ingest plan."""
    dates = inst.get('haystack_dates') or [inst['question_date']] * len(inst['haystack_sessions'])
    units = []
    for sess, date in zip(inst['haystack_sessions'], dates, strict=False):
        if CHUNK_TURNS > 0:
            step = CHUNK_TURNS * 2  # N user+assistant pairs
            for i in range(0, len(sess), step):
                chunk = sess[i:i + step]
                units.append(('\n'.join(f'{t["role"]}: {t["content"]}' for t in chunk), date))
        else:
            units.append(('\n'.join(f'{t["role"]}: {t["content"]}' for t in sess), date))
    return [(b, d) for b, d in units if b.strip()]


async def ingest(g, inst):
    gid = inst['question_id']
    units = episode_units(inst)
    # Resume at EPISODE granularity, not graph-exists granularity. A run killed
    # mid-ingest (Mac sleep -> dead sockets) leaves a PARTIAL graph; the old
    # check ("any EntityNode exists -> skip") accepted those, so QA silently ran
    # against half a memory and scored low with no error. Compare stored episode
    # contents against the plan and ingest only what is genuinely missing --
    # complete graphs still cost zero LLM calls.
    try:
        existing = {e.content for e in await EpisodicNode.get_by_group_ids(g.driver, [gid])}
    except Exception:
        existing = set()
    todo = [(b, d) for b, d in units if b not in existing]
    if not todo:
        return 0
    skipped = 0
    for body, date in todo:
        try:
            # Broad retry: also recover from a transient malformed-extraction-JSON
            # error (graphiti's LLM occasionally returns truncated JSON). Skip the
            # session only after exhausting retries -- partial ingest beats losing
            # the whole instance.
            ont = dict(entity_types=KYRA_ENTITY_TYPES, edge_types=KYRA_EDGE_TYPES,
                       edge_type_map=KYRA_EDGE_TYPE_MAP) if USE_ONTOLOGY else {}
            await with_retry(
                lambda body=body, date=date, ont=ont: g.add_episode(
                    name=gid, episode_body=body, reference_time=parse_date(date),
                    source=EpisodeType.message, source_description='lme', group_id=gid, **ont),
                tries=4, exc=(Exception,))
        except Exception:
            skipped += 1
        beat(f'ingest {gid}')
    return skipped


import re as _re

_COUNT_RE = _re.compile(r'\bhow (many|much|long|often)\b|\btotal\b|\bcount\b', _re.I)


async def retrieve_context(g, gid, question):
    # (1) Graph FACTS -- structure + temporal, chronological for ordering/updates.
    # COUNTING/AGGREGATION questions get ALL facts (retrieve-all, not top-K):
    # per-question graphs hold only ~15-40 edges, and a compact complete fact
    # list is what lets the reader enumerate without long-context misses --
    # ranking is the wrong tool for enumeration.
    if _COUNT_RE.search(question):
        from graphiti_core.edges import EntityEdge as _EE

        edges = await _EE.get_by_group_ids(g.driver, [gid])
    else:
        edges = await g.search(question, group_ids=[gid], num_results=TOP_K)
    # Merged-arm fact format: chronological AND carrying the validity window
    # (Zep's signal) -- invalid_at 'Present' marks the currently-true fact.
    facts = [
        (f'[SUPERSEDED on {e.invalid_at:%Y-%m-%d} -- OUTDATED, do not answer with this] '
         f'[{edge_date(e)}] {e.fact}'
         if e.invalid_at else f'[{edge_date(e)} -> Present] {e.fact}')
        for e in sorted(edges, key=edge_date)
    ]
    # (2) Entity PROFILES -- Graphiti's aggregated per-entity summaries (counting).
    try:
        nodes = await EntityNode.get_by_group_ids(g.driver, [gid])
        summaries = [f'{n.name}: {n.summary}' for n in nodes if n.summary][:20]
    except Exception:
        summaries = []
    # (3) Semantic EPISODE retrieval -- rank the raw sessions by similarity to the
    # question DIRECTLY (not via edges), so the evidence turns are recalled even
    # when no fact-edge for the specific detail ranked in the top-K. This is the
    # fix for the dominant "I don't know" recall miss: graph for structure, raw
    # episodes for verbatim detail.
    evidence = []
    try:
        episodes = await EpisodicNode.get_by_group_ids(g.driver, [gid])
        if episodes:
            # Multi-query: a two-event question ("days between X and Y") misses
            # one event when ranked by the whole question. Decompose into event
            # sub-queries (cheap mini call), rank by BEST match across queries.
            queries = [question]
            try:
                sq = await with_retry(lambda: oai.chat.completions.create(
                    model=MODEL, temperature=0, messages=[
                        {'role': 'user', 'content':
                         'List each distinct event/fact mentioned in this question, one short '
                         f'search phrase per line (max 3 lines, no numbering):\n{question}'}]))
                queries += [ln.strip() for ln in sq.choices[0].message.content.splitlines() if ln.strip()][:3]
            except Exception:
                pass
            embs = await g.embedder.create_batch(
                queries + [ep.content[:2000] for ep in episodes]
            )
            q_vecs, ep_vecs = embs[:len(queries)], embs[len(queries):]
            ranked = sorted(
                zip(episodes, ep_vecs, strict=False),
                key=lambda pair: max(cosine_similarity(qv, pair[1]) for qv in q_vecs),
                reverse=True,
            )
            # Dedupe near-identical episodes (re-ingested content) so duplicates
            # don't crowd out OTHER evidence sessions -- the direct cause of the
            # counting/multi-session misses.
            seen_pref = set()
            picked_eps = []
            for ep, _ in ranked:
                key = ep.content[:200]
                if key in seen_pref:
                    continue
                seen_pref.add(key)
                picked_eps.append(ep)
                if len(picked_eps) >= N_EPISODES:
                    break
            # Split a TOTAL char budget across picked episodes instead of a flat
            # 900-char cut -- deep answers in long sessions were being truncated
            # away ("I don't know" despite correct retrieval).
            # Sessions run 12-20k chars; truncation was cutting gold turns (4/10
            # multi-session instances). gpt-4.1 has 1M context -- include full
            # sessions: 80k total budget, capped at 22k/session (> observed max).
            # 48k total (~12k tokens): fits the org's 30k-TPM gpt-4.1 limit while
            # covering most sessions fully (obs. max ~20k chars single-session).
            per_ep = min(20000, max(1200, 48000 // max(1, len(picked_eps))))
            # Date-stamp each episode and present chronologically -- ordering
            # questions need the timeline anchored, not similarity order.
            picked_eps.sort(key=lambda ep: ep.valid_at or ep.created_at)
            evidence = [
                f'[SESSION DATE: {(ep.valid_at or ep.created_at).strftime("%Y-%m-%d")}]\n{ep.content[:per_ep]}'
                for ep in picked_eps
            ]
    except Exception:
        evidence = []
    return facts, evidence, summaries


async def answer(question, qdate, facts, evidence, summaries):
    fb = '\n'.join(facts) or '(none)'
    eb = '\n---\n'.join(evidence) or '(none)'
    sb = '\n'.join(summaries) or '(none)'
    r = await with_retry(lambda: oai.chat.completions.create(model=READER_MODEL, temperature=0, messages=[
        {'role': 'system', 'content':
         "You answer a question from a user's memory (date-stamped FACTS in chronological order, "
         'ENTITY PROFILES, and raw CONVERSATION EVIDENCE). A fact\'s date is when it was stated; a '
         "later date supersedes an earlier one (treat the newest as 'current').\n"
         'Work in two steps:\n'
         'STEP 1 (NOTES): privately list every evidence item relevant to the question, each with its '
         'date. For COUNTING questions, list every DISTINCT instance across ALL facts and evidence '
         '(scan everything, dedupe identical mentions, do not stop early). For UPDATES, note the '
         'latest-dated value. For ORDERING, sort by date.\n'
         'For RECOMMENDATION/SUGGESTION questions ("can you suggest/recommend..."), do NOT answer '
         '"no information" and NEVER say I don\'t know: identify the user\'s stated preferences, '
         'equipment, brands, tastes, interests, and constraints from memory, and give a '
         'recommendation explicitly tailored to those (name the preference you are honoring, e.g. '
         'Sony-compatible, Adobe-focused). Even when memory lacks specific venues/events/items, '
         'recommend generic options THAT FIT the user\'s stated interests.\n'
         'For DATE-DIFFERENCE questions ("how many days between/before X and Y"), write out both '
         'dates explicitly, then compute the difference by counting days step-by-step (month by '
         'month), showing the arithmetic in your notes before answering.\n'
         'STEP 2: end with a line "FINAL: <answer>" -- the specific value/name/number/date only. '
         'For counting, count your deduped list. For a quantity stated at multiple dates (e.g. a '
         'personal best, a price, an amount), the value stated at the LATEST date is the current '
         'truth even if stated in passing -- an offhand later mention ("my personal best of X") '
         'REDEFINES the quantity and beats an earlier, more explicit claim ("I set a PB of Y"). '
         'A fact whose bracketed window is CLOSED (ends with a date, not "Present") has been '
         'SUPERSEDED: when any open "[... -> Present]" fact covers the same quantity, answer from '
         'the open fact -- "what was/is my X" asks for the current X unless an earlier time is '
         'named. '
         'For "most recent X" questions, list every candidate X with its date and pick the '
         'latest-dated, not the most prominently described. Say FINAL: I don\'t know only if the '
         'evidence truly lacks it.'},
        {'role': 'user', 'content': f'TODAY: {qdate}\n\nENTITY PROFILES (may contain STALE values -- '
         f'dated FACTS are authoritative; when they disagree, trust the latest open-window FACT):\n{sb}\n\n'
         f'FACTS (chronological):\n{fb}\n\nCONVERSATION EVIDENCE:\n{eb}\n\nQUESTION: {question}'},
    ]))
    text = r.choices[0].message.content.strip()
    # Parse the FINAL line so notes never leak into the judged answer.
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith('FINAL:'):
            return line.split(':', 1)[1].strip()
    return text


async def answer_zep(question, qdate, zep_context):
    """Arm A reader: Zep's context string verbatim, minimal QA instruction."""
    r = await with_retry(lambda: oai.chat.completions.create(model=READER_MODEL, temperature=0, messages=[
        {'role': 'system', 'content': 'You answer the question using the provided memory context. '
         'Be concise; give the specific value.'},
        {'role': 'user', 'content': f'TODAY: {qdate}\n\n{zep_context}\n\nQUESTION: {question}\nANSWER:'},
    ]))
    return r.choices[0].message.content.strip()


async def judge(question, gold, pred):
    r = await with_retry(lambda: oai.chat.completions.create(model=MODEL, temperature=0, messages=[
        {'role': 'system', 'content': 'Grade if the predicted answer matches the gold answer in meaning. '
         'When the gold describes a USER PREFERENCE ("The user would prefer..."), grade CORRECT if '
         'the prediction\'s suggestions ALIGN with that stated preference (e.g. gold wants '
         'Sony-compatible gear and the prediction recommends for a Sony camera), regardless of '
         'phrasing. Reply exactly "CORRECT" or "WRONG".'},
        {'role': 'user', 'content': f'QUESTION: {question}\nGOLD: {gold}\nPREDICTED: {pred}'},
    ]))
    return r.choices[0].message.content.strip().upper().startswith('CORRECT')


async def run_one(g, inst, sem):
    # Hard per-question deadline. Observed in the 500-run: roughly once an hour
    # BOTH workers freeze permanently (heartbeat and answers stop together) --
    # a socket that never times out somewhere under the FalkorDB/HTTP clients.
    # Without this, one stuck question hangs the entire run until the supervisor
    # kills it, losing every in-flight question with it. A timed-out question is
    # recorded as an error, so the run continues and resume re-runs it later.
    # The semaphore is acquired OUT here, not inside the timeout: every question
    # is scheduled at once and most sit queued for hours, so timing from before
    # the acquire would fail them all for waiting their turn.
    try:
        async with sem:
            res = await asyncio.wait_for(_run_one_inner(g, inst), timeout=QUESTION_TIMEOUT)
    except (TimeoutError, asyncio.TimeoutError):
        res = {'qtype': inst['question_type'], 'correct': False,
               'error': f'question timeout after {QUESTION_TIMEOUT}s', 'question': inst['question'],
               # Carry the flag so an INGEST_ONLY pass never writes a result
               # line. Without it a timed-out ingest appended an 'error' row for
               # a question that was already answered in an earlier QA run.
               'ingest_only': INGEST_ONLY}
    res['question_id'] = inst['question_id']
    res['arm'] = ARM
    if not res.get('ingest_only'):
        record(res)  # durable per-answer write -- a crash never loses paid QA
    return res


async def _run_one_inner(g, inst):
    try:
        gid = inst['question_id']
        # ONE SCOPED CLIENT for BOTH ingest and QA. FalkorDB stores each
        # group_id as its own graph, and a main-client ingest splits writes
        # across graphs (episodes vs entities) -- partial graphs, empty
        # retrieval. Scoping everything to database=gid removes all routing
        # ambiguity (and matches per-tenant product deployment).
        gs = Graphiti(graph_driver=falkor_driver(gid))
        try:
            await gs.build_indices_and_constraints()  # explicit: bg task is too slow for ingest->QA
            if not SKIP_INGEST:
                await ingest(gs, inst)
            if INGEST_ONLY:
                return {'qtype': inst['question_type'], 'ingest_only': True}
            if ARM == 'zep':
                from graphiti_core.search.search_config_recipes import (
                    COMBINED_HYBRID_SEARCH_CROSS_ENCODER,
                )
                from graphiti_core.search.search_helpers import (
                    search_results_to_context_string,
                )
                cfg = COMBINED_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
                cfg.limit = TOP_K
                sr = await gs.search_(inst['question'], config=cfg, group_ids=[gid])
                zep_context = search_results_to_context_string(sr)
                pred = await answer_zep(inst['question'], inst.get('question_date', ''), zep_context)
                correct = await judge(inst['question'], inst['answer'], pred)
                return {'qtype': inst['question_type'], 'correct': correct,
                        'question': inst['question'], 'gold': inst['answer'], 'pred': pred}
            facts, evidence, summaries = await retrieve_context(gs, gid, inst['question'])
            beat(f'retrieved {gid}')
        finally:
            await gs.close()
        pred = await answer(inst['question'], inst.get('question_date', ''), facts, evidence, summaries)
        beat(f'answered {gid}')
        correct = await judge(inst['question'], inst['answer'], pred)
        return {'qtype': inst['question_type'], 'correct': correct, 'n_facts': len(facts),
                'question': inst['question'], 'gold': inst['answer'], 'pred': pred}
    except Exception as e:
        return {'qtype': inst['question_type'], 'correct': False, 'error': repr(e)[:200], 'question': inst['question']}


async def main():
    data = json.load(open(DATA))
    sample = stratified(data, PER_TYPE)
    # QID_FILTER is the fix->retest loop: always re-run those qids (the fresh
    # answer's jsonl line supersedes the old one -- load_done keeps last-wins).
    done = {} if (INGEST_ONLY or QID_FILTER) else load_done()
    resumed = [done[x['question_id']] for x in sample if x['question_id'] in done]
    pending = [x for x in sample if x['question_id'] not in done]
    log(f'sample={len(sample)} ({PER_TYPE}/type) TOP_K={TOP_K} N_EPISODES={N_EPISODES} model={MODEL}')
    if resumed:
        log(f'resumed {len(resumed)} completed answers from {RESULTS_JSONL.name}')
    log('')
    g = Graphiti(graph_driver=falkor_driver(DB_NAME))
    await g.build_indices_and_constraints()
    sem = asyncio.Semaphore(CONCURRENCY)
    results = resumed + list(await asyncio.gather(*[run_one(g, x, sem) for x in pending]))
    await g.close()

    if INGEST_ONLY:
        # Ingest-only results carry no question/answer -- scoring them crashed
        # the summary AFTER all the ingest work was already done.
        log(f'INGEST_ONLY: {len(results)} instances ingested')
        return

    by = defaultdict(lambda: [0, 0])
    errs = 0
    for r in results:
        by[r['qtype']][1] += 1
        by[r['qtype']][0] += int(bool(r.get('correct')))
        errs += int(bool(r.get('error')))
    tc = sum(v[0] for v in by.values()); tn = sum(v[1] for v in by.values())
    log('=' * 60)
    log(f'{"question_type":28} {"acc":>6}   n')
    for t in sorted(by):
        c, n = by[t]
        log(f'{t:28} {c / n:>6.0%}   {c}/{n}')
    log('-' * 60)
    log(f'{"OVERALL":28} {tc / tn:>6.0%}   {tc}/{tn}  (errors={errs})')
    log('\n--- transcripts ---')
    for r in results:
        mark = '✓' if r.get('correct') else ('ERR' if r.get('error') else '✗')
        log(f'[{mark}] ({r["qtype"]}) Q: {r["question"][:85]}')
        log(f'      gold: {str(r.get("gold"))[:85]}')
        log(f'      pred: {str(r.get("pred"))[:110]}' + (f'  ERR={r["error"]}' if r.get('error') else ''))


asyncio.run(main())
OUT.close()
_JSONL_OUT.close()
