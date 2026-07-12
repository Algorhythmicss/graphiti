"""LongMemEval retrieval-QA harness (v2) for the Kyra memory layer.

Fair measurement: per question, ingest evidence sessions as episodes, retrieve
DATED facts + the raw evidence episodes they came from, then let a reader reason
temporally (latest-for-updates, count-for-counts) and judge vs gold. v1 undersold
badly by handing the reader bare undated facts; the graph already carries the
temporal metadata SOTA needs.

Writes a full report to longmemeval_results.out.
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


async def with_retry(coro_fn, tries=8, exc=RETRYABLE):
    delay = 3.0
    for i in range(tries):
        try:
            return await coro_fn()
        except exc:
            if i == tries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 1.8, 45)

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # noqa: E402
from graph_service.ambient import cosine_similarity  # noqa: E402
from graph_service.ontology import KYRA_EDGE_TYPE_MAP, KYRA_EDGE_TYPES, KYRA_ENTITY_TYPES  # noqa: E402

DATA = Path(__file__).parent / 'longmemeval_oracle.json'
OUT = open(Path(__file__).parent / 'longmemeval_results.out', 'w')
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
DB_NAME = os.environ.get('DB_NAME', 'longmemeval5')
oai = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])


def log(*a):
    print(*a, file=OUT, flush=True)


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
    return [x for t in sorted(by) for x in by[t][:per_type]]


async def ingest(g, inst):
    gid = inst['question_id']
    try:
        if await EntityNode.get_by_group_ids(g.driver, [gid]):
            return 0  # already ingested -- idempotent resume across runs
    except Exception:
        pass
    dates = inst.get('haystack_dates') or [inst['question_date']] * len(inst['haystack_sessions'])
    skipped = 0
    for sess, date in zip(inst['haystack_sessions'], dates):
        body = '\n'.join(f'{t["role"]}: {t["content"]}' for t in sess)
        if not body.strip():
            continue
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
    return skipped


async def retrieve_context(g, gid, question):
    # (1) Graph FACTS -- structure + temporal, chronological for ordering/updates.
    edges = await g.search(question, group_ids=[gid], num_results=TOP_K)
    facts = [f'[{edge_date(e)}] {e.fact}' for e in sorted(edges, key=edge_date)]
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
            embs = await g.embedder.create_batch(
                [question] + [ep.content[:2000] for ep in episodes]
            )
            qv, ep_vecs = embs[0], embs[1:]
            ranked = sorted(
                zip(episodes, ep_vecs, strict=False),
                key=lambda pair: cosine_similarity(qv, pair[1]),
                reverse=True,
            )
            # Dedupe near-identical episodes (re-ingested content) so duplicates
            # don't crowd out OTHER evidence sessions -- the direct cause of the
            # counting/multi-session misses.
            seen_pref = set()
            for ep, _ in ranked:
                key = ep.content[:200]
                if key in seen_pref:
                    continue
                seen_pref.add(key)
                evidence.append(ep.content[:900])
                if len(evidence) >= N_EPISODES:
                    break
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
         'STEP 2: end with a line "FINAL: <answer>" -- the specific value/name/number/date only. '
         'For counting, count your deduped list. For a quantity stated at multiple dates (e.g. a '
         'personal best, a price, an amount), the value stated at the LATEST date is the current '
         'truth even if stated in passing. Say FINAL: I don\'t know only if the evidence truly '
         'lacks it.'},
        {'role': 'user', 'content': f'TODAY: {qdate}\n\nENTITY PROFILES:\n{sb}\n\nFACTS (chronological):\n{fb}\n\nCONVERSATION EVIDENCE:\n{eb}\n\nQUESTION: {question}'},
    ]))
    text = r.choices[0].message.content.strip()
    # Parse the FINAL line so notes never leak into the judged answer.
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith('FINAL:'):
            return line.split(':', 1)[1].strip()
    return text


async def judge(question, gold, pred):
    r = await with_retry(lambda: oai.chat.completions.create(model=MODEL, temperature=0, messages=[
        {'role': 'system', 'content': 'Grade if the predicted answer matches the gold answer in meaning. '
         'Reply exactly "CORRECT" or "WRONG".'},
        {'role': 'user', 'content': f'QUESTION: {question}\nGOLD: {gold}\nPREDICTED: {pred}'},
    ]))
    return r.choices[0].message.content.strip().upper().startswith('CORRECT')


async def run_one(g, inst, sem):
    async with sem:
        try:
            gid = inst['question_id']
            # ONE SCOPED CLIENT for BOTH ingest and QA. FalkorDB stores each
            # group_id as its own graph, and a main-client ingest splits writes
            # across graphs (episodes vs entities) -- partial graphs, empty
            # retrieval. Scoping everything to database=gid removes all routing
            # ambiguity (and matches per-tenant product deployment).
            gs = Graphiti(graph_driver=FalkorDriver(host='localhost', port=6379, database=gid))
            try:
                await gs.build_indices_and_constraints()  # explicit: bg task is too slow for ingest->QA
                if not SKIP_INGEST:
                    await ingest(gs, inst)
                if INGEST_ONLY:
                    return {'qtype': inst['question_type'], 'ingest_only': True}
                facts, evidence, summaries = await retrieve_context(gs, gid, inst['question'])
            finally:
                await gs.close()
            pred = await answer(inst['question'], inst.get('question_date', ''), facts, evidence, summaries)
            correct = await judge(inst['question'], inst['answer'], pred)
            return {'qtype': inst['question_type'], 'correct': correct, 'n_facts': len(facts),
                    'question': inst['question'], 'gold': inst['answer'], 'pred': pred}
        except Exception as e:
            return {'qtype': inst['question_type'], 'correct': False, 'error': repr(e)[:200], 'question': inst['question']}


async def main():
    data = json.load(open(DATA))
    sample = stratified(data, PER_TYPE)
    log(f'sample={len(sample)} ({PER_TYPE}/type) TOP_K={TOP_K} N_EPISODES={N_EPISODES} model={MODEL}\n')
    g = Graphiti(graph_driver=FalkorDriver(host='localhost', port=6379, database='longmemeval5'))
    await g.build_indices_and_constraints()
    sem = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*[run_one(g, x, sem) for x in sample])
    await g.close()

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
