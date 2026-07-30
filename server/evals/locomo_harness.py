"""LoCoMo harness v2 -- full LongMemEval-proven fix stack.

Fixes ported: scoped per-conversation clients + explicit index build (FalkorDB
multi-graph), chunked turn-pair ingestion, SUPERSEDED-tagged facts, authority-
ordered context, retrieve-all for counting, split evidence budget with session
dates, chain-of-note reader with FINAL parsing, 12-try/75s retry.

Usage: N_CONV=2 QA_PER_CONV=25 CONCURRENCY=2 python locomo_harness.py
Writes locomo_results.out (report) + locomo_results.jsonl (one line per QA,
appended as each completes). Restarting resumes: completed non-error results
are loaded from the jsonl and skipped; errored ones re-run. Delete the jsonl
to start a fresh scoring run.
"""

import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, '/Users/mac/graphiti-local/server')
import openai  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.edges import EntityEdge  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # noqa: E402
from graph_service.ambient import cosine_similarity  # noqa: E402

DATA = Path(__file__).parent / 'locomo' / 'data' / 'locomo10.json'
OUT = open(Path(__file__).parent / 'locomo_results.out', 'w')
RESULTS_JSONL = Path(__file__).parent / 'locomo_results.jsonl'
N_CONV = int(os.environ.get('N_CONV', '2'))
QA_PER_CONV = int(os.environ.get('QA_PER_CONV', '25'))
CONCURRENCY = int(os.environ.get('CONCURRENCY', '2'))
TOP_K = int(os.environ.get('TOP_K', '40'))
N_EPISODES = int(os.environ.get('N_EPISODES', '10'))
CHUNK_TURNS = int(os.environ.get('CHUNK_TURNS', '2'))
MODEL = os.environ.get('MODEL_NAME', 'gpt-4.1-mini')
READER_MODEL = os.environ.get('READER_MODEL', 'gpt-4.1')
CATS = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}
COUNT_RE = re.compile(r'\bhow (many|much|long|often)\b|\btotal\b|\bcount\b', re.I)
oai = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)


def log(*a):
    print(*a, file=OUT, flush=True)


def load_done():
    # Resume map from prior runs: (gid, question) -> result. Errored results are
    # NOT treated as done (errors score as wrong -- always re-attempt them).
    done = {}
    if RESULTS_JSONL.exists():
        for line in RESULTS_JSONL.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get('error'):
                done[(r.get('gid'), r.get('q'))] = r
    return done


_JSONL_OUT = RESULTS_JSONL.open('a')


def record(result):
    _JSONL_OUT.write(json.dumps(result, ensure_ascii=False) + '\n')
    _JSONL_OUT.flush()


async def with_retry(fn, tries=12, exc=RETRYABLE):
    delay = 3.0
    for i in range(tries):
        try:
            return await fn()
        except exc:
            if i == tries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 1.8, 75)


def parse_date(s):
    for fmt in ('%I:%M %p on %d %B, %Y', '%I:%M %p on %d %b, %Y'):
        try:
            return datetime.strptime(s.strip(), fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime(2023, 1, 1, tzinfo=timezone.utc)


def edge_date(e):
    d = e.valid_at or getattr(e, 'reference_time', None) or e.created_at
    return d.strftime('%Y-%m-%d') if d else '?'


def fmt_fact(e):
    if e.invalid_at:
        return (f'[SUPERSEDED on {e.invalid_at:%Y-%m-%d} -- OUTDATED, do not answer with this] '
                f'[{edge_date(e)}] {e.fact}')
    return f'[{edge_date(e)} -> Present] {e.fact}'


def sample_qa(qa, per_conv):
    by = defaultdict(list)
    for q in qa:
        by[q.get('category')].append(q)
    out, idx = [], 0
    while len(out) < per_conv and any(idx < len(v) for v in by.values()):
        for cat in sorted(by):
            if idx < len(by[cat]) and len(out) < per_conv:
                out.append(by[cat][idx])
        idx += 1
    return out


async def ingest_conversation(gs, conv, gid):
    sess_keys = sorted(
        [k for k in conv if k.startswith('session') and not k.endswith('date_time')],
        key=lambda k: int(k.split('_')[1]),
    )
    units = []
    for k in sess_keys:
        turns, date = conv[k], conv.get(f'{k}_date_time', '')
        step = max(1, CHUNK_TURNS * 2)
        for i in range(0, len(turns), step):
            body = '\n'.join(f'{t["speaker"]}: {t["text"]}' for t in turns[i:i + step])
            if body.strip():
                units.append((body, date))
    # Resume at EPISODE granularity, not graph-exists granularity: a run killed
    # mid-ingest (Mac sleep -> dead sockets) leaves a PARTIAL graph, and a
    # "graph is non-empty -> skip" check accepts it, so QA silently runs against
    # half a memory. Ingest only genuinely-missing episodes; a complete
    # conversation still costs zero LLM calls.
    try:
        existing = {e.content for e in await EpisodicNode.get_by_group_ids(gs.driver, [gid])}
    except Exception:
        existing = set()
    for body, date in units:
        if body in existing:
            continue
        try:
            await with_retry(lambda b=body, d=date: gs.add_episode(
                name=gid, episode_body=b, reference_time=parse_date(d),
                source=EpisodeType.message, source_description='locomo', group_id=gid),
                tries=4, exc=(Exception,))
        except Exception:
            pass


_EP_CACHE: dict = {}


async def retrieve(gs, gid, question):
    if COUNT_RE.search(question):
        edges = await EntityEdge.get_by_group_ids(gs.driver, [gid])
    else:
        edges = await gs.search(question, group_ids=[gid], num_results=TOP_K)
    facts = [fmt_fact(e) for e in sorted(edges, key=edge_date)]
    try:
        nodes = await EntityNode.get_by_group_ids(gs.driver, [gid])
        profiles = [f'{n.name}: {n.summary}' for n in nodes if n.summary][:25]
    except Exception:
        profiles = []
    ev = []
    try:
        if gid not in _EP_CACHE:
            eps = await EpisodicNode.get_by_group_ids(gs.driver, [gid])
            vecs = await gs.embedder.create_batch([e.content[:2000] for e in eps]) if eps else []
            _EP_CACHE[gid] = (eps, vecs)
        eps, vecs = _EP_CACHE[gid]
        if eps:
            qv = (await gs.embedder.create_batch([question]))[0]
            ranked = sorted(zip(eps, vecs), key=lambda p: cosine_similarity(qv, p[1]), reverse=True)
            seen, picked = set(), []
            for ep, _ in ranked:
                key = ep.content[:200]
                if key in seen:
                    continue
                seen.add(key)
                picked.append(ep)
                if len(picked) >= N_EPISODES:
                    break
            per_ep = min(20000, max(1200, 48000 // max(1, len(picked))))
            picked.sort(key=lambda ep: ep.valid_at or ep.created_at)
            ev = [f'[SESSION DATE: {(ep.valid_at or ep.created_at):%Y-%m-%d}]\n{ep.content[:per_ep]}'
                  for ep in picked]
    except Exception:
        ev = []
    return facts, profiles, ev


async def answer(question, facts, profiles, ev):
    fb, pb, eb = '\n'.join(facts) or '(none)', '\n'.join(profiles) or '(none)', '\n---\n'.join(ev) or '(none)'
    r = await with_retry(lambda: oai.chat.completions.create(model=READER_MODEL, temperature=0, messages=[
        {'role': 'system', 'content':
         "You answer a question from two users' shared conversational memory (date-stamped FACTS in "
         'chronological order, ENTITY PROFILES, raw CONVERSATION EVIDENCE). Facts marked SUPERSEDED '
         'are outdated -- never answer from them; a later-dated statement supersedes an earlier one '
         'even if stated in passing.\n'
         'STEP 1 (NOTES): list every relevant evidence item with its date. COUNTING: enumerate every '
         'distinct instance across ALL facts and evidence, dedupe, then count. ORDERING/WHEN: compare '
         'dates; for date differences compute step-by-step. If the memory genuinely lacks the answer '
         '(adversarial questions), the correct answer is that there is no information.\n'
         'STEP 2: end with "FINAL: <answer>" -- the specific value/name/date only, or "FINAL: No '
         'information available" when the memory truly lacks it.'},
        {'role': 'user', 'content': f'ENTITY PROFILES (may contain STALE values -- dated FACTS are '
         f'authoritative):\n{pb}\n\nFACTS (chronological):\n{fb}\n\nCONVERSATION EVIDENCE:\n{eb}\n\n'
         f'QUESTION: {question}'},
    ]))
    text = r.choices[0].message.content.strip()
    for line in reversed(text.splitlines()):
        if line.strip().upper().startswith('FINAL:'):
            return line.split(':', 1)[1].strip()
    return text


async def judge(question, gold, pred):
    r = await with_retry(lambda: oai.chat.completions.create(model=MODEL, temperature=0, messages=[
        {'role': 'system', 'content': 'Grade if the predicted answer matches the gold answer in '
         'meaning (dates may differ in format; for gold like "not mentioned"/"no information", an '
         'abstention is CORRECT). Reply exactly CORRECT or WRONG.'},
        {'role': 'user', 'content': f'QUESTION: {question}\nGOLD: {gold}\nPREDICTED: {pred}'},
    ]))
    return r.choices[0].message.content.strip().upper().startswith('CORRECT')


async def run_qa(gs, gid, q, sem):
    async with sem:
        try:
            facts, profiles, ev = await retrieve(gs, gid, q['question'])
            pred = await answer(q['question'], facts, profiles, ev)
            correct = await judge(q['question'], str(q.get('answer')), pred)
            res = {'gid': gid, 'cat': q.get('category'), 'correct': correct, 'q': q['question'],
                   'gold': str(q.get('answer'))[:90], 'pred': pred[:100]}
        except Exception as e:
            res = {'gid': gid, 'cat': q.get('category'), 'correct': False,
                   'error': repr(e)[:120], 'q': q['question']}
        record(res)  # durable per-answer write -- a crash never loses paid QA
        return res


async def main():
    data = json.load(open(DATA))
    done = load_done()
    log(f'N_CONV={N_CONV} QA_PER_CONV={QA_PER_CONV} CHUNK_TURNS={CHUNK_TURNS} TOP_K={TOP_K} reader={READER_MODEL}')
    if done:
        log(f'resumed {len(done)} completed answers from {RESULTS_JSONL.name}')
    log('')
    results = []
    for ci, item in enumerate(data[:N_CONV]):
        gid = item.get('sample_id', f'conv{ci}')
        qs = sample_qa(item['qa'], QA_PER_CONV)
        resumed = [done[(gid, q['question'])] for q in qs if (gid, q['question']) in done]
        pending = [q for q in qs if (gid, q['question']) not in done]
        results.extend(resumed)
        if not pending:
            log(f'  {gid}: all {len(resumed)} answers resumed, skipping')
            continue
        gs = Graphiti(graph_driver=FalkorDriver(host='localhost', port=6379, database=gid))
        try:
            await gs.build_indices_and_constraints()
            log(f'ingesting {gid} (chunked)...')
            await ingest_conversation(gs, item['conversation'], gid)
            sem = asyncio.Semaphore(CONCURRENCY)
            res = await asyncio.gather(*[run_qa(gs, gid, q, sem) for q in pending])
            results.extend(res)
            log(f'  {gid}: {sum(bool(r.get("correct")) for r in res)}/{len(res)} new'
                + (f' (+{len(resumed)} resumed)' if resumed else ''))
        finally:
            await gs.close()

    by = defaultdict(lambda: [0, 0])
    errs = 0
    for r in results:
        by[r['cat']][1] += 1
        by[r['cat']][0] += int(bool(r.get('correct')))
        errs += int(bool(r.get('error')))
    tc, tn = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    log('\n' + '=' * 60)
    for cat in sorted(by):
        c, n = by[cat]
        log(f'{CATS.get(cat, cat)!s:22} {c / n:>6.0%}   {c}/{n}')
    log('-' * 60)
    log(f'{"OVERALL":22} {tc / tn:>6.0%}   {tc}/{tn}  (errors={errs})')
    for r in results:
        mark = '✓' if r.get('correct') else ('ERR' if r.get('error') else '✗')
        log(f'[{mark}] (cat{r["cat"]}) {r["q"][:70]} | gold: {r.get("gold")} | pred: {r.get("pred")}')


asyncio.run(main())
OUT.close()
_JSONL_OUT.close()
