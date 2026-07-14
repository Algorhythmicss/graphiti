"""LoCoMo retrieval-QA harness for the Kyra memory layer.

LoCoMo is the QA benchmark Mem0/Zep/Letta report -> apples-to-apples. Per
conversation: ingest all sessions as episodes (group_id = sample_id), then for a
stratified sample of its QA, retrieve DATED facts + raw evidence and answer +
judge. Reports accuracy by category (1=multi-hop, 2=temporal, 3=open-domain,
4=single-hop, 5=adversarial/abstain).

Writes to locomo_results.out.
Usage: N_CONV=2 QA_PER_CONV=25 CONCURRENCY=1 python locomo_harness.py
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

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # noqa: E402
from graph_service.ambient import cosine_similarity  # noqa: E402
from graph_service.ontology import KYRA_EDGE_TYPE_MAP, KYRA_EDGE_TYPES, KYRA_ENTITY_TYPES  # noqa: E402

_EP_CACHE: dict = {}  # gid -> (episodes, embeddings), so a conversation's sessions embed once


async def _group_episodes(g, gid):
    if gid not in _EP_CACHE:
        eps = await EpisodicNode.get_by_group_ids(g.driver, [gid])
        embs = await g.embedder.create_batch([e.content[:2000] for e in eps]) if eps else []
        _EP_CACHE[gid] = (eps, embs)
    return _EP_CACHE[gid]

DATA = Path(__file__).parent / 'locomo' / 'data' / 'locomo10.json'
OUT = open(Path(__file__).parent / 'locomo_results.out', 'w')
N_CONV = int(os.environ.get('N_CONV', '2'))
QA_PER_CONV = int(os.environ.get('QA_PER_CONV', '25'))
CONCURRENCY = int(os.environ.get('CONCURRENCY', '1'))
TOP_K = int(os.environ.get('TOP_K', '30'))
N_EPISODES = int(os.environ.get('N_EPISODES', '8'))
MODEL = os.environ.get('MODEL_NAME', 'gpt-4.1-mini')
READER_MODEL = os.environ.get('READER_MODEL', 'gpt-4.1')
CATS = {1: 'multi-hop', 2: 'temporal', 3: 'open-domain', 4: 'single-hop', 5: 'adversarial'}
oai = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])


def log(*a):
    print(*a, file=OUT, flush=True)


RETRYABLE = (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)


async def with_retry(fn, tries=8, exc=RETRYABLE):
    delay = 3.0
    for i in range(tries):
        try:
            return await fn()
        except exc:
            if i == tries - 1:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 1.8, 45)


def parse_date(s):
    # "1:56 pm on 8 May, 2023"
    try:
        return datetime.strptime(s.strip(), '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s.strip(), '%I:%M %p on %d %b, %Y').replace(tzinfo=timezone.utc)
        except Exception:
            return datetime(2023, 1, 1, tzinfo=timezone.utc)


def edge_date(e):
    d = e.valid_at or getattr(e, 'reference_time', None) or e.created_at
    return d.strftime('%Y-%m-%d') if d else '?'


def sample_qa(qa, per_conv):
    by = defaultdict(list)
    for q in qa:
        by[q.get('category')].append(q)
    out = []
    # round-robin across categories up to per_conv
    idx = 0
    while len(out) < per_conv and any(idx < len(v) for v in by.values()):
        for cat in sorted(by):
            if idx < len(by[cat]) and len(out) < per_conv:
                out.append(by[cat][idx])
        idx += 1
    return out


async def ingest_conversation(g, conv, gid):
    sess_keys = sorted(
        [k for k in conv if k.startswith('session') and not k.endswith('date_time')],
        key=lambda k: int(k.split('_')[1]),
    )
    for k in sess_keys:
        turns = conv[k]
        date = conv.get(f'{k}_date_time', '')
        body = '\n'.join(f'{t["speaker"]}: {t["text"]}' for t in turns)
        if not body.strip():
            continue
        try:
            await with_retry(lambda body=body, date=date: g.add_episode(
                name=gid, episode_body=body, reference_time=parse_date(date),
                source=EpisodeType.message, source_description='locomo', group_id=gid,
                entity_types=KYRA_ENTITY_TYPES, edge_types=KYRA_EDGE_TYPES,
                edge_type_map=KYRA_EDGE_TYPE_MAP), tries=4, exc=(Exception,))
        except Exception:
            pass


async def retrieve(g, gid, question):
    # (1) Graph facts (structure + temporal).
    edges = await g.search(question, group_ids=[gid], num_results=TOP_K)
    facts = [f'[{edge_date(e)}] {e.fact}' for e in sorted(edges, key=edge_date)]
    # (2) Entity profiles (aggregation).
    try:
        nodes = await EntityNode.get_by_group_ids(g.driver, [gid])
        summaries = [f'{n.name}: {n.summary}' for n in nodes if n.summary][:25]
    except Exception:
        summaries = []
    # (3) Semantic episode retrieval -- raw sessions ranked by similarity to the
    # question directly, so specific evidence is recalled even when no fact-edge
    # ranked. Episode embeddings cached per conversation.
    ev = []
    try:
        eps, ep_vecs = await _group_episodes(g, gid)
        if eps:
            qv = (await g.embedder.create_batch([question]))[0]
            ranked = sorted(
                zip(eps, ep_vecs, strict=False),
                key=lambda pair: cosine_similarity(qv, pair[1]),
                reverse=True,
            )
            ev = [e.content[:900] for e, _ in ranked[:N_EPISODES]]
    except Exception:
        ev = []
    return facts, ev, summaries


async def answer(question, facts, ev, summaries):
    fb = '\n'.join(facts) or '(none)'
    eb = '\n---\n'.join(ev) or '(none)'
    sb = '\n'.join(summaries) or '(none)'
    r = await with_retry(lambda: oai.chat.completions.create(model=READER_MODEL, temperature=0, messages=[
        {'role': 'system', 'content':
         "Answer the question from the user's memory (ENTITY PROFILES + date-stamped FACTS + raw "
         'EVIDENCE). For "when" use dates; for ordering compare dates; for counting enumerate all '
         'distinct items across profiles and facts; for updates use the latest. If the memory '
         'genuinely does not contain the answer, reply "No information available." Be concise.'},
        {'role': 'user', 'content': f'ENTITY PROFILES:\n{sb}\n\nFACTS:\n{fb}\n\nEVIDENCE:\n{eb}\n\nQUESTION: {question}\nANSWER:'},
    ]))
    return r.choices[0].message.content.strip()


async def judge(question, gold, pred):
    r = await with_retry(lambda: oai.chat.completions.create(model=MODEL, temperature=0, messages=[
        {'role': 'system', 'content': 'Grade if the predicted answer matches the gold answer in meaning '
         '(for gold like "not mentioned", an abstention is correct). Reply exactly CORRECT or WRONG.'},
        {'role': 'user', 'content': f'QUESTION: {question}\nGOLD: {gold}\nPREDICTED: {pred}'},
    ]))
    return r.choices[0].message.content.strip().upper().startswith('CORRECT')


async def run_qa(g, gid, q, sem):
    async with sem:
        try:
            facts, ev, summaries = await retrieve(g, gid, q['question'])
            pred = await answer(q['question'], facts, ev, summaries)
            correct = await judge(q['question'], str(q.get('answer')), pred)
            return {'cat': q.get('category'), 'correct': correct, 'q': q['question'],
                    'gold': str(q.get('answer')), 'pred': pred}
        except Exception as e:
            return {'cat': q.get('category'), 'correct': False, 'error': repr(e)[:150], 'q': q['question']}


async def main():
    data = json.load(open(DATA))
    g = Graphiti(graph_driver=FalkorDriver(host='localhost', port=6379, database='locomo1'))
    await g.build_indices_and_constraints()
    log(f'N_CONV={N_CONV} QA_PER_CONV={QA_PER_CONV} TOP_K={TOP_K} reader={READER_MODEL}\n')
    all_results = []
    sem = asyncio.Semaphore(CONCURRENCY)
    for ci, conv_item in enumerate(data[:N_CONV]):
        gid = conv_item.get('sample_id', f'conv{ci}')
        log(f'ingesting conversation {ci} ({gid}) ...')
        await ingest_conversation(g, conv_item['conversation'], gid)
        qs = sample_qa(conv_item['qa'], QA_PER_CONV)
        res = await asyncio.gather(*[run_qa(g, gid, q, sem) for q in qs])
        all_results.extend(res)
        log(f'  conversation {ci}: {sum(bool(r.get("correct")) for r in res)}/{len(res)} correct')
    await g.close()

    by = defaultdict(lambda: [0, 0])
    errs = 0
    for r in all_results:
        by[r['cat']][1] += 1
        by[r['cat']][0] += int(bool(r.get('correct')))
        errs += int(bool(r.get('error')))
    tc = sum(v[0] for v in by.values()); tn = sum(v[1] for v in by.values())
    log('\n' + '=' * 60)
    log(f'{"category":22} {"acc":>6}   n')
    for cat in sorted(by):
        c, n = by[cat]
        log(f'{CATS.get(cat, cat)!s:22} {c / n:>6.0%}   {c}/{n}')
    log('-' * 60)
    log(f'{"OVERALL":22} {tc / tn:>6.0%}   {tc}/{tn}  (errors={errs})')
    log('\n--- transcripts ---')
    for r in all_results[:12]:
        mark = '✓' if r.get('correct') else ('ERR' if r.get('error') else '✗')
        log(f'[{mark}] (cat{r["cat"]}) Q: {r["q"][:80]}')
        log(f'      gold: {str(r.get("gold"))[:80]}  pred: {str(r.get("pred"))[:90]}')


asyncio.run(main())
OUT.close()
