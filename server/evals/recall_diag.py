"""CHEAP retrieval-recall diagnostic -- no reader, no judge. SCOPED-driver fix.

FalkorDB is multi-graph: each group_id is its own graph. Retrieval must use a
client SCOPED to database=group_id (else search reads the empty default graph --
the bug that made every prior benchmark score the reader guessing on empty
context). Reuses already-ingested per-qid graphs -> only cost is embeddings.

For each question, measures whether the GOLD-answer turns (has_answer) appear in
the retrieved context, under edge-ref vs semantic episode retrieval.
Usage: N_EPISODES=8 python recall_diag.py
"""

import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/Users/mac/graphiti-local/server')
from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodicNode  # noqa: E402
from graph_service.ambient import cosine_similarity  # noqa: E402

DATA = Path(__file__).parent / 'longmemeval_oracle.json'
TOP_K = int(os.environ.get('TOP_K', '30'))
N_EPISODES = int(os.environ.get('N_EPISODES', '8'))
PER_TYPE = int(os.environ.get('PER_TYPE', '10'))
OUT = open(Path(__file__).parent / 'recall_diag.out', 'w')


def log(*a):
    print(*a, file=OUT, flush=True)
    print(*a, flush=True)


def gold_turns(inst):
    out = []
    for sess in inst['haystack_sessions']:
        for t in sess:
            if t.get('has_answer'):
                out.append(t['content'].strip())
    return out


def covered(gold, blob):
    if not gold:
        return None
    return sum(1 for g in gold if g[:120] in blob) / len(gold)


async def edge_ref_eps(gs, gid, question):
    edges = await gs.search(question, group_ids=[gid], num_results=TOP_K)
    uuids, seen, eps = [], set(), []
    for e in edges:
        uuids.extend(e.episodes)
    for u in uuids:
        if u in seen:
            continue
        seen.add(u)
        try:
            eps.append((await EpisodicNode.get_by_uuid(gs.driver, u)).content)
        except Exception:
            pass
        if len(eps) >= N_EPISODES:
            break
    return len(edges), eps


async def semantic_eps(gs, gid, question):
    eps = await EpisodicNode.get_by_group_ids(gs.driver, [gid])
    if not eps:
        return []
    embs = await gs.embedder.create_batch([question] + [e.content[:2000] for e in eps])
    qv, evs = embs[0], embs[1:]
    ranked = sorted(zip(eps, evs, strict=False), key=lambda p: cosine_similarity(qv, p[1]), reverse=True)
    return [e.content for e, _ in ranked[:N_EPISODES]]


async def main():
    data = json.load(open(DATA))
    by = defaultdict(list)
    for x in data:
        by[x['question_type']].append(x)
    sample = [x for t in sorted(by) for x in by[t][:PER_TYPE]]
    log(f'N_EPISODES={N_EPISODES} sample={len(sample)} (SCOPED-driver retrieval)\n')

    agg = defaultdict(lambda: {'edge': [], 'sem': [], 'nedges': [], 'nsess': []})
    skipped = 0
    for inst in sample:
        gid = inst['question_id']
        gold = gold_turns(inst)
        if not gold:
            continue
        gs = Graphiti(graph_driver=FalkorDriver(host='localhost', port=6379, database=gid))
        try:
            if not await EntityNode.get_by_group_ids(gs.driver, [gid]):
                skipped += 1
                await gs.close()
                continue
            nedges, edge_eps = await edge_ref_eps(gs, gid, inst['question'])
            sem_eps = await semantic_eps(gs, gid, inst['question'])
        finally:
            await gs.close()
        qt = inst['question_type']
        agg[qt]['edge'].append(covered(gold, '\n'.join(edge_eps)))
        agg[qt]['sem'].append(covered(gold, '\n'.join(sem_eps)))
        agg[qt]['nedges'].append(nedges)
        agg[qt]['nsess'].append(len(inst['haystack_sessions']))

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    log(f'{"question_type":26} {"edge-ref":>9} {"semantic":>9} {"avg#edges":>10} {"#sess":>6}  n')
    ae, as_ = [], []
    for qt in sorted(agg):
        d = agg[qt]
        ae += d['edge']
        as_ += d['sem']
        log(f'{qt:26} {avg(d["edge"]):>9.0%} {avg(d["sem"]):>9.0%} {avg(d["nedges"]):>10.1f} {avg(d["nsess"]):>6.1f}  {len(d["edge"])}')
    log('-' * 68)
    log(f'{"OVERALL evidence-recall":26} {avg(ae):>9.0%} {avg(as_):>9.0%}')
    log(f'\n(skipped {skipped} not-ingested)')
    log('avg #edges/question (>0 means search now works):', round(avg([x for d in agg.values() for x in d['nedges']]), 1))


asyncio.run(main())
OUT.close()
