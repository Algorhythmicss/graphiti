"""Reactive context assembly: the benchmark-proven retrieval recipe as a product API.

Assembles the three-channel context that measured best on LongMemEval (67-68%
overall, single-session 100%, vs Zep's recipe 58% in the identical harness):

1. FACTS -- graph edges, chronological, each carrying its validity window
   ``[valid_at -> invalid_at|Present]`` (the knowledge-update signal).
2. ENTITY PROFILES -- node summaries (cross-session aggregation).
3. EPISODES -- raw sessions ranked semantically against the query (recall of
   verbatim detail that fact extraction may compress away), near-dup deduped,
   under a total char budget split across episodes (flat truncation was proven
   to cut answers away).

Counting/aggregation queries ("how many/much/total...") swap top-K fact search
for retrieve-ALL facts in the group: ranking is the wrong tool for enumeration
(proven: multi-session counting failed at 20-40% under every top-K variant).

No LLM calls here -- pure retrieval + assembly; the caller's reader model
consumes the block. See server/evals/ for the harness this was validated in.
"""

from __future__ import annotations

import re
from datetime import datetime

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

from graph_service.ambient import _ensure_aware, cosine_similarity

COUNT_RE = re.compile(r'\bhow (many|much|long|often)\b|\btotal\b|\bcount\b', re.I)

DEFAULT_TOP_K = 40
DEFAULT_MAX_EPISODES = 8
DEFAULT_EPISODE_CHAR_BUDGET = 48_000
MAX_PER_EPISODE_CHARS = 20_000
MAX_PROFILES = 20


def _edge_date(e: EntityEdge) -> str:
    d = e.valid_at or getattr(e, 'reference_time', None) or e.created_at
    return d.strftime('%Y-%m-%d') if d else 'unknown'


def _edge_validity(e: EntityEdge) -> str:
    end = e.invalid_at.strftime('%Y-%m-%d') if e.invalid_at else 'Present'
    return f'[{_edge_date(e)} -> {end}] {e.fact}'


def format_facts(edges: list[EntityEdge]) -> list[str]:
    """Chronological facts carrying validity windows; 'Present' marks current truth."""
    return [_edge_validity(e) for e in sorted(edges, key=_edge_date)]


def format_profiles(nodes: list[EntityNode], limit: int = MAX_PROFILES) -> list[str]:
    return [f'{n.name}: {n.summary}' for n in nodes if n.summary][:limit]


def rank_episodes(
    query_vec: list[float],
    episodes: list[EpisodicNode],
    episode_vecs: list[list[float]],
    max_episodes: int = DEFAULT_MAX_EPISODES,
    char_budget: int = DEFAULT_EPISODE_CHAR_BUDGET,
) -> list[str]:
    """Semantic episode selection: rank by cosine to the query, dedupe
    near-identical content, date-stamp, present chronologically, and split the
    char budget across picks (never a flat truncation)."""
    ranked = sorted(
        zip(episodes, episode_vecs, strict=False),
        key=lambda p: cosine_similarity(query_vec, p[1]),
        reverse=True,
    )
    seen: set[str] = set()
    picked: list[EpisodicNode] = []
    for ep, _ in ranked:
        key = ep.content[:200]
        if key in seen:
            continue
        seen.add(key)
        picked.append(ep)
        if len(picked) >= max_episodes:
            break
    per_ep = min(MAX_PER_EPISODE_CHARS, max(1200, char_budget // max(1, len(picked))))
    picked.sort(key=lambda ep: _ensure_aware(ep.valid_at or ep.created_at))
    return [
        f'[SESSION DATE: {_ensure_aware(ep.valid_at or ep.created_at).strftime("%Y-%m-%d")}]\n'
        f'{ep.content[:per_ep]}'
        for ep in picked
    ]


def assemble_context_block(facts: list[str], profiles: list[str], episodes: list[str]) -> str:
    """The reader-facing block. Facts carry validity windows ('Present' =
    currently true); episodes are date-stamped and chronological."""
    fb = '\n'.join(facts) or '(none)'
    pb = '\n'.join(profiles) or '(none)'
    eb = '\n---\n'.join(episodes) or '(none)'
    return (
        'MEMORY CONTEXT\n'
        'Facts are valid between their bracketed dates; an end of "Present" means '
        'currently true. A later-dated statement supersedes an earlier one.\n\n'
        f'ENTITY PROFILES:\n{pb}\n\n'
        f'FACTS (chronological):\n{fb}\n\n'
        f'CONVERSATION EVIDENCE (chronological):\n{eb}'
    )


def is_aggregation_query(query: str) -> bool:
    """Counting/aggregation intent -> retrieve-ALL facts, not top-K."""
    return bool(COUNT_RE.search(query))


async def assemble_memory_context(
    graphiti,
    group_id: str,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    episode_char_budget: int = DEFAULT_EPISODE_CHAR_BUDGET,
    now: datetime | None = None,  # noqa: ARG001 (future: TTL/confidence gating)
) -> tuple[str, dict]:
    """Full three-channel retrieval + assembly against a Graphiti client.

    Returns (context_block, parts) where parts carries the raw channel lists
    for structured consumers.
    """
    if is_aggregation_query(query):
        edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id])
    else:
        edges = await graphiti.search(query, group_ids=[group_id], num_results=top_k)
    facts = format_facts(edges)

    try:
        nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
        profiles = format_profiles(nodes)
    except Exception:
        profiles = []

    episodes_fmt: list[str] = []
    try:
        eps = await EpisodicNode.get_by_group_ids(graphiti.driver, [group_id])
        if eps:
            vecs = await graphiti.embedder.create_batch([query] + [e.content[:2000] for e in eps])
            episodes_fmt = rank_episodes(
                vecs[0],
                eps,
                vecs[1:],
                max_episodes=max_episodes,
                char_budget=episode_char_budget,
            )
    except Exception:
        episodes_fmt = []

    block = assemble_context_block(facts, profiles, episodes_fmt)
    return block, {'facts': facts, 'profiles': profiles, 'episodes': episodes_fmt}
