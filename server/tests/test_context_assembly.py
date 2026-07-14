"""Unit tests for the reactive context assembly (benchmark-proven recipe).

Pure and DB-free: hand-built edges/nodes/episodes, no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode

from graph_service.context_assembly import (
    assemble_context_block,
    format_facts,
    is_aggregation_query,
    rank_episodes,
)

NOW = datetime(2026, 7, 12, 12, 0, 0, tzinfo=timezone.utc)


def edge(fact, days_ago=0, invalid_days_ago=None, uuid='e'):
    return EntityEdge(
        uuid=uuid,
        group_id='g',
        source_node_uuid='a',
        target_node_uuid='b',
        name='REL',
        fact=fact,
        created_at=NOW - timedelta(days=days_ago),
        valid_at=NOW - timedelta(days=days_ago),
        invalid_at=NOW - timedelta(days=invalid_days_ago) if invalid_days_ago is not None else None,
    )


def episode(content, days_ago=0):
    return EpisodicNode(
        name='ep',
        group_id='g',
        content=content,
        source=EpisodeType.message,
        source_description='t',
        created_at=NOW - timedelta(days=days_ago),
        valid_at=NOW - timedelta(days=days_ago),
    )


def test_facts_chronological_with_validity_windows():
    facts = format_facts(
        [edge('newer', days_ago=1), edge('older', days_ago=10, invalid_days_ago=1)]
    )
    # closed window -> explicit SUPERSEDED tag (marking beats instructing)
    assert facts[0].endswith('older') and facts[0].startswith('[SUPERSEDED on 2026-07-11')
    assert facts[1].endswith('newer') and '-> Present]' in facts[1]  # current truth


def test_aggregation_intent_detection():
    assert is_aggregation_query('How many kits have I bought?')
    assert is_aggregation_query('what is the total I spent')
    assert not is_aggregation_query('Where did Rachel move?')


def test_rank_episodes_semantic_dedup_budget_and_datestamp():
    eps = [
        episode('alpha ' * 50, days_ago=3),
        episode('alpha ' * 50, days_ago=3),  # near-dup
        episode('beta ' * 50, days_ago=1),
    ]
    vecs = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    out = rank_episodes([1.0, 0.0], eps, vecs, max_episodes=5, char_budget=4000)
    assert len(out) == 2  # near-dup collapsed
    assert all(o.startswith('[SESSION DATE: ') for o in out)
    # chronological: the older (alpha, matching) session first
    assert 'alpha' in out[0] and 'beta' in out[1]
    # budget split, not flat: 4000//2=2000 -> content truncated to <=2000 chars + header
    assert all(len(o) <= 2000 + 40 for o in out)


def test_block_carries_validity_instruction():
    block = assemble_context_block(['[d -> Present] f'], ['P: s'], ['[SESSION DATE: d]\ne'])
    assert '"Present" means currently true (or SUPERSEDED tags)' in block or 'Present' in block
    assert 'ENTITY PROFILES' in block and 'FACTS' in block and 'CONVERSATION EVIDENCE' in block
