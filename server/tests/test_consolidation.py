"""Unit tests for the Phase-5 consolidation job (pure parts + fake LLM)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode

from graph_service.consolidation import (
    find_stale_entity_uuids,
    find_ttl_expired,
    open_facts_for,
    regenerate_summary,
)

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


def edge(uuid, s='a', t='b', fact='f', invalid=None, expired=None, expires=None):
    return EntityEdge(
        uuid=uuid,
        group_id='g',
        source_node_uuid=s,
        target_node_uuid=t,
        name='R',
        fact=fact,
        created_at=NOW - timedelta(days=5),
        invalid_at=invalid,
        expired_at=expired,
        expires_at=expires,
    )


def test_ttl_expired_only_past_and_unexpired():
    edges = [
        edge('past', expires=NOW - timedelta(days=1)),
        edge('future', expires=NOW + timedelta(days=1)),
        edge('already', expires=NOW - timedelta(days=2), expired=NOW - timedelta(days=1)),
        edge('none'),
    ]
    assert [e.uuid for e in find_ttl_expired(edges, NOW)] == ['past']


def test_stale_entities_are_endpoints_of_superseded_edges():
    edges = [
        edge('live', s='a', t='b'),
        edge('dead', s='b', t='c', invalid=NOW - timedelta(days=1)),
    ]
    assert find_stale_entity_uuids(edges) == {'b', 'c'}


def test_open_facts_exclude_superseded():
    edges = [
        edge('e1', s='x', t='y', fact='current fact'),
        edge('e2', s='x', t='z', fact='old fact', invalid=NOW - timedelta(days=1)),
        edge('e3', s='q', t='r', fact='unrelated'),
    ]
    assert open_facts_for('x', edges) == ['current fact']


class _FakeLLM:
    def __init__(self):
        self.last_messages = None

    async def generate_response(self, messages, **kw):
        self.last_messages = messages
        return {'summary': 'fresh summary from open facts'}


@pytest.mark.asyncio
async def test_regenerate_summary_withholds_stale_summary():
    node = EntityNode(
        uuid='n1',
        name='runner',
        group_id='g',
        labels=['Entity'],
        summary='STALE: personal best is 27:12',
        created_at=NOW,
    )
    llm = _FakeLLM()
    out = await regenerate_summary(llm, node, ['personal best is 25:50'])
    assert out == 'fresh summary from open facts'
    blob = ' '.join(m.content for m in llm.last_messages)
    assert '27:12' not in blob  # stale summary must NOT reach the LLM
    assert '25:50' in blob  # open facts must
