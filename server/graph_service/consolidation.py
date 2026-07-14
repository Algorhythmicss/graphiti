"""Phase 5: sleep-time consolidation -- the single-writer maintenance job.

Evidence-backed by the LongMemEval loop: unversioned entity summaries leak stale
values into answers (a node summary flatly asserting an outdated personal best
overrode correctly-windowed facts until read-time SUPERSEDED tags patched it).
The durable fix is write-time: when an entity's facts get invalidated, its
summary must be REGENERATED from the still-open facts.

The job (run off the hot path, e.g. cron or post-ingest worker; serving stays
read-only):
1. TTL sweep -- expire edges past their `expires_at` validity horizon.
2. Stale-summary sweep -- find entities touching recently-invalidated edges and
   rebuild their summaries from open (non-superseded, non-expired) facts only.

Pure helpers are separated from IO for unit testing.
"""

from __future__ import annotations

import logging
from datetime import datetime

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_nodes import Summary

from graph_service.ambient import _ensure_aware

logger = logging.getLogger(__name__)

MAX_FACTS_FOR_SUMMARY = 30


def find_ttl_expired(edges: list[EntityEdge], now: datetime) -> list[EntityEdge]:
    """Edges past their expires_at horizon that are not yet marked expired."""
    out = []
    for e in edges:
        expires = getattr(e, 'expires_at', None)
        if expires is not None and _ensure_aware(expires) <= now and e.expired_at is None:
            out.append(e)
    return out


def find_stale_entity_uuids(edges: list[EntityEdge]) -> set[str]:
    """Entities touching any invalidated/expired edge -- their summaries may
    still assert the superseded value and must be regenerated."""
    stale: set[str] = set()
    for e in edges:
        if e.invalid_at is not None or e.expired_at is not None:
            stale.add(e.source_node_uuid)
            stale.add(e.target_node_uuid)
    return stale


def open_facts_for(node_uuid: str, edges: list[EntityEdge]) -> list[str]:
    """Current-truth facts touching a node: not invalidated, not expired."""
    return [
        e.fact
        for e in edges
        if (e.source_node_uuid == node_uuid or e.target_node_uuid == node_uuid)
        and e.invalid_at is None
        and e.expired_at is None
    ][:MAX_FACTS_FOR_SUMMARY]


async def regenerate_summary(llm_client, node: EntityNode, facts: list[str]) -> str:
    """Rebuild a node summary from ONLY the currently-true facts."""
    context = {
        'node_name': node.name,
        # Deliberately WITHHOLD the old summary: passing it would let the LLM
        # carry the superseded value forward -- the exact leak being fixed.
        'node_summary': '',
        # The open facts are the source content to summarize from.
        'episode_content': 'Currently true facts:\n' + '\n'.join(f'- {f}' for f in facts),
        'previous_episodes': [],
        'attributes': [],
    }
    response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summarize_context(context),
        response_model=Summary,
        model_size=ModelSize.small,
        prompt_name='summarize_nodes.summarize_context',
    )
    return str(response.get('summary', '')) or node.summary


async def consolidate_group(graphiti, group_id: str, now: datetime) -> dict:
    """Run one consolidation pass for a group. Returns a report dict."""
    edges = await EntityEdge.get_by_group_ids(graphiti.driver, [group_id])

    # 1. TTL sweep: close out facts past their validity horizon.
    ttl_expired = find_ttl_expired(edges, now)
    for e in ttl_expired:
        e.expired_at = now
        e.invalid_at = e.invalid_at or getattr(e, 'expires_at', None) or now
        await e.save(graphiti.driver)

    # 2. Stale-summary sweep: regenerate summaries of entities whose facts
    #    were superseded, from open facts only.
    stale_uuids = find_stale_entity_uuids(edges)
    regenerated = 0
    if stale_uuids:
        nodes = await EntityNode.get_by_group_ids(graphiti.driver, [group_id])
        by_uuid = {n.uuid: n for n in nodes}
        for uuid in stale_uuids:
            node = by_uuid.get(uuid)
            if node is None:
                continue
            facts = open_facts_for(uuid, edges)
            try:
                new_summary = await regenerate_summary(graphiti.llm_client, node, facts)
            except Exception:
                logger.warning('summary regeneration failed for %s', uuid, exc_info=True)
                continue
            if new_summary and new_summary != node.summary:
                node.summary = new_summary
                await node.save(graphiti.driver)
                regenerated += 1

    report = {
        'group_id': group_id,
        'edges_scanned': len(edges),
        'ttl_expired': len(ttl_expired),
        'stale_entities': len(stale_uuids),
        'summaries_regenerated': regenerated,
    }
    logger.info('consolidation: %s', report)
    return report
