"""Phase 6: the decision-trace ledger + usefulness feedback loop.

Every ambient gate decision is logged as a first-class record -- what was
considered, what was admitted, with what scores, and (later) what the user did
about it. memory-engine's hardest-won lesson: "the most valuable field is the
one that records the questions the system did NOT ask"; a gate you can't replay
is a gate you can't tune.

The ledger enables:
- SUPPRESSION (proactive): don't re-surface a fact injected recently -- a
  proactive assistant that repeats itself gets dismissed.
- OUTCOME labels (engaged / dismissed / ignored) via POST /ambient-outcome --
  the labeled dataset that later tunes salience weights (the Mem0-style
  usefulness boost stays OFF until this data exists; clamp [0.3, 1.5] when it
  ships, reactive-only).

Traces are stored as `DecisionTrace` nodes in the group's graph (single-writer:
one small write per ambient call is the decision log itself, not serving state).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

logger = logging.getLogger(__name__)

DEFAULT_SUPPRESSION_WINDOW_MIN = 240  # don't re-surface a fact within 4h
MAX_RECENT_TRACES = 20


def build_trace(
    group_id: str,
    window: str,
    admitted_edge_uuids: list[str],
    considered: int,
    top_relevance: float | None,
    spoke: bool,
    now: datetime,
) -> dict:
    """Pure: the trace record for one ambient gate decision."""
    return {
        'uuid': str(uuid4()),
        'group_id': group_id,
        'window_head': window[:200],
        'admitted_edge_uuids': json.dumps(admitted_edge_uuids),
        'considered': considered,
        'top_relevance': top_relevance if top_relevance is not None else -1.0,
        'spoke': spoke,
        'outcome': '',  # engaged | dismissed | ignored -- set via /ambient-outcome
        'created_at': now.isoformat(),
    }


def suppressed_uuids(
    recent_traces: list[dict],
    now: datetime,
    window_minutes: int = DEFAULT_SUPPRESSION_WINDOW_MIN,
) -> set[str]:
    """Pure: edge uuids injected within the suppression window. A trace whose
    outcome is 'engaged' does NOT suppress (the user wanted it; surfacing related
    context again is fine) -- dismissed/ignored/unlabeled traces do."""
    cutoff = now - timedelta(minutes=window_minutes)
    out: set[str] = set()
    for t in recent_traces:
        if t.get('outcome') == 'engaged':
            continue
        try:
            created = datetime.fromisoformat(t['created_at'])
        except Exception:
            continue
        if created >= cutoff:
            try:
                out.update(json.loads(t.get('admitted_edge_uuids') or '[]'))
            except Exception:
                pass
    return out


async def save_trace(driver, trace: dict) -> None:
    await driver.execute_query(
        'CREATE (t:DecisionTrace {uuid: $uuid, group_id: $group_id, window_head: $window_head, '
        'admitted_edge_uuids: $admitted_edge_uuids, considered: $considered, '
        'top_relevance: $top_relevance, spoke: $spoke, outcome: $outcome, created_at: $created_at})',
        **trace,
    )


async def load_recent_traces(driver, group_id: str, limit: int = MAX_RECENT_TRACES) -> list[dict]:
    records, _, _ = await driver.execute_query(
        'MATCH (t:DecisionTrace {group_id: $group_id}) RETURN t.uuid AS uuid, '
        't.admitted_edge_uuids AS admitted_edge_uuids, t.outcome AS outcome, '
        't.created_at AS created_at ORDER BY t.created_at DESC LIMIT $limit',
        group_id=group_id,
        limit=limit,
    )
    return [dict(r) for r in records]


async def record_outcome(driver, trace_uuid: str, outcome: str) -> bool:
    records, _, _ = await driver.execute_query(
        'MATCH (t:DecisionTrace {uuid: $uuid}) SET t.outcome = $outcome RETURN t.uuid AS uuid',
        uuid=trace_uuid,
        outcome=outcome,
    )
    return bool(records)
