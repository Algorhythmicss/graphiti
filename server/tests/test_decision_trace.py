"""Unit tests for the Phase-6 decision-trace ledger (pure parts)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from graph_service.decision_trace import build_trace, suppressed_uuids

NOW = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)


def trace(uuids, minutes_ago, outcome=''):
    return {
        'admitted_edge_uuids': json.dumps(uuids),
        'outcome': outcome,
        'created_at': (NOW - timedelta(minutes=minutes_ago)).isoformat(),
    }


def test_build_trace_shape():
    t = build_trace(
        'g', 'win' * 200, ['e1', 'e2'], considered=9, top_relevance=0.61, spoke=True, now=NOW
    )
    assert t['group_id'] == 'g' and t['spoke'] is True and t['considered'] == 9
    assert len(t['window_head']) <= 200
    assert json.loads(t['admitted_edge_uuids']) == ['e1', 'e2']
    assert t['outcome'] == ''


def test_recent_unlabeled_and_dismissed_suppress():
    recent = [trace(['a'], 30), trace(['b'], 30, outcome='dismissed')]
    assert suppressed_uuids(recent, NOW) == {'a', 'b'}


def test_engaged_does_not_suppress():
    recent = [trace(['a'], 30, outcome='engaged')]
    assert suppressed_uuids(recent, NOW) == set()


def test_old_traces_do_not_suppress():
    recent = [trace(['a'], 999)]
    assert suppressed_uuids(recent, NOW, window_minutes=240) == set()


def test_malformed_trace_rows_are_skipped():
    recent = [
        {'admitted_edge_uuids': 'not json', 'outcome': '', 'created_at': 'bad'},
        trace(['ok'], 10),
    ]
    assert suppressed_uuids(recent, NOW) == {'ok'}
