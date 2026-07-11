"""Unit tests for the ambient proactive-memory assembly layer.

Pure and DB-free: every test builds ``EntityEdge`` fixtures by hand and exercises
``compose_ambient_block`` / ``heuristic_confidence`` directly. No network, no
graph, no OpenAI -- these run in milliseconds and are the fast counterpart to the
live FalkorDB integration test.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from graphiti_core.edges import EntityEdge

from graph_service.ambient import (
    CONTESTED_UNCERTAINTY,
    DEFAULT_DECAY_PER_DAY,
    _default_token_counter,
    compose_ambient_block,
    heuristic_confidence,
    self_uuid_for_group,
)

NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_edge(
    source: str,
    target: str,
    fact: str,
    *,
    episodes: list[str] | None = None,
    age_days: float = 0.0,
    valid_at: datetime | None = None,
    invalid_at: datetime | None = None,
    expired_at: datetime | None = None,
    uuid: str | None = None,
) -> EntityEdge:
    created = NOW - timedelta(days=age_days)
    return EntityEdge(
        uuid=uuid or f'edge-{source}-{target}-{fact[:6]}',
        group_id='g',
        source_node_uuid=source,
        target_node_uuid=target,
        name='rel',
        fact=fact,
        episodes=episodes if episodes is not None else ['ep1'],
        created_at=created,
        valid_at=valid_at,
        invalid_at=invalid_at,
        expired_at=expired_at,
    )


# --- heuristic_confidence -----------------------------------------------------


def test_fresh_single_fact_is_admissible_and_not_contested():
    rating, uncertainty = heuristic_confidence(make_edge('a', 'b', 'x'), NOW)
    assert rating > 0.5
    assert uncertainty < CONTESTED_UNCERTAINTY  # a plain fresh fact is not hedged


def test_corroboration_raises_rating_and_narrows_uncertainty():
    single = make_edge('a', 'b', 'x', episodes=['e1'])
    corroborated = make_edge('a', 'b', 'x', episodes=['e1', 'e2', 'e3'])
    r1, u1 = heuristic_confidence(single, NOW)
    r3, u3 = heuristic_confidence(corroborated, NOW)
    assert r3 > r1
    assert u3 < u1


def test_age_widens_uncertainty():
    fresh = heuristic_confidence(make_edge('a', 'b', 'x', age_days=0), NOW)[1]
    old = heuristic_confidence(make_edge('a', 'b', 'x', age_days=40), NOW)[1]
    assert old > fresh


def test_naive_timestamp_does_not_raise():
    edge = make_edge('a', 'b', 'x')
    edge.created_at = edge.created_at.replace(tzinfo=None)  # naive -> treated as UTC
    rating, uncertainty = heuristic_confidence(edge, NOW)
    assert 0.0 <= rating <= 1.0 and 0.0 <= uncertainty <= 1.0


def test_utc_iso_treats_naive_as_utc_regardless_of_host_tz():
    # Regression: astimezone() on a naive datetime assumes host-local tz. Under a
    # non-UTC host that shifts the emitted timestamp; _utc_iso must treat naive as
    # UTC (matching the confidence math) so output is host-tz-independent.
    import os
    import time

    from graph_service.dto.retrieve import _utc_iso

    naive = datetime(2026, 7, 11, 12, 0, 0)
    old_tz = os.environ.get('TZ')
    try:
        os.environ['TZ'] = 'America/New_York'
        time.tzset()
        assert _utc_iso(naive) == '2026-07-11T12:00:00+00:00'  # UTC noon, not 16:00
    finally:
        if old_tz is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = old_tz
        time.tzset()


def test_naive_edge_timestamp_serializes_as_utc_in_citation():
    # End-to-end: a naive edge valid_at must reach the API as the SAME UTC value
    # the recency gate scored -- not shifted by the server's UTC offset.
    edge = make_edge('a', 'b', 'x', uuid='A')
    edge.valid_at = datetime(2026, 7, 11, 12, 0, 0)  # naive (FalkorDB-style)
    _, citations = compose_ambient_block([edge], token_budget=10_000, now=NOW)
    dumped = citations[0].model_dump(mode='json')
    assert dumped['reference_time'] == '2026-07-11T12:00:00+00:00'


# --- compose_ambient_block ----------------------------------------------------


def test_empty_edges_yields_empty_block():
    block, citations = compose_ambient_block([], token_budget=100, now=NOW)
    assert block == ''
    assert citations == []


def test_near_duplicate_facts_are_deduped():
    # Second fact elaborates the first; even though it touches a different entity,
    # its content is ~contained in the first -> dropped. This is the live
    # "blocks Fridays" / "blocks Fridays with no meetings" [7]/[8] case.
    edges = [
        make_edge('alex', 'friday', 'Alex blocks Fridays for deep work', uuid='A'),
        make_edge(
            'alex', 'meeting', 'Alex blocks Fridays for deep work with no meetings', uuid='B'
        ),
    ]
    _, citations = compose_ambient_block(edges, token_budget=10_000, now=NOW)
    assert {c.edge_uuid for c in citations} == {'A'}  # B is a near-duplicate of A


def test_distinct_facts_sharing_an_entity_are_both_kept():
    # Two genuinely different facts about the same central entity (the on-topic
    # depth case) must BOTH survive -- the failure the old hard-diversity showed
    # by dropping "migrated onto Adyen, Q3" after "migrated off Stripe".
    edges = [
        make_edge(
            'billing', 'stripe', 'The billing service is being migrated off Stripe', uuid='OFF'
        ),
        make_edge(
            'billing',
            'adyen',
            'The billing service is being migrated onto Adyen targeting Q3',
            uuid='ONTO',
        ),
    ]
    _, citations = compose_ambient_block(edges, token_budget=10_000, now=NOW)
    assert {c.edge_uuid for c in citations} == {'OFF', 'ONTO'}


def test_same_pair_repetition_capped_per_entity():
    # Three distinct-content facts on the SAME (a,b) pair: with the default cap of
    # 2, the third is dropped once both endpoints are saturated.
    edges = [
        make_edge('a', 'b', 'alpha detail one', uuid='F1'),
        make_edge('a', 'b', 'beta detail two', uuid='F2'),
        make_edge('a', 'b', 'gamma detail three', uuid='F3'),
    ]
    _, citations = compose_ambient_block(edges, token_budget=10_000, now=NOW)
    assert {c.edge_uuid for c in citations} == {'F1', 'F2'}


def test_token_budget_respected():
    # Distinct content per fact (identical text would be near-dup-collapsed).
    edges = [make_edge(f's{i}', f't{i}', f'{i:02d} ' + 'x' * 37, uuid=f'E{i}') for i in range(10)]
    budget = 25  # each fact is 40 chars -> 40 // 4 = 10 tokens
    _, citations = compose_ambient_block(edges, token_budget=budget, now=NOW)
    spent = sum(_default_token_counter(c.fact) for c in citations)
    assert spent <= budget
    assert len(citations) == 2  # 10 + 10 fit, third 30 > 25


def test_continue_not_break_lets_a_cheaper_deeper_fact_fit():
    big = make_edge('a', 'b', 'b' * 40, uuid='BIG')  # cost 10, distinct token
    small = make_edge('c', 'd', 's' * 8, uuid='SMALL')  # cost 2, distinct token
    _, citations = compose_ambient_block([big, small], token_budget=5, now=NOW)
    picked = {c.edge_uuid for c in citations}
    assert picked == {'SMALL'}  # BIG over budget was skipped, scan continued


def test_self_endpoint_not_counted_toward_saturation():
    # Self-exclusion means the self node never consumes an entity slot. With a cap
    # of 1 and 'a' saturated by an ordinary fact, a (self, a) fact is skipped when
    # self is excluded (a is its only counted endpoint, saturated) but admitted
    # when self is NOT excluded (self is then an unsaturated endpoint).
    s = self_uuid_for_group('g')
    edges = [
        make_edge('a', 'x', 'ordinary fact saturating that node', uuid='AX'),
        make_edge(s, 'a', 'distinct self statement touching it', uuid='SA'),
    ]
    _, with_excl = compose_ambient_block(
        edges, token_budget=10_000, now=NOW, self_uuid=s, max_facts_per_entity=1
    )
    assert {c.edge_uuid for c in with_excl} == {'AX'}
    _, without_excl = compose_ambient_block(
        edges, token_budget=10_000, now=NOW, self_uuid=None, max_facts_per_entity=1
    )
    assert {c.edge_uuid for c in without_excl} == {'AX', 'SA'}


def test_expired_edges_are_dropped():
    edges = [
        make_edge('a', 'b', 'current', uuid='LIVE'),
        make_edge('c', 'd', 'superseded', uuid='DEAD', expired_at=NOW - timedelta(days=1)),
    ]
    _, citations = compose_ambient_block(edges, token_budget=10_000, now=NOW)
    assert {c.edge_uuid for c in citations} == {'LIVE'}


def test_stale_uncorroborated_fact_dropped_by_uncertainty_gate():
    # 60 days old at default decay 0.005/day -> uncertainty ~0.65 > 0.6 threshold.
    stale = make_edge('a', 'b', 'ancient', uuid='OLD', age_days=60)
    _, citations = compose_ambient_block(
        [stale], token_budget=10_000, now=NOW, decay_per_day=DEFAULT_DECAY_PER_DAY
    )
    assert citations == []


def test_min_rating_gate_drops_low_rated_facts():
    edge = make_edge('a', 'b', 'x')  # fresh rating ~0.7
    _, citations = compose_ambient_block([edge], token_budget=10_000, now=NOW, min_rating=0.9)
    assert citations == []


def test_contested_fact_is_kept_and_hedged():
    # ~30 days old -> uncertainty ~0.50: between CONTESTED_UNCERTAINTY and the drop
    # threshold, so admitted but flagged.
    edge = make_edge('a', 'b', 'shaky fact', uuid='SHAKY', age_days=30)
    block, citations = compose_ambient_block([edge], token_budget=10_000, now=NOW)
    assert len(citations) == 1
    assert citations[0].contested is True
    assert '(unconfirmed)' in block


def test_citation_shape_and_block_indexing():
    edges = [
        make_edge('a', 'b', 'alpha', episodes=['epA'], uuid='A'),
        make_edge('a', 'c', 'beta', episodes=['epB'], uuid='B'),
    ]
    block, citations = compose_ambient_block(edges, token_budget=10_000, now=NOW)
    assert len(citations) == 2
    first = citations[0]
    assert first.edge_uuid == 'A'
    assert first.fact == 'alpha'
    assert first.episode_uuid == 'epA'
    assert first.reference_time is not None
    assert 0.0 <= first.rating <= 1.0
    assert 0.0 <= first.uncertainty <= 1.0
    # Block lines are 1-indexed and align with citation order.
    lines = block.split('\n')
    assert lines[0].startswith('[1] ')
    assert lines[1].startswith('[2] ')
    assert 'alpha' in lines[0]
    assert 'beta' in lines[1]


def test_self_uuid_is_deterministic_and_group_scoped():
    assert self_uuid_for_group('g') == self_uuid_for_group('g')
    assert self_uuid_for_group('g') != self_uuid_for_group('other')


# --- salience gate ("when to stay silent") ------------------------------------


def test_cosine_similarity_helper():
    from graph_service.ambient import cosine_similarity

    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector
    assert cosine_similarity([1.0, 0.0], []) == 0.0  # mismatched length


def test_salience_gate_drops_facts_below_floor():
    edges = [make_edge('a', 'b', 'relevant', uuid='A'), make_edge('c', 'd', 'off topic', uuid='B')]
    _, citations = compose_ambient_block(
        edges,
        token_budget=10_000,
        now=NOW,
        relevance_by_uuid={'A': 0.5, 'B': 0.1},
        min_relevance=0.22,
    )
    assert {c.edge_uuid for c in citations} == {'A'}
    assert citations[0].relevance == 0.5


def test_salience_gate_empties_block_when_nothing_relevant():
    edges = [make_edge('a', 'b', 'x', uuid='A'), make_edge('c', 'd', 'y', uuid='B')]
    block, citations = compose_ambient_block(
        edges,
        token_budget=10_000,
        now=NOW,
        relevance_by_uuid={'A': 0.1, 'B': 0.05},
        min_relevance=0.22,
    )
    assert block == '' and citations == []  # the assistant stays silent


def test_salience_gate_disabled_when_relevance_is_none():
    edge = make_edge('a', 'b', 'x', uuid='A')
    _, citations = compose_ambient_block(
        [edge], token_budget=10_000, now=NOW, relevance_by_uuid=None, min_relevance=0.9
    )
    assert {c.edge_uuid for c in citations} == {'A'}  # no gating without scores
    assert citations[0].relevance is None


def test_block_level_gate_stays_silent_when_nothing_strongly_relevant():
    # Both facts clear the per-fact floor (0.22) but neither clears the block-level
    # floor (0.40): a self-centric conversation that weakly matches everything but
    # is about nothing in particular -> silence.
    edges = [make_edge('a', 'b', 'x', uuid='A'), make_edge('c', 'd', 'y', uuid='B')]
    block, citations = compose_ambient_block(
        edges,
        token_budget=10_000,
        now=NOW,
        relevance_by_uuid={'A': 0.35, 'B': 0.30},
        min_top_relevance=0.40,
        min_relevance=0.22,
    )
    assert block == '' and citations == []


def test_block_level_gate_speaks_and_includes_tangential_facts_once_triggered():
    # One strongly-relevant fact clears the block floor, so we speak -- and the
    # weaker-but-still-above-per-fact-floor fact rides along.
    edges = [make_edge('a', 'b', 'strong', uuid='A'), make_edge('c', 'd', 'weak', uuid='B')]
    _, citations = compose_ambient_block(
        edges,
        token_budget=10_000,
        now=NOW,
        relevance_by_uuid={'A': 0.50, 'B': 0.30},
        min_top_relevance=0.40,
        min_relevance=0.22,
    )
    assert {c.edge_uuid for c in citations} == {'A', 'B'}


# --- endpoint (HTTP handler, DB dependency faked) -----------------------------


class _FakeEmbedder:
    """Deterministic embedder: returns a preset vector per text, so salience is
    controllable in tests."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    async def create_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[t] for t in texts]


class _FakeGraphiti:
    """Stands in for the request-scoped ZepGraphiti; returns canned edges so the
    real get_ambient_context handler runs without a graph/OpenAI. ``embedder`` is
    None by default, which exercises the salience path's graceful degradation."""

    def __init__(self, edges: list[EntityEdge], embedder=None):
        self._edges = edges
        self.embedder = embedder
        self.calls: list[dict] = []

    async def search(self, *, group_ids, query, num_results):
        self.calls.append({'group_ids': group_ids, 'query': query, 'num_results': num_results})
        return self._edges


def _client(edges: list[EntityEdge], embedder=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from graph_service.config import Settings, get_settings
    from graph_service.routers import retrieve
    from graph_service.zep_graphiti import get_graphiti

    fake = _FakeGraphiti(edges, embedder=embedder)
    settings = Settings(openai_api_key='test')  # type: ignore[call-arg]

    app = FastAPI()
    app.include_router(retrieve.router)
    app.dependency_overrides[get_graphiti] = lambda: fake
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app), fake


def test_endpoint_empty_window_short_circuits_without_searching():
    client, fake = _client([make_edge('a', 'b', 'x')])
    resp = client.post('/get-ambient-context', json={'group_id': 'g', 'transcript_window': '   '})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {'injection_block': '', 'citations': []}
    assert fake.calls == []  # never hit search


def test_endpoint_returns_block_and_citations():
    edges = [
        make_edge('a', 'b', 'alpha fact', episodes=['epA'], uuid='A'),
        make_edge('a', 'c', 'beta fact', episodes=['epB'], uuid='B'),
    ]
    client, fake = _client(edges)
    resp = client.post(
        '/get-ambient-context',
        json={'group_id': 'g', 'transcript_window': 'we were talking about a'},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body['injection_block'].startswith('[1] ')
    assert len(body['citations']) == 2
    assert body['citations'][0]['edge_uuid'] == 'A'
    # Provenance fields are present in the contract but null in Slice 1.
    assert body['citations'][0]['supporting_quote'] is None
    # Search was called with the window as query and the group scoped.
    assert fake.calls[0]['group_ids'] == ['g']
    assert fake.calls[0]['query'] == 'we were talking about a'


def test_endpoint_honors_request_token_budget_override():
    edges = [make_edge(f's{i}', f't{i}', f'{i:02d} ' + 'x' * 37, uuid=f'E{i}') for i in range(5)]
    client, _ = _client(edges)
    resp = client.post(
        '/get-ambient-context',
        json={'group_id': 'g', 'transcript_window': 'cue', 'token_budget': 25},
    )
    assert resp.status_code == 200
    citations = resp.json()['citations']
    assert len(citations) == 2  # 40//4 == 10 each; only two fit in 25


def test_endpoint_rejects_nonpositive_token_budget():
    client, _ = _client([make_edge('a', 'b', 'x')])
    resp = client.post(
        '/get-ambient-context',
        json={'group_id': 'g', 'transcript_window': 'cue', 'token_budget': 0},
    )
    assert resp.status_code == 422  # DTO ge=1 rejects at the contract boundary


def test_endpoint_stays_silent_when_conversation_is_off_topic():
    # Window vector orthogonal to both fact vectors -> cosine 0 < 0.22 floor.
    window = 'totally unrelated small talk about lunch'
    edges = [make_edge('a', 'b', 'fact one', uuid='A'), make_edge('a', 'c', 'fact two', uuid='B')]
    vectors = {window: [1.0, 0.0], 'fact one': [0.0, 1.0], 'fact two': [0.0, 1.0]}
    client, _ = _client(edges, embedder=_FakeEmbedder(vectors))
    resp = client.post('/get-ambient-context', json={'group_id': 'g', 'transcript_window': window})
    assert resp.status_code == 200
    assert resp.json() == {'injection_block': '', 'citations': []}  # stays silent


def test_endpoint_injects_and_scores_relevant_facts():
    window = 'billing migration status'
    edges = [
        make_edge('billing', 'adyen', 'billing is migrating to Adyen', uuid='REL'),
        make_edge('x', 'y', 'unrelated hobby detail', uuid='IRR'),
    ]
    vectors = {
        window: [1.0, 0.0],
        'billing is migrating to Adyen': [0.95, 0.05],  # cosine ~0.999 -> admit
        'unrelated hobby detail': [0.0, 1.0],  # cosine 0 -> drop
    }
    client, _ = _client(edges, embedder=_FakeEmbedder(vectors))
    resp = client.post('/get-ambient-context', json={'group_id': 'g', 'transcript_window': window})
    assert resp.status_code == 200
    citations = resp.json()['citations']
    assert {c['edge_uuid'] for c in citations} == {'REL'}
    assert citations[0]['relevance'] > 0.9


def test_endpoint_stays_silent_when_nothing_clears_block_floor():
    # Both facts score 0.28-0.30: above the per-fact floor (0.22) but below the
    # block-level floor (0.40) -- weakly-related self-chatter about nothing.
    window = 'weakly related self chatter'
    edges = [make_edge('a', 'b', 'fact one', uuid='A'), make_edge('a', 'c', 'fact two', uuid='B')]
    vectors = {window: [1.0, 0.0], 'fact one': [0.30, 0.954], 'fact two': [0.28, 0.960]}
    client, _ = _client(edges, embedder=_FakeEmbedder(vectors))
    resp = client.post('/get-ambient-context', json={'group_id': 'g', 'transcript_window': window})
    assert resp.status_code == 200
    assert resp.json() == {'injection_block': '', 'citations': []}  # block floor not cleared
