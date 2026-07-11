"""Ambient proactive-memory assembly layer.

This is a *read-time* layer on top of stock Graphiti search. It takes the
edges that ``graphiti.search`` already returns and assembles a compact,
relevance-first, deduplicated, confidence-gated "injection block" suitable for
proactively priming an assistant's next turn in an ongoing conversation.

Design notes (Slice 1 — no schema migration):
- The assembly walk is deliberately run HERE, in the server assembly layer, over
  the ``list[EntityEdge]`` that ``graphiti.search`` returns -- NOT as a core
  ``EdgeReranker`` -- because the core search recipes are mutable module
  singletons (``graphiti.search`` sets ``.limit`` on a shared
  ``EDGE_HYBRID_SEARCH_RRF``) and would leak per-request state across concurrent
  ambient calls. Keeping the walk out-of-core keeps ``graphiti_core`` untouched.
- Assembly is relevance-first with a content near-duplicate guard and a soft
  per-entity cap (see ``compose_ambient_block``) -- refined from live evidence
  that pure endpoint-diversity both admitted verbatim-ish dupes and dropped the
  single most on-topic fact.
- ``heuristic_confidence`` is a pure function of fields that ALREADY persist on
  every edge (recency of ``valid_at``/``reference_time``/``created_at`` and the
  corroboration proxy ``len(edge.episodes)``). Nothing is written; there is no
  migration. Slice 2 replaces this with a real persisted ``Confidence`` that is
  corroborated on re-assertion and contested on contradiction.
- The confidence math (base rating/uncertainty, decay-per-day, corroboration
  steps) is UNCALIBRATED -- it mirrors memory-engine's ``confidence.py`` shape
  (``with_time_decay`` / ``corroborate``) and is tunable via server settings.

Ported in spirit from memory-engine ``retrieval.py:search_node_diverse`` and
``confidence.py`` -- the ideas, not the code.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from graphiti_core.edges import EntityEdge
from graphiti_core.utils.confidence import Confidence, with_time_decay

from graph_service.dto import Citation

# --- Heuristic confidence constants (UNCALIBRATED; superseded by Slice 2) -----
# A single, plainly-stated, freshly-observed fact should be admitted and NOT
# flagged as unconfirmed: base uncertainty sits below CONTESTED_UNCERTAINTY.
BASE_RATING = 0.7
BASE_UNCERTAINTY = 0.35
# Each additional corroborating episode closes CORROB_RATING_STEP of the gap to
# 1.0 and multiplies uncertainty by CORROB_UNCERTAINTY_FACTOR (narrows it).
CORROB_RATING_STEP = 0.3
CORROB_UNCERTAINTY_FACTOR = 0.7
# A fact loses trust as it ages without reinforcement. Read-time only.
DEFAULT_DECAY_PER_DAY = 0.005
# Cap how many corroborations we bother compounding -- purely a loop guard.
_MAX_CORROBORATION = 25
# Admitted facts at/above this uncertainty are surfaced but hedged.
CONTESTED_UNCERTAINTY = 0.5
# Default drop threshold -- an admitted fact must be below this uncertainty.
DEFAULT_UNCERTAINTY_THRESHOLD = 0.6

# --- Assembly (retrieval-quality) constants -----------------------------------
# Allow topical DEPTH -- multiple facts about the same entity when the
# conversation is about it -- but stop one entity from dominating the block.
# (Live evidence: pure "one new endpoint per fact" diversity dropped the single
# most on-topic fact because its entities appeared elsewhere.)
DEFAULT_MAX_FACTS_PER_ENTITY = 2
# Two facts whose content tokens overlap at/above this containment ratio are
# near-duplicates; the later (lower-ranked) one is dropped. (Live evidence:
# "X blocks Fridays" and "X blocks Fridays with no meetings" both got admitted.)
DEFAULT_DEDUP_CONTAINMENT = 0.8
# --- Salience (proactive "when to stay silent") constants ---------------------
# Two-level salience gate (the proactive "when to speak" control):
#
# - min_top_relevance (BLOCK level): decide whether to speak AT ALL. If the most
#   relevant fact is below this, the whole conversation is off-topic for memory
#   and the block is empty -- the assistant stays silent. This is the primary
#   silence control: in a self-centric graph every window weakly matches the
#   user's own facts (~0.3), so a per-fact floor alone never goes silent; the
#   block-level "is ANYTHING strongly relevant?" test is what does.
# - min_relevance (FACT level): once we've decided to speak, shape the block by
#   dropping individual facts below this lower floor.
#
# CALIBRATED on the demo corpus with OpenAI text-embedding-3-small: off-topic
# windows topped out ~0.37, on-topic windows led with a fact >= 0.48. Retune per
# deployment + embedding model.
DEFAULT_MIN_TOP_RELEVANCE = 0.40
DEFAULT_MIN_RELEVANCE = 0.22

# Tiny stopword set so shared function words don't inflate the overlap ratio.
_STOPWORDS = frozenset(
    {
        'the',
        'a',
        'an',
        'is',
        'are',
        'was',
        'were',
        'be',
        'being',
        'been',
        'of',
        'to',
        'for',
        'and',
        'or',
        'with',
        'in',
        'on',
        'at',
        'that',
        'this',
        'it',
        'as',
        'by',
        'from',
    }
)


def self_uuid_for_group(group_id: str) -> str:
    """Deterministic per-namespace 'self'/user key.

    Single source of truth so the ambient reader (exclusion key only, Slice 1)
    and the future self-entity upsert + ingest binding (Slice 4) agree on the
    same uuid without any shared state. In Slice 1 no node carries this uuid, so
    passing it as an exclusion key is a harmless no-op that wires the
    self-exclusion path from day one.
    """
    return str(uuid5(NAMESPACE_URL, f'graphiti-self:{group_id}'))


def _ensure_aware(dt: datetime) -> datetime:
    """Treat naive timestamps as UTC (FalkorDB stores ISO strings; Neo4j returns
    tz-aware) so arithmetic against a tz-aware ``now`` never raises."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _default_token_counter(text: str) -> int:
    """Coarse ~4-chars-per-token estimate (memory-engine's default). A caller
    can pass a real tokenizer if exactness matters."""
    return max(1, len(text) // 4)


def _content_tokens(text: str) -> frozenset[str]:
    """Lowercased content words of a fact (stopwords removed) -- the unit of the
    near-duplicate check."""
    return frozenset(w for w in re.findall(r'[a-z0-9]+', text.lower()) if w not in _STOPWORDS)


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    """Overlap of two token sets normalized by the SMALLER set, so 'X' vs
    'X, with more detail' scores ~1.0 -- catching elaboration-duplicates a
    Jaccard score would miss."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain cosine similarity; 0.0 for a zero or mismatched vector. Used to
    score each candidate fact against the transcript window for salience."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _effective_timestamp(edge: EntityEdge) -> datetime:
    """The fact's effective time: when it became true, else the producing
    episode's reference time, else when the edge row was created.

    ``reference_time`` only exists on newer graphiti-core (the local fork); the
    ``getattr`` keeps Slice 1 working against the published core the server
    currently installs, so this layer needs no core change to ship."""
    return edge.valid_at or getattr(edge, 'reference_time', None) or edge.created_at


def heuristic_confidence(
    edge: EntityEdge,
    now: datetime,
    decay_per_day: float = DEFAULT_DECAY_PER_DAY,
) -> tuple[float, float]:
    """(rating, uncertainty) from stock edge fields only -- no persisted state.

    - Corroboration: ``len(edge.episodes)`` beyond the first raises rating and
      narrows uncertainty (mirrors ``confidence.corroborate``).
    - Recency: uncertainty widens with the age of the fact's effective
      timestamp (mirrors ``confidence.with_time_decay``).
    - Invalidation: an edge with ``invalid_at`` in the past is downweighted
      (``expired_at`` edges are dropped by ``compose_ambient_block`` before this
      is even consulted, but we stay defensive here too).
    """
    rating = BASE_RATING
    uncertainty = BASE_UNCERTAINTY

    extra_corroboration = min(max(len(edge.episodes) - 1, 0), _MAX_CORROBORATION)
    for _ in range(extra_corroboration):
        rating += (1.0 - rating) * CORROB_RATING_STEP
        uncertainty *= CORROB_UNCERTAINTY_FACTOR

    effective_ts = _effective_timestamp(edge)
    elapsed_days = max((now - _ensure_aware(effective_ts)).total_seconds() / 86400.0, 0.0)
    uncertainty += elapsed_days * decay_per_day

    if edge.expired_at is not None or (
        edge.invalid_at is not None and _ensure_aware(edge.invalid_at) <= now
    ):
        rating *= 0.5
        uncertainty += 0.2

    rating = min(max(rating, 0.0), 1.0)
    uncertainty = min(max(uncertainty, 0.0), 1.0)
    return rating, uncertainty


def effective_confidence(
    edge: EntityEdge,
    now: datetime,
    decay_per_day: float = DEFAULT_DECAY_PER_DAY,
) -> tuple[float, float]:
    """Read-time (rating, uncertainty) for an edge -- the Phase-3 trust signal.

    Prefers the PERSISTED confidence written on ingest (Phase 1-2: corroborated on
    re-assertion, contested on contradiction), applying read-time decay via the
    canonical ``with_time_decay``. A ``confirmed`` (user-stated) fact is pinned --
    full trust, exempt from decay. Falls back to the stock-field
    ``heuristic_confidence`` for edges that predate the persisted write-path
    (``confidence_last_touched_at`` is None), so this works against stock core too.
    """
    if getattr(edge, 'confirmed', False):
        return 1.0, 0.0  # pinned ground truth -- never decays

    last_touched = getattr(edge, 'confidence_last_touched_at', None)
    if last_touched is None:
        return heuristic_confidence(edge, now, decay_per_day=decay_per_day)

    decayed = with_time_decay(
        Confidence(
            rating=edge.confidence_rating,
            uncertainty=edge.confidence_uncertainty,
            last_touched_at=last_touched,
            corroboration_count=getattr(edge, 'corroboration_count', 1),
        ),
        now,
    )
    rating, uncertainty = decayed.rating, decayed.uncertainty
    if edge.expired_at is not None or (
        edge.invalid_at is not None and _ensure_aware(edge.invalid_at) <= now
    ):
        rating *= 0.5
        uncertainty += 0.2
    return min(max(rating, 0.0), 1.0), min(max(uncertainty, 0.0), 1.0)


def compose_ambient_block(
    edges: list[EntityEdge],
    *,
    token_budget: int,
    now: datetime,
    self_uuid: str | None = None,
    relevance_by_uuid: dict[str, float] | None = None,
    min_top_relevance: float = 0.0,
    min_relevance: float = 0.0,
    min_rating: float = 0.0,
    uncertainty_threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    decay_per_day: float = DEFAULT_DECAY_PER_DAY,
    max_facts_per_entity: int = DEFAULT_MAX_FACTS_PER_ENTITY,
    dedup_containment: float = DEFAULT_DEDUP_CONTAINMENT,
    token_counter=None,
) -> tuple[str, list[Citation]]:
    """Relevance-first fill of a TOKEN budget over the search ranking.

    Walks ``edges`` in their incoming (relevance) rank order and admits a fact
    unless it is (a) not salient to the current conversation, (b) below the
    confidence gate, (c) a near-duplicate of an already-admitted fact, or (d)
    about entities that are already saturated. This keeps the block on-topic (a
    conversation about the billing migration should get several billing facts)
    while preventing off-topic injection, verbatim repetition, and any single
    entity crowding out everything else.

    Design, refined from live evidence over the earlier pure endpoint-diversity:
    - Salience gate (the "when to speak" control), two levels when
      ``relevance_by_uuid`` is provided: the block is emitted only if some fact
      clears ``min_top_relevance`` (else the assistant stays silent on an
      off-topic conversation); within an emitted block, facts below the lower
      ``min_relevance`` are dropped. A self-centric graph weakly matches every
      window, so the block-level test -- not the per-fact floor -- is what
      produces silence.
    - Confidence gate: drop expired / low-rating / high-uncertainty facts.
    - Near-duplicate guard: drop a fact whose content tokens are ~contained in an
      already-admitted fact's (catches "X" vs "X with more detail").
    - Soft per-entity cap: allow up to ``max_facts_per_entity`` facts touching a
      given non-self entity -- depth, not a hard "one new endpoint" rule that
      would drop the most on-topic fact just because its entities recurred.
    - Token budget, not fact count: on an over-budget fact ``continue`` scanning
      (a cheaper, deeper fact may still fit) rather than ``break``.

    ``relevance_by_uuid`` maps edge uuid -> cosine similarity to the window;
    ``None`` disables the salience gate entirely (used by pure unit tests).

    Returns ``(injection_block_text, citations)`` where the block is
    ``[i] fact`` lines whose ``[i]`` indexes into ``citations``.
    """
    count = token_counter or _default_token_counter

    entity_facts: dict[str, int] = {}  # non-self endpoint -> admitted count
    admitted_tokens: list[frozenset[str]] = []
    spent = 0
    picked: list[tuple[EntityEdge, float, float, float | None]] = []

    for edge in edges:
        if edge.expired_at is not None:
            continue  # superseded fact -- proactive injection wants current truth

        expires = getattr(edge, 'expires_at', None)
        if expires is not None and _ensure_aware(expires) <= now:
            continue  # past its validity horizon (TTL) -- e.g. "OOO until Friday"

        relevance = None if relevance_by_uuid is None else relevance_by_uuid.get(edge.uuid, 0.0)
        if relevance is not None and relevance < min_relevance:
            continue  # not salient to the current conversation -- stay silent

        rating, uncertainty = effective_confidence(edge, now, decay_per_day=decay_per_day)
        if rating < min_rating or uncertainty > uncertainty_threshold:
            continue  # confidence gate (persisted confidence w/ decay, else heuristic)

        tokens = _content_tokens(edge.fact)
        if any(_containment(tokens, seen) >= dedup_containment for seen in admitted_tokens):
            continue  # near-duplicate of an already-admitted, higher-ranked fact

        endpoints = {u for u in (edge.source_node_uuid, edge.target_node_uuid) if u != self_uuid}
        if endpoints and all(entity_facts.get(u, 0) >= max_facts_per_entity for u in endpoints):
            continue  # every entity this fact touches is already saturated

        cost = count(edge.fact)
        if spent + cost > token_budget:
            continue  # over budget for THIS fact; keep scanning for a cheaper one

        for u in endpoints:
            entity_facts[u] = entity_facts.get(u, 0) + 1
        admitted_tokens.append(tokens)
        spent += cost
        picked.append((edge, rating, uncertainty, relevance))

    # Block-level salience: stay silent unless SOMETHING is strongly relevant.
    if relevance_by_uuid is not None and picked:
        top_relevance = max((r for *_, r in picked if r is not None), default=0.0)
        if top_relevance < min_top_relevance:
            picked = []  # nothing worth interrupting for -- say nothing

    lines: list[str] = []
    citations: list[Citation] = []
    for i, (edge, rating, uncertainty, relevance) in enumerate(picked, start=1):
        contested = uncertainty >= CONTESTED_UNCERTAINTY
        prefix = '(unconfirmed) ' if contested else ''
        lines.append(f'[{i}] {prefix}{edge.fact}')
        citations.append(
            Citation(
                edge_uuid=edge.uuid,
                fact=edge.fact,
                episode_uuid=edge.episodes[0] if edge.episodes else None,
                # Normalize to tz-aware UTC so the emitted timestamp matches the
                # value the confidence math scored (both treat naive as UTC).
                reference_time=_ensure_aware(_effective_timestamp(edge)),
                rating=round(rating, 4),
                uncertainty=round(uncertainty, 4),
                contested=contested,
                relevance=None if relevance is None else round(relevance, 4),
            )
        )

    return '\n'.join(lines), citations
