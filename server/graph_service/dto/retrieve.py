from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


def _utc_iso(v: datetime) -> str:
    """Serialize a datetime to UTC ISO-8601.

    A naive value is interpreted as UTC (matching the ambient layer's
    ``_ensure_aware`` convention), NOT as host-local time -- which
    ``astimezone()`` on a naive datetime would otherwise silently assume,
    shifting the emitted timestamp by the server's UTC offset on non-UTC hosts
    whose backend (e.g. FalkorDB) returns naive datetimes.
    """
    aware = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).isoformat()


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None

    class Config:
        json_encoders = {datetime: _utc_iso}


class SearchResults(BaseModel):
    facts: list[FactResult]


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')


class Citation(BaseModel):
    """One admitted fact in an ambient injection block, with the signals a
    proactive agent needs to decide how much to trust and how to attribute it.

    Provenance fields (``supporting_quote``/``char_start``/``char_end``) are
    optional and null in Slice 1; they are populated once verbatim-quote
    provenance lands (Slice 3). Declaring them now freezes the response contract
    so hardening does not break the API.
    """

    # Required fields first, then optionals -- keeps pyright's synthesized
    # __init__ from mis-modelling the trailing optionals as required.
    edge_uuid: str
    fact: str
    rating: float = Field(description='Heuristic likelihood the fact is currently true, ~0-1')
    uncertainty: float = Field(description='How little recent corroboration; higher = shakier')
    episode_uuid: str | None = Field(
        default=None, description='The episode this fact was (first) extracted from'
    )
    reference_time: datetime | None = Field(
        default=None,
        description='Effective time of the fact (valid_at, else reference_time, else created_at)',
    )
    contested: bool = Field(
        default=False, description='True when the fact was surfaced but hedged (low confidence)'
    )
    relevance: float | None = Field(
        default=None, description='Cosine similarity of the fact to the transcript window'
    )
    why: str | None = Field(
        default=None, description='Legibility: the relation chain that surfaced this fact'
    )
    supporting_quote: str | None = Field(default=None, description='Verbatim source span (Slice 3)')
    char_start: int | None = Field(default=None, description='Start offset of the quote (Slice 3)')
    char_end: int | None = Field(default=None, description='End offset of the quote (Slice 3)')

    class Config:
        json_encoders = {datetime: _utc_iso}


class AmbientContextRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory namespace to draw from')
    transcript_window: str = Field(
        ...,
        description=(
            'The rolling window of the ongoing conversation to prime memory against. '
            'This is the retrieval cue -- NOT a user query. Keep it to the recent turns; '
            'it is embedded and BM25-matched as-is.'
        ),
    )
    token_budget: int | None = Field(
        default=None, ge=1, description='Max tokens of fact text in the injection block'
    )
    draw_limit: int | None = Field(
        default=None,
        ge=1,
        description='How many edges to over-fetch from search before node-diverse fill',
    )
    uncertainty_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description='Admitted facts must sit below this uncertainty'
    )
    min_rating: float = Field(
        default=0.0, ge=0.0, le=1.0, description='Admitted facts must sit at or above this rating'
    )
    min_top_relevance: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description='Block-level salience: if no fact is at least this cosine-similar to the window, '
        'the block is empty (the assistant stays silent on an off-topic conversation)',
    )
    min_relevance: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description='Per-fact salience floor: individual facts below this similarity are dropped '
        'from an emitted block',
    )


class GetContextRequest(BaseModel):
    group_id: str = Field(..., description='The memory namespace to draw from')
    query: str = Field(..., description='The question or retrieval cue')
    top_k: int = Field(default=40, ge=1, le=200, description='Fact search depth')
    max_episodes: int = Field(default=8, ge=1, le=20, description='Raw episodes to include')
    episode_char_budget: int = Field(
        default=48000, ge=2000, le=200000, description='Total char budget across episodes'
    )


class GetContextResponse(BaseModel):
    context: str = Field(..., description='Ready-to-inject reader context block')
    facts: list[str] = Field(default_factory=list)
    profiles: list[str] = Field(default_factory=list)
    episodes: list[str] = Field(default_factory=list)


class AmbientOutcomeRequest(BaseModel):
    trace_uuid: str = Field(..., description='The trace to label')
    outcome: Literal['engaged', 'dismissed', 'ignored'] = Field(
        ..., description='What the user did'
    )


class AmbientContextResponse(BaseModel):
    trace_uuid: str | None = Field(
        default=None, description='Decision-trace id for outcome labeling'
    )
    injection_block: str = Field(
        ..., description='Ready-to-inject text; [i] lines index into citations'
    )
    citations: list[Citation] = Field(
        default_factory=list, description='One entry per admitted fact, in block order'
    )
