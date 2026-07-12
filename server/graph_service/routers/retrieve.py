import logging
from datetime import datetime, timezone

from fastapi import APIRouter, status

from graph_service.ambient import compose_ambient_block, cosine_similarity, self_uuid_for_group
from graph_service.config import ZepEnvDep
from graph_service.context_assembly import assemble_memory_context
from graph_service.dto import (
    AmbientContextRequest,
    AmbientContextResponse,
    GetContextRequest,
    GetContextResponse,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


@router.post('/get-context', status_code=status.HTTP_200_OK)
async def get_context(request: GetContextRequest, graphiti: ZepGraphitiDep):
    """Reactive memory context: the benchmark-proven three-channel assembly
    (validity-window facts + entity profiles + semantically-ranked raw episodes;
    retrieve-ALL facts for counting queries). Always answers -- no salience gate."""
    block, parts = await assemble_memory_context(
        graphiti,
        request.group_id,
        request.query,
        top_k=request.top_k,
        max_episodes=request.max_episodes,
        episode_char_budget=request.episode_char_budget,
    )
    return GetContextResponse(context=block, **parts)


@router.post('/get-ambient-context', status_code=status.HTTP_200_OK)
async def get_ambient_context(
    request: AmbientContextRequest,
    graphiti: ZepGraphitiDep,
    settings: ZepEnvDep,
):
    """Proactive-memory endpoint: a rolling transcript window in, a salience- and
    confidence-gated, token-budgeted, citation-carrying injection block out.

    Read-only -- reuses the request-scoped Graphiti client (finishes within the
    request, unlike the background /messages ingest). Runs the assembly walk in
    the server layer over stock ``graphiti.search`` results, gating each fact on
    its cosine relevance to the window so an off-topic conversation yields an
    empty block (the assistant stays silent).
    """
    window = request.transcript_window.strip()
    if not window:
        return AmbientContextResponse(injection_block='', citations=[])

    # Resolve request overrides against deployment defaults with `is not None`
    # (not `or`) so an explicit, in-range value is honored rather than treated
    # as unset. DTO bounds already reject 0/negative for the int knobs.
    token_budget = (
        request.token_budget
        if request.token_budget is not None
        else settings.ambient_default_token_budget
    )
    draw_limit = (
        request.draw_limit
        if request.draw_limit is not None
        else settings.ambient_default_draw_limit
    )
    uncertainty_threshold = (
        request.uncertainty_threshold
        if request.uncertainty_threshold is not None
        else settings.ambient_uncertainty_threshold
    )
    min_top_relevance = (
        request.min_top_relevance
        if request.min_top_relevance is not None
        else settings.ambient_min_top_relevance
    )
    min_relevance = (
        request.min_relevance
        if request.min_relevance is not None
        else settings.ambient_min_relevance
    )

    # Exclusion key only in Slice 1 (no self node exists yet); Slice 4 upserts it.
    self_uuid = self_uuid_for_group(request.group_id)

    edges = await graphiti.search(
        group_ids=[request.group_id],
        query=window,
        num_results=draw_limit,
    )

    # Salience: score each candidate fact against the window. graphiti.search
    # strips fact_embedding, so embed the window + the candidate facts in a single
    # batched call and cosine-score. Degrade gracefully (no salience gate) if the
    # embedder is unavailable rather than failing the request.
    relevance_by_uuid: dict[str, float] | None = None
    if edges:
        try:
            vectors = await graphiti.embedder.create_batch([window] + [e.fact for e in edges])
            window_vec, fact_vecs = vectors[0], vectors[1:]
            relevance_by_uuid = {
                edge.uuid: cosine_similarity(window_vec, fact_vec)
                for edge, fact_vec in zip(edges, fact_vecs, strict=False)
            }
        except Exception:  # noqa: BLE001 -- salience is best-effort; never fail the read
            logger.warning(
                'ambient salience scoring failed; proceeding without the gate', exc_info=True
            )

    injection_block, citations = compose_ambient_block(
        edges,
        token_budget=token_budget,
        now=datetime.now(timezone.utc),
        self_uuid=self_uuid,
        relevance_by_uuid=relevance_by_uuid,
        min_top_relevance=min_top_relevance,
        min_relevance=min_relevance,
        min_rating=request.min_rating,
        uncertainty_threshold=uncertainty_threshold,
        decay_per_day=settings.ambient_decay_per_day,
    )
    return AmbientContextResponse(injection_block=injection_block, citations=citations)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
