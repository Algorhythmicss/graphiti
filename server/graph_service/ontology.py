"""Kyra's memory ontology: the entity and relationship types extracted from
ingested transcripts, emails, and messages.

Derived from EVIDENCE, not intuition (per memory-engine's schema-from-evidence
lesson). A first pass modelled Task/Commitment/Meeting as entity *types* and it
failed live: Graphiti's extractor is entity-centric -- nouns become nodes, and
ACTIONS become edges. The same messages that produced no Commitment node
produced exactly the right edges: ``PROMISED_TO_SEND`` "Alex promised Priya to
send the revised billing spec by Thursday", ``NEEDS_TO_REVIEW``,
``BOOKED_MEETING_WITH``. So actions are modelled here as EDGE types (with the
structured attributes Kyra acts on) and only genuine nouns as entity types.

- Entity types: Person, Organization, Project (the durable nouns).
- Edge types: COMMITMENT, TASK, MEETING (the action items) + a few relationship
  edges, each carrying due_date / status / counterparty-style attributes.

Type is a SIGNAL, not a dedup key: Graphiti's candidate search is label-agnostic
(node_operations.py uses an empty SearchFilters), so richer typing does not
fragment entities. All attributes are optional and dates are ISO-8601 strings so
a partially-specified message still yields a usable, typed edge.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# --- Entity types (durable nouns) ---------------------------------------------


class Person(BaseModel):
    """A human the user communicates with or refers to -- a colleague, client,
    manager, report, friend, or family member. Not the user themselves."""

    role: str | None = Field(
        default=None,
        description="The person's job title or role, e.g. 'billing lead', 'CEO'. "
        "NOT the word 'Person'. Omit if unknown.",
    )
    organization: str | None = Field(
        default=None, description='The organization or team this person belongs to'
    )
    relationship_to_user: str | None = Field(
        default=None,
        description="Their relationship to the user, e.g. 'manager', 'direct report', 'client', "
        "'co-founder', 'friend', 'spouse'",
    )


class Organization(BaseModel):
    """A company, team, vendor, client, or other organization."""

    kind: str | None = Field(
        default=None, description="Kind of organization, e.g. 'company', 'team', 'vendor', 'client'"
    )


class Project(BaseModel):
    """An ongoing initiative, workstream, or effort the user is involved in
    (e.g. a migration, a launch, a hiring round)."""

    status: str | None = Field(
        default=None, description="Current status, e.g. 'active', 'planned', 'blocked', 'done'"
    )


# --- Edge types (actions / relationships) -------------------------------------


class Commitment(BaseModel):
    """A promise from one party to another -- "I'll send the deck by Friday" (the
    user owes it) or "Dana said she'd approve the budget" (owed to the user).
    Source is who owes it, target is who it is owed to."""

    what: str | None = Field(default=None, description='What was promised')
    due_date: str | None = Field(
        default=None, description='When it is due, as an ISO-8601 date/datetime if determinable'
    )
    status: str | None = Field(
        default=None, description="One of 'open', 'fulfilled', 'broken' if indicated"
    )
    direction: str | None = Field(
        default=None,
        description="'i_owe' if the user (source) owes it, 'owed_to_me' if it is owed to the user",
    )


class Task(BaseModel):
    """An actionable to-do someone needs to do -- a next action, follow-up, or a
    thing to review/send. Source is who must do it, target is what it concerns."""

    what: str | None = Field(default=None, description='What needs to be done')
    due_date: str | None = Field(
        default=None, description='When it is due, as an ISO-8601 date/datetime if determinable'
    )
    status: str | None = Field(
        default=None, description="One of 'open', 'in_progress', 'done', 'blocked' if indicated"
    )
    priority: str | None = Field(
        default=None, description="One of 'low', 'medium', 'high' if indicated"
    )


class Meeting(BaseModel):
    """A scheduled meeting or call -- something on the calendar or being arranged.
    Connects the people or project involved."""

    scheduled_time: str | None = Field(
        default=None,
        description='When it is scheduled, as an ISO-8601 date/datetime if determinable',
    )
    channel: str | None = Field(
        default=None,
        description="Where it happens, e.g. 'Zoom', 'in person', 'phone', 'Google Meet'",
    )
    topic: str | None = Field(default=None, description='What the meeting is about')


class Preference(BaseModel):
    """A stated or implied preference, taste, or working style of a person -- e.g.
    "prefers async standups", "always books window seats", "prefers Slack over email".
    Source is who holds it, target is what it concerns."""

    strength: str | None = Field(
        default=None, description="One of 'stated' (explicit) or 'implied' (behavioral)"
    )


class WorksAt(BaseModel):
    """Employment or membership of a person in an organization."""

    role: str | None = Field(default=None, description='Their role there, if stated')


# Passed to graphiti.add_episode(entity_types=...). 'Entity' is Graphiti's
# reserved default (untyped) and must NOT appear here.
KYRA_ENTITY_TYPES: dict[str, type[BaseModel]] = {
    'Person': Person,
    'Organization': Organization,
    'Project': Project,
}

# Passed as edge_types=. Keys are the relationship labels the extractor uses.
KYRA_EDGE_TYPES: dict[str, type[BaseModel]] = {
    'COMMITMENT': Commitment,
    'TASK': Task,
    'MEETING': Meeting,
    'WORKS_AT': WorksAt,
    'PREFERENCE': Preference,
}

# Passed as edge_type_map=. Maps (source_type, target_type) -> allowed edge types
# for that pair. 'Entity' is the default type, so pairs involving it catch cases
# where an endpoint wasn't classified into a custom type (e.g. the user, or an
# unnamed thing). Broad on purpose: the extractor still creates untyped edges for
# anything unmapped, so this only governs which edges get STRUCTURED attributes.
KYRA_EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ('Person', 'Person'): ['COMMITMENT', 'MEETING'],
    ('Person', 'Organization'): ['WORKS_AT', 'COMMITMENT'],
    ('Person', 'Project'): ['TASK', 'COMMITMENT', 'MEETING'],
    ('Person', 'Entity'): ['COMMITMENT', 'TASK', 'PREFERENCE'],
    ('Entity', 'Person'): ['COMMITMENT', 'MEETING'],
    ('Entity', 'Organization'): ['WORKS_AT', 'COMMITMENT'],
    ('Entity', 'Project'): ['TASK', 'COMMITMENT', 'MEETING'],
    ('Entity', 'Entity'): ['COMMITMENT', 'TASK', 'PREFERENCE'],
}
