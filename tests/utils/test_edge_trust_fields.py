"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from datetime import datetime, timezone

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import get_entity_edge_from_record
from graphiti_core.helpers import coalesce

NOW = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)


def _record(**overrides):
    rec = dict(
        uuid='e1',
        source_node_uuid='a',
        target_node_uuid='b',
        fact='f',
        name='REL',
        group_id='g',
        episodes=['ep1'],
        created_at=NOW,
        expired_at=None,
        valid_at=None,
        invalid_at=None,
        reference_time=None,
        attributes={},
    )
    rec.update(overrides)
    return rec


def test_coalesce_preserves_falsy_zero_and_false():
    assert coalesce(None, 0.75) == 0.75
    assert coalesce(0.0, 0.75) == 0.0  # a legit low rating must NOT be replaced
    assert coalesce(False, True) is False
    assert coalesce(0, 1) == 0


def test_legacy_record_missing_trust_fields_gets_defaults():
    # An edge written before the columns existed: keys absent / NULL.
    edge = get_entity_edge_from_record(_record(), GraphProvider.NEO4J)
    assert edge.confidence_rating == 0.75
    assert edge.confidence_uncertainty == 0.5
    assert edge.corroboration_count == 1
    assert edge.confirmed is False
    assert edge.expires_at is None
    assert edge.confidence_last_touched_at is None


def test_trust_fields_are_read_and_removed_from_attributes():
    rec = _record(
        confidence_rating=0.42,
        confidence_uncertainty=0.11,
        corroboration_count=3,
        confirmed=True,
        expires_at=NOW,
        confidence_last_touched_at=NOW,
        # properties(e) returns everything, including the trust columns + custom attrs
        attributes={'confidence_rating': 0.42, 'confirmed': True, 'what': 'send spec'},
    )
    edge = get_entity_edge_from_record(rec, GraphProvider.NEO4J)
    assert edge.confidence_rating == 0.42
    assert edge.confidence_uncertainty == 0.11
    assert edge.corroboration_count == 3
    assert edge.confirmed is True
    assert edge.expires_at is not None
    assert edge.confidence_last_touched_at is not None
    # trust columns popped out of the generic attributes bag; custom attrs preserved
    assert 'confidence_rating' not in edge.attributes
    assert 'confirmed' not in edge.attributes
    assert edge.attributes.get('what') == 'send spec'


def test_zero_rating_from_db_is_preserved_not_defaulted():
    edge = get_entity_edge_from_record(_record(confidence_rating=0.0), GraphProvider.NEO4J)
    assert edge.confidence_rating == 0.0
