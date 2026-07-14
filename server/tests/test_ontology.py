"""Validation tests for Kyra's entity + edge ontology.

Pure and DB-free: confirms the ontology is well-formed and accepted by Graphiti
(no attribute shadows a reserved field, no reserved 'Entity' key), that every
type carries the docstring the extractor classifies on, and that the edge type
map only references defined edge types over valid entity-type endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel

from graph_service.ontology import (
    KYRA_EDGE_TYPE_MAP,
    KYRA_EDGE_TYPES,
    KYRA_ENTITY_TYPES,
)


def test_entity_types_pass_graphiti_validation():
    from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types

    assert validate_entity_types(KYRA_ENTITY_TYPES) is True


def test_all_types_are_basemodels_with_classification_docstrings():
    for registry in (KYRA_ENTITY_TYPES, KYRA_EDGE_TYPES):
        for name, model in registry.items():
            assert isinstance(name, str) and name
            assert issubclass(model, BaseModel)
            assert model.__doc__ and model.__doc__.strip(), f'{name} needs a docstring'


def test_reserved_default_type_not_redefined():
    assert 'Entity' not in KYRA_ENTITY_TYPES


def test_actions_are_edge_types_not_entity_types():
    # Evidence-driven: Graphiti extracts actions as EDGES, so commitments/tasks/
    # meetings live in the edge registry, never as entity types.
    for action in ('COMMITMENT', 'TASK', 'MEETING'):
        assert action in KYRA_EDGE_TYPES
        assert action not in KYRA_ENTITY_TYPES
    for noun in ('Person', 'Organization', 'Project'):
        assert noun in KYRA_ENTITY_TYPES


def test_edge_type_map_references_only_defined_types():
    valid_endpoints = set(KYRA_ENTITY_TYPES) | {'Entity'}
    for (src, tgt), edge_names in KYRA_EDGE_TYPE_MAP.items():
        assert src in valid_endpoints, f'unknown source type {src}'
        assert tgt in valid_endpoints, f'unknown target type {tgt}'
        for edge_name in edge_names:
            assert edge_name in KYRA_EDGE_TYPES, f'edge_type_map references undefined {edge_name}'


def test_all_attributes_are_optional():
    for registry in (KYRA_ENTITY_TYPES, KYRA_EDGE_TYPES):
        for name, model in registry.items():
            for field_name, field in model.model_fields.items():
                assert not field.is_required(), f'{name}.{field_name} must be optional'
