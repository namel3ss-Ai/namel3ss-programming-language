"""CRUD parsing tests for the Namel3ss DSL."""

from __future__ import annotations

import pytest

from namel3ss.ast import CrudResource
from namel3ss.parser import N3SyntaxError, Parser


def test_parse_crud_minimal() -> None:
    source = (
        'app "Demo".\n'
        '\n'
        'enable crud for table orders.\n'
    )

    app = Parser(source).parse()

    assert len(app.crud_resources) == 1
    resource = app.crud_resources[0]
    assert isinstance(resource, CrudResource)
    assert resource.name == "orders"
    assert resource.source_type == "table"
    assert resource.source_name == "orders"
    assert resource.primary_key == "id"
    assert resource.select_fields == []
    assert resource.mutable_fields == []
    assert resource.allowed_operations == ["list", "retrieve", "create", "update", "delete"]
    assert resource.default_limit == 100
    assert resource.max_limit == 500
    assert resource.read_only is False


def test_parse_crud_with_options_block() -> None:
    source = (
        'app "Demo" connects to postgres "PRIMARY".\n'
        '\n'
        'enable crud for dataset latest_orders as "Customer Orders":\n'
        '  primary key: order_id\n'
        '  select: id, status, total\n'
        '  mutable: status, total\n'
        '  options:\n'
    '    read only: true\n'
    '    allow: read, update\n'
        '    deny: delete\n'
        '    tenant column: tenant_id\n'
        '    page size: 25\n'
        '    max limit: 250\n'
        '    label: Orders Portal\n'
    )

    app = Parser(source).parse()

    assert len(app.crud_resources) == 1
    resource = app.crud_resources[0]
    assert resource.name == "customer-orders"
    assert resource.label == "Orders Portal"
    assert resource.source_type == "dataset"
    assert resource.source_name == "latest_orders"
    assert resource.primary_key == "order_id"
    assert resource.select_fields == ["id", "status", "total"]
    assert resource.mutable_fields == ["status", "total"]
    assert resource.read_only is True
    assert resource.tenant_column == "tenant_id"
    assert resource.default_limit == 25
    assert resource.max_limit == 250
    assert resource.allowed_operations == ["list", "retrieve"]


def test_parse_crud_invalid_identifier() -> None:
    source = (
        'app "Demo".\n'
        '\n'
        'enable crud for table "bad name".\n'
    )

    with pytest.raises(N3SyntaxError):
        Parser(source).parse()
