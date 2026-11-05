# String annotations exercise the get_type_hints path in extract_function_schema.
from __future__ import annotations

from justllms.tools.utils import extract_function_schema, python_type_to_json_schema


def test_bare_containers_are_arrays():
    for t in (list, tuple, set):
        assert python_type_to_json_schema(t) == {"type": "array"}


def test_schema_types_and_required_flags():
    def search(query: str, ids: list[int], limit: int | None, *args, **kwargs):
        pass

    params = extract_function_schema(search)
    assert set(params) == {"query", "ids", "limit"}
    assert (params["query"].type, params["query"].required) == ("string", True)
    assert (params["ids"].type, params["ids"].items) == ("array", {"type": "integer"})
    assert (params["limit"].type, params["limit"].required) == ("integer", False)
