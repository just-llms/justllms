from types import SimpleNamespace

import pytest

from justllms.routing.router import Router


def _provider(*models):
    return SimpleNamespace(validate_model=lambda m: m in models)


def test_route_model_id_containing_slash():
    providers = {"ollama": _provider("hf.co/user/model:q4")}
    assert Router().route([], model="hf.co/user/model:q4", providers=providers) == (
        "ollama",
        "hf.co/user/model:q4",
    )


def test_route_provider_prefix_still_works():
    providers = {"ollama": _provider("hf.co/user/model")}
    assert Router().route([], model="ollama/hf.co/user/model", providers=providers) == (
        "ollama",
        "hf.co/user/model",
    )


def test_route_unknown_slash_model_raises():
    with pytest.raises(ValueError, match="not found in any available provider"):
        Router().route([], model="nope/x", providers={"ollama": _provider("llama3")})
