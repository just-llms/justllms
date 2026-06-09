"""Tests for token counter encoding selection."""

from justllms.utils.token_counter import TokenCounter


def test_encoding_uses_longest_matching_prefix() -> None:
    """Versioned model IDs should match the most specific prefix."""
    counter = TokenCounter()

    gpt4o_encoding = counter._get_encoding("gpt-4o-2024-08-06")
    gpt4_encoding = counter._get_encoding("gpt-4-turbo-2024-04-09")
    gpt5_encoding = counter._get_encoding("gpt-5-mini-2025-01-01")

    assert gpt4o_encoding is not None
    assert gpt4_encoding is not None
    assert gpt5_encoding is not None

    assert gpt4o_encoding.name == "o200k_base"
    assert gpt4_encoding.name == "cl100k_base"
    assert gpt5_encoding.name == "o200k_base"
