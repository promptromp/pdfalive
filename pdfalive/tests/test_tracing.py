"""The test suite must not send LangSmith traces, whatever the developer's environment."""

from langsmith import utils as langsmith_utils


def test_tracing_is_disabled_for_the_test_session():
    """Asserts on langsmith's own decision rather than calling a traced function (which would post if broken)."""
    assert langsmith_utils.tracing_is_enabled() is False
