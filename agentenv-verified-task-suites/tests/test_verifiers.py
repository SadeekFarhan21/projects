import pytest

from agentenv.task import Task
from agentenv.verifiers import VerifyContext, normalize_text, numeric_or_text_answer, parse_number


@pytest.mark.parametrize("x,want", [
    (12, 12.0), ("12.5", 12.5), ("$1,234.50", 1234.5), ("about -3.2 minutes", -3.2),
    ([7], 7.0), ("1e3", 1000.0), ("none", None), (None, None), (True, None), (float("nan"), None),
])
def test_parse_number(x, want):
    assert parse_number(x) == want


def ctx(sub, submitted=True):
    t = Task.from_dict(dict(id="t", family="sql", prompt="p", fixtures=[], allowed_tools=["submit"],
                            verifier={"type": "python", "function": "m:f"}))
    return VerifyContext(t, None, sub, submitted)


def test_numeric_tolerance():
    assert numeric_or_text_answer(ctx("100.004"), 100.0).passed
    assert not numeric_or_text_answer(ctx("100.5"), 100.0).passed
    assert numeric_or_text_answer(ctx("1000100"), 1000000, rel_tol=1e-4).passed
    assert not numeric_or_text_answer(ctx(""), 1.0).passed
    assert not numeric_or_text_answer(ctx(None, submitted=False), 1.0).passed


def test_text_answers():
    assert normalize_text("  'Iron  Maiden' ") == "iron maiden"
    assert numeric_or_text_answer(ctx("iron maiden"), "Iron Maiden", kind="text").passed
    assert not numeric_or_text_answer(ctx("Metallica"), "Iron Maiden", kind="text").passed
