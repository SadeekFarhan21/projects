import ast
import random

from agentenv.families.mutations import OPERATORS, enumerate_mutants

SRC = '''
def clamp(x, lo, hi):
    """Clamp x < 3 into [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    total = x + 1
    return total - 1


def first(items):
    for i in range(len(items)):
        if items[i] == 0:
            return i
    return -1
'''


def test_all_operators_and_exact_revert():
    muts = enumerate_mutants(SRC, random.Random(0))
    ops = {m.operator for m in muts}
    assert ops == set(OPERATORS)
    for m in muts:
        mutated = m.apply(SRC)
        assert mutated != SRC
        ast.parse(mutated)
        # the inverse edit restores the original exactly
        restored = mutated[:m.offset] + m.old + mutated[m.offset + len(m.new):]
        assert restored == SRC


def test_docstring_untouched_and_single_span():
    for m in enumerate_mutants(SRC):
        assert "Clamp" not in m.old
        mutated = m.apply(SRC)
        diff = [i for i, (a, b) in enumerate(zip(SRC.splitlines(), mutated.splitlines())) if a != b]
        if m.operator != "dropped_branch":
            assert len(diff) == 1


def test_flip_examples():
    flips = {(m.old, m.new) for m in enumerate_mutants(SRC) if m.operator == "flip_comparison"}
    assert ("<", "<=") in flips and (">", ">=") in flips and ("==", "!=") in flips
