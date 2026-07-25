from solution import is_balanced


def test_easy():
    assert is_balanced("()")
    assert is_balanced("([])")


def test_medium():
    assert is_balanced("")            # empty string is balanced
    assert not is_balanced("(]")      # wrong-type match


def test_hard():
    assert not is_balanced("([)]")            # equal counts but interleaved / wrong nesting
    assert is_balanced("a(b[c]{d}e)f")        # balanced despite non-bracket characters mixed in
