from solution import run_length_encode


def test_easy():
    assert run_length_encode("aaab") == "a3b1"
    assert run_length_encode("abc") == "a1b1c1"


def test_medium():
    assert run_length_encode("") == ""                  # empty input
    assert run_length_encode("aaaaaaaaaa") == "a10"     # multi-digit count (> 9)


def test_hard():
    assert run_length_encode("aAaA") == "a1A1a1A1"      # case-sensitive: no merging across case
    assert run_length_encode("aabbaa") == "a2b2a2"       # non-adjacent runs stay separate (no merge)
