from solution import roman_to_int


def test_easy():
    assert roman_to_int("III") == 3
    assert roman_to_int("X") == 10


def test_medium():
    assert roman_to_int("IV") == 4      # subtractive: a lookup/sum-only solution gets 6
    assert roman_to_int("XL") == 40     # subtractive: sum-only gets 60


def test_hard():
    assert roman_to_int("MCMXCIV") == 1994      # multiple subtractive groups (CM, XC, IV)
    assert roman_to_int("MMMCMXCIX") == 3999     # largest standard numeral, full logic
