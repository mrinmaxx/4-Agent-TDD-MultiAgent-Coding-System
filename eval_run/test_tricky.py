"""
Tricky edge-case tests for:
  solution4py   -> is_match(s, p)          (wildcard matching)
  solution5py  -> add_strings(num1, num2) (string addition)
  solution6.py  -> num_decodings(s)        (decode ways)
Run: python test_tricky.py
"""

from solution import is_match
from solution2 import add_strings
from solution3 import num_decodings


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"  [{status}] {label}: got {got!r}, expected {expected!r}")


# ─────────────────────────────────────────────────────────────
# PROBLEM 1: is_match(s, p)  — wildcard '?' and '*'
# ─────────────────────────────────────────────────────────────
print("=== Problem 1: is_match (Wildcard Matching) ===")

# EASY / basic
check("exact match", is_match("abc", "abc"), True)
check("single ? ", is_match("abc", "a?c"), True)

# MEDIUM / the * traps
check("star matches empty (empty s)", is_match("", "*"), True)
check("empty vs empty", is_match("", ""), True)
check("nonempty vs empty pattern", is_match("a", ""), False)
check("star matches everything", is_match("aa", "*"), True)
check("? cannot match empty", is_match("", "?"), False)
check("wrong single char", is_match("cb", "?a"), False)

# HARD / multi-star interaction + backtracking
check("multi-star middle", is_match("adceb", "*a*b"), True)
check("mixed ? and * fails", is_match("acdcb", "a*c?b"), False)
# pathological: exponential solutions will HANG here; correct DP returns fast
check("pathological no-match (must not hang)",
      is_match("aaaaaaaaaab", "a*a*a*a*a*a*a*a*a*a*c"), False)


# ─────────────────────────────────────────────────────────────
# PROBLEM 2: add_strings(num1, num2)  — string addition, no int()
# ─────────────────────────────────────────────────────────────
print("\n=== Problem 2: add_strings (Arbitrary-Precision String Add) ===")

# EASY
check("simple", add_strings("11", "123"), "134")
check("both zero", add_strings("0", "0"), "0")

# MEDIUM / carry + unequal length + leading-zero handling
check("carry adds a digit", add_strings("999", "1"), "1000")
check("carry, reversed lengths", add_strings("1", "999"), "1000")
check("unequal lengths", add_strings("456", "77"), "533")
check("zero plus number", add_strings("0", "12345"), "12345")

# HARD / big numbers (would overflow 64-bit int in other langs)
check("20-digit carry ripple",
      add_strings("99999999999999999999", "1"),
      "100000000000000000000")
check("two large numbers",
      add_strings("123456789012345678901234567890",
                  "987654321098765432109876543210"),
      "1111111110111111111011111111100")


# ─────────────────────────────────────────────────────────────
# PROBLEM 3: num_decodings(s)  — decode ways, zero handling
# ─────────────────────────────────────────────────────────────
print("\n=== Problem 3: num_decodings (Decode Ways) ===")

# EASY
check("simple two ways", num_decodings("12"), 2)
check("single digit", num_decodings("8"), 1)

# MEDIUM / classic + basic zero rules
check("226 -> 3 ways", num_decodings("226"), 3)
check("leading zero invalid", num_decodings("0"), 0)
check("06 invalid group", num_decodings("06"), 0)
check("10 -> only J", num_decodings("10"), 1)

# HARD / nasty zero placement
check("100 -> impossible", num_decodings("100"), 0)
check("27 -> only 2-7 (27>26)", num_decodings("27"), 1)
check("2101 -> 1 way", num_decodings("2101"), 1)
check("2020 -> ?", num_decodings("2020"), 1)

print("\nDone.")
