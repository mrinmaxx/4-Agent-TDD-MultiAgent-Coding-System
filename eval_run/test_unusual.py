"""
Tricky-only tests for the three unusual problems.
  rotate_rings(grid, k)
  allocate(requests, budget, min_each)
  collapse(tokens)
Imports match the saved filenames solution7/8/9.

NOTE: the 4x4 rotate_rings expected value was CORRECTED during test audit.
The original file had [[16,15,14,13],[5,11,10,9],[1,6,7,12],[2,3,4,8]], which is a
hand-computation error (it isn't a valid ring rotation). The correct clockwise
per-ring rotation — consistent with the 3x3 case — is
[[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]].
"""

from solution7 import rotate_rings
from solution8 import allocate
from solution9 import collapse


def check(label, got, expected):
    print(f"  [{'PASS' if got == expected else 'FAIL'}] {label}: got {got!r}, expected {expected!r}")


# ─────────────────────────────────────────────────────────────
# PROBLEM 1: rotate_rings(grid, k)  — clockwise, per-ring, k mod ring length
# ─────────────────────────────────────────────────────────────
print("=== Problem 1: rotate_rings (tricky) ===")

check("3x3 k=1 clockwise",
      rotate_rings([[1,2,3],[4,5,6],[7,8,9]], 1),
      [[4,1,2],[7,5,3],[8,9,6]])

check("3x3 k=8 (full turn, unchanged)",
      rotate_rings([[1,2,3],[4,5,6],[7,8,9]], 8),
      [[1,2,3],[4,5,6],[7,8,9]])

check("1x1 unchanged", rotate_rings([[7]], 5), [[7]])

# CORRECTED expected (see NOTE at top)
check("4x4 k=6 (rings rotate by different effective amounts)",
      rotate_rings([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], 6),
      [[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]])


# ─────────────────────────────────────────────────────────────
# PROBLEM 2: allocate(requests, budget, min_each)  — ordered rules
# ─────────────────────────────────────────────────────────────
print("\n=== Problem 2: allocate (tricky) ===")

check("min_each overrides small request",
      allocate([1,1,1], 10, 2), [2,2,2])

check("impossible (budget < mins)", allocate([5,5], 3, 2), [])

check("round-robin remainder skips satisfied",
      allocate([10,1,10], 7, 1), [3,1,3])

check("leftover discarded at caps", allocate([3,2], 10, 1), [3,2])


# ─────────────────────────────────────────────────────────────
# PROBLEM 3: collapse(tokens)  — stack machine with @ # ~ $
# ─────────────────────────────────────────────────────────────
print("\n=== Problem 3: collapse (tricky) ===")

check("@ order (b - a)", collapse(["5","3","@"]), 2)
check("@ with negatives", collapse(["-5","-2","@"]), -3)

check("~ median of three", collapse(["1","2","3","~"]), 2)
check("~ underflow does nothing", collapse(["7","~"]), 7)

check("$ collapse to sum", collapse(["1","2","3","$"]), 6)

check("@ on empty ignored -> 0", collapse(["@"]), 0)

check("# then @", collapse(["4","#","@"]), 0)

check("empty tokens", collapse([]), 0)

print("\nDone.")
