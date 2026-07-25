"""
Manual test cases for solution.py (max_profit),
solution2.py (min_semesters), and solution3.py (min_cost_drive).
Each function prints PASS/FAIL per case. Tricky/adversarial cases included.
"""

from solution import max_profit
from solution2 import min_semesters
from solution3 import min_cost_drive


def check(label, got, expected):
    status = "PASS" if got == expected else "FAIL"
    print(f"  [{status}] {label}: got {got}, expected {expected}")


# ─────────────────────────────────────────────────────────────
# PROBLEM 1: max_profit(jobs, cooldown)   jobs = (start, end, profit)
# ─────────────────────────────────────────────────────────────
print("=== Problem 1: max_profit (Weighted Job Scheduling + Cooldown) ===")

# EASY
check("empty list", max_profit([], 0), 0)
check("single job", max_profit([(0, 5, 10)], 0), 10)

# MEDIUM
# touching endpoints, cooldown 0 -> both allowed
check("touching endpoints, C=0", max_profit([(0, 5, 10), (5, 10, 20)], 0), 30)
# gap of 1 < cooldown 2 -> can only take one; must pick higher profit
check("cooldown blocks second, pick higher",
      max_profit([(0, 5, 10), (6, 10, 20)], 2), 20)

# HARD
# greedy trap: taking the fat early job beats three small ones
check("greedy trap (DP not greedy)",
      max_profit([(0, 10, 100), (0, 2, 30), (3, 5, 30), (6, 8, 30)], 0), 100)
# cooldown forces skipping; optimal picks non-adjacent high earners
# jobs: A(0-2,50) B(3-5,50) C(6-8,50) with cooldown 3
#   A ends 2, next must start >=5 -> B starts 3 (blocked), C starts 6 (ok) => A+C=100
#   or just B alone=50; best is A(0-2)+C(6-8)=100  (A end2 +3 =5 <=6 ok)
check("cooldown chain optimization",
      max_profit([(0, 2, 50), (3, 5, 50), (6, 8, 50)], 3), 100)


# ─────────────────────────────────────────────────────────────
# PROBLEM 2: min_semesters(num_courses, prerequisites)
# ─────────────────────────────────────────────────────────────
print("\n=== Problem 2: min_semesters (Course Schedule / Min Semesters) ===")

# EASY
check("zero courses", min_semesters(0, []), 0)
check("no prerequisites -> 1 semester", min_semesters(3, []), 1)

# MEDIUM
# straight chain must be sequential
check("chain 0->1->2", min_semesters(3, [[0, 1], [1, 2]]), 3)
# simple cycle -> impossible
check("cycle -> -1", min_semesters(2, [[0, 1], [1, 0]]), -1)

# HARD
# diamond: 0 -> {1,2} -> 3 ; 1 and 2 share a semester => 3 total
check("diamond dependency", min_semesters(4, [[0, 1], [0, 2], [1, 3], [2, 3]]), 3)
# wide+shallow: one root, four dependents -> 2 semesters (root, then all four)
# PLUS an isolated course that has no deps (still fits in semester 1)
check("wide shallow + isolated node",
      min_semesters(6, [[0, 1], [0, 2], [0, 3], [0, 4]]), 2)


# ─────────────────────────────────────────────────────────────
# PROBLEM 3: min_cost_drive(n, roads, fuel_stations, capacity, start, target)
# roads = [u, v, dist] ; fuel_stations = {city: cost}
# ─────────────────────────────────────────────────────────────
print("\n=== Problem 3: min_cost_drive (Constrained Shortest Path + Fuel) ===")

# EASY
check("start == target", min_cost_drive(1, [], {}, 10, 0, 0), 0)
# reachable on a full tank, no refuel needed -> cost 0 (even though a station exists)
check("reachable without refuel",
      min_cost_drive(2, [[0, 1, 5]], {0: 100}, 10, 0, 1), 0)

# MEDIUM
# tank=5, path 0->1->2 each dist 4. Full tank=5 covers 0->1 (fuel left 1, can't do next 4).
# Must refuel at city 1 (cost 7). Then 1->2. => cost 7
check("must refuel once",
      min_cost_drive(3, [[0, 1, 4], [1, 2, 4]], {1: 7}, 5, 0, 2), 7)
# unreachable: no path to target
check("no path -> -1",
      min_cost_drive(3, [[0, 1, 2]], {}, 10, 0, 2), -1)

# HARD
# single road longer than capacity -> that road unusable, target unreachable -> -1
check("road longer than tank -> -1",
      min_cost_drive(2, [[0, 1, 12]], {0: 5}, 10, 0, 1), -1)
# cost-vs-distance trap:
#   Route A (short but expensive refuel): 0->1(dist5, refuel@1 cost 50)->2(dist5)  total cost 50
#   Route B (longer but cheap refuel):   0->3(dist5, refuel@3 cost 5)->2(dist5)     total cost 5
#   tank=5 forces a refuel at the middle city on either route.
#   Minimizing COST must choose Route B (5), NOT the distance-equal Route A.
check("min COST not distance",
      min_cost_drive(
          4,
          [[0, 1, 5], [1, 2, 5], [0, 3, 5], [3, 2, 5]],
          {1: 50, 3: 5},
          5,
          0,
          2,
      ), 5)

print("\nDone.")
