"""Run each test case in its own subprocess with a timeout, so an infinite loop in
one solution is reported as HANG instead of blocking the whole suite."""
import subprocess

# (problem, label, module-qualified call expression, expected object)
CASES = [
    (1, "3x3 k=1 clockwise", "solution7.rotate_rings([[1,2,3],[4,5,6],[7,8,9]], 1)", [[4,1,2],[7,5,3],[8,9,6]]),
    (1, "3x3 k=8 full turn", "solution7.rotate_rings([[1,2,3],[4,5,6],[7,8,9]], 8)", [[1,2,3],[4,5,6],[7,8,9]]),
    (1, "1x1 unchanged", "solution7.rotate_rings([[7]], 5)", [[7]]),
    (1, "4x4 k=6 (corrected expected)", "solution7.rotate_rings([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], 6)", [[16,15,14,13],[12,11,10,9],[8,7,6,5],[4,3,2,1]]),

    (2, "min_each overrides small request", "solution8.allocate([1,1,1], 10, 2)", [2,2,2]),
    (2, "impossible (budget < mins)", "solution8.allocate([5,5], 3, 2)", []),
    (2, "round-robin remainder skips satisfied", "solution8.allocate([10,1,10], 7, 1)", [3,1,3]),
    (2, "leftover discarded at caps", "solution8.allocate([3,2], 10, 1)", [3,2]),

    (3, "@ order (b - a)", "solution9.collapse(['5','3','@'])", 2),
    (3, "@ with negatives", "solution9.collapse(['-5','-2','@'])", -3),
    (3, "~ median of three", "solution9.collapse(['1','2','3','~'])", 2),
    (3, "~ underflow does nothing", "solution9.collapse(['7','~'])", 7),
    (3, "$ collapse to sum", "solution9.collapse(['1','2','3','$'])", 6),
    (3, "@ on empty ignored -> 0", "solution9.collapse(['@'])", 0),
    (3, "# then @", "solution9.collapse(['4','#','@'])", 0),
    (3, "empty tokens", "solution9.collapse([])", 0),
]

results = []  # (problem, status, label, detail)
for prob, label, call, expected in CASES:
    mod = call.split(".")[0]
    code = f"import {mod}; import sys; sys.stdout.write(repr({call}))"
    try:
        r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=8)
        if r.returncode != 0:
            err = (r.stderr.strip().splitlines() or ["?"])[-1]
            results.append((prob, "FAIL", label, f"ERROR: {err}"))
        else:
            got = r.stdout.strip()
            ok = got == repr(expected)
            results.append((prob, "PASS" if ok else "FAIL", label,
                            "" if ok else f"got {got}, expected {expected!r}"))
    except subprocess.TimeoutExpired:
        results.append((prob, "HANG", label, "infinite loop / timed out after 8s"))

for prob, status, label, detail in results:
    print(f"P{prob} [{status}] {label}" + (f": {detail}" if detail else ""))
