import json
from solution import is_palindrome

cases = [
    # (args_tuple, expected)
    ("A man, a plan, a canal: Panama", True),
    ("Not a palindrome", False),
    ("race a car", False),
    ("No 'x' in Nixon", True),
    ("Was it a car or a cat I saw?", True),
    ("Able was I ere I saw Elba", True),
    ("A Santa at NASA", True),
    ("", True),
    ("a", True),
    ("ab", False),
]
failures, passed = [], 0
for i, (args, expected) in enumerate(cases):
    try:
        actual = is_palindrome(args)
    except Exception as e:
        actual = f"EXCEPTION: {e}"
    if actual == expected:
        passed += 1
    else:
        failures.append({"case": f"case{i}", "input": repr(args),
                         "expected": repr(expected), "actual": repr(actual),
                         "reason": "wrong output" if "EXCEPTION" not in str(actual) else "raised error"})
report = {"all_passed": not failures, "total": len(cases), "passed": passed, "failures": failures}
json.dump(report, open("test_report.json", "w"), indent=2)
print(report)
