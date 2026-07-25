import random
from solution import length_of_longest_substring

def _reference(s: str) -> int:
    """Simple brute-force reference function"""
    max_length = 0
    for i in range(len(s)):
        char_set = set()
        for j in range(i, len(s)):
            if s[j] in char_set:
                break
            char_set.add(s[j])
            max_length = max(max_length, j - i + 1)
    return max_length

def test_fuzz():
    random.seed(0)
    for _ in range(50):
        # generate random input within small bounds (max 10 characters, 50% chance of repeating character)
        max_len = random.randint(1, 10)
        s = ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFabcdef') for _ in range(max_len))
        if random.random() < 0.5:
            # repeat a random character
            repeat_char = random.choice(s)
            s += repeat_char * random.randint(1, max_len)
        # build random args within small bounds
        assert length_of_longest_substring(s) == _reference(s)

