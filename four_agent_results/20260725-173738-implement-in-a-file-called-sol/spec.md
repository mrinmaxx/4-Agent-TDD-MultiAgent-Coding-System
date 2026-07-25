Function: def length_of_longest_substring(s: str) -> int
Summary: Compute the length of the longest substring without repeating characters.
Invariants:
- The input string `s` is a sequence of characters (letters, digits, symbols, and spaces).
- The output is a non-negative integer (0 or greater).
Edge cases:
- Empty input string (`s == ""`)
- Single character input string (`s == "c"`)
- No repeating characters in the input string (`s == "abcd"`)
- Repeating characters in the input string (`s == "abcabcbb"`)
- String with only one unique character repeated (`s == "bbbbb"`)
- Long string with repeating characters (`s == "pwwkew"` length 10^4)
- String containing non-ASCII characters (`s == ""` )
Brute-force verifier: For each potential substring length, try all possible substrings of that length, checking if each has repeating characters.
