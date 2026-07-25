def length_of_longest_substring(s: str) -> int:
    """
    Compute the length of the longest substring without repeating characters.
    """
    char_set = set()
    max_length = 0
    left = 0  # window start
    
    for right in range(len(s)):  # extend window to the right
        while s[right] in char_set:
            char_set.remove(s[left])  # move window to the right
            left += 1
        
        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)
    
    return max_length

