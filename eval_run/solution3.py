def num_decodings(s: str) -> int:
    """
    Calculate the number of distinct ways a given digit string can be decoded back into letters.

    Args:
    s (str): The digit string to be decoded.

    Returns:
    int: The number of distinct ways the digit string can be decoded.
    """
    # Initialize a dynamic programming table of size len(s) + 1
    dp = [0] * (len(s) + 1)
    
    # The empty string can be decoded in one way
    dp[0] = 1
    
    # Iterate through the string from the second character to the end
    for i in range(1, len(s) + 1):
        # If the current character is not zero, it can be decoded as a single letter
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        
        # If the last two characters form a valid group, they can be decoded together
        if i >= 2 and '10' <= s[i-2:i] <= '26':
            dp[i] += dp[i-2]
    
    # Return the number of ways to decode the entire string
    return dp[-1]
