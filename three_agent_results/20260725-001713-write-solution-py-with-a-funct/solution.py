def count_vowels(s: str) -> int:
    """
    This function counts the number of vowels (a, e, i, o, u) in a given string.
    
    Parameters:
    s (str): The input string to count vowels from.
    
    Returns:
    int: The total count of vowels in the string.
    """
    # Initialize a counter variable to store the count of vowels
    vowel_count = 0
    
    # Convert the string to lowercase to handle case-insensitive comparison
    s = s.lower()
    
    # Iterate over each character in the string
    for char in s:
        # Check if the character is a vowel
        if char in 'aeiou':
            # If it's a vowel, increment the vowel count
            vowel_count += 1
    
    # Return the total count of vowels
    return vowel_count
