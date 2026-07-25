def add_strings(num1: str, num2: str) -> str:
    """
    Returns the sum of two non-negative integers represented as strings.
    
    This function simulates digit-by-digit addition with carry, handling strings of any length.
    
    Parameters:
    num1 (str): The first non-negative integer as a string.
    num2 (str): The second non-negative integer as a string.
    
    Returns:
    str: The sum of num1 and num2 as a string, with no leading zeros except for the number "0" itself.
    """
    result = []
    carry = 0
    p, q = len(num1) - 1, len(num2) - 1
    
    while p >= 0 or q >= 0:
        x = ord(num1[p]) - ord('0') if p >= 0 else 0
        y = ord(num2[q]) - ord('0') if q >= 0 else 0
        sum = x + y + carry
        result.append(str(sum % 10))
        carry = sum // 10
        p -= 1
        q -= 1
        
    if carry:
        result.append(str(carry))
        
    return ''.join(reversed(result)).lstrip('0') or '0'

