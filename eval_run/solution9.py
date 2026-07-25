def collapse(tokens: list[str]) -> int:
    """
    Process a list of string tokens left to right using a stack of integers.
    
    Rules:
    - A token that is a (possibly negative) integer string -> push its value.
    - "@"  -> pop the top TWO values a (top) and b (second), push (b - a).
    - "#"  -> pop the top value v, push v then v again (duplicate the top).
    - "~"  -> pop the top THREE values and push back only their MEDIAN (middle value by sort).
              If fewer than three values are present, "~" does nothing.
    - "$"  -> collapse the ENTIRE current stack into a single value equal to the sum of all values on the stack,
              leaving just that one value.
    
    If an operator cannot be applied because there aren't enough operands (except "~" and "$", whose under-flow 
    behavior is defined above), that operator is IGNORED.
    
    After processing all tokens, return the value on TOP of the stack, or 0 if the stack is empty.
    
    Parameters:
    tokens (list[str]): A list of string tokens to be processed.
    
    Returns:
    int: The value on top of the stack after processing all tokens, or 0 if the stack is empty.
    """
    stack = []
    
    for token in tokens:
        if token.lstrip('-').isdigit():
            stack.append(int(token))
        elif token == "@":
            if len(stack) >= 2:
                a = stack.pop()
                b = stack.pop()
                stack.append(b - a)
        elif token == "#":
            if stack:
                v = stack.pop()
                stack.append(v)
                stack.append(v)
        elif token == "~":
            if len(stack) >= 3:
                v1 = stack.pop()
                v2 = stack.pop()
                v3 = stack.pop()
                median = sorted([v1, v2, v3])[1]
                stack.append(median)
        elif token == "$":
            if stack:
                total = sum(stack)
                stack = [total]
    
    return stack[-1] if stack else 0
