def allocate(requests: list[int], budget: int, min_each: int) -> list[int]:
    """
    Distribute tokens among clients based on their requests and a given budget.

    Args:
        requests (list[int]): A list of token requests from each client.
        budget (int): The total number of tokens available for distribution.
        min_each (int): The minimum number of tokens each client must receive.

    Returns:
        list[int]: The final allocation of tokens to each client.
    """
    # Calculate the total minimum tokens required
    total_min_tokens = len(requests) * min_each
    
    # If the budget is less than the total minimum tokens, return an empty list
    if budget < total_min_tokens:
        return []
    
    # Initialize the allocation list with the minimum tokens for each client
    allocation = [min_each] * len(requests)
    
    # Calculate the remaining budget after allocating the minimum tokens
    remaining_budget = budget - total_min_tokens
    
    # Initialize an index to keep track of the current client in the round-robin distribution
    client_index = 0
    
    # Continue distributing tokens until the budget runs out or no client wants more
    while remaining_budget > 0:
        # Check if the current client wants more tokens
        if allocation[client_index] < requests[client_index]:
            # Allocate one more token to the current client
            allocation[client_index] += 1
            # Decrement the remaining budget
            remaining_budget -= 1
        # Move to the next client in the round-robin distribution
        client_index = (client_index + 1) % len(requests)
    
    return allocation
