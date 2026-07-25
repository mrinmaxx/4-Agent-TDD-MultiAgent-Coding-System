from typing import List, Tuple

def max_profit(jobs: List[Tuple[int, int, int]], cooldown: int) -> int:
    """
    This function calculates the maximum achievable total profit for the given jobs and cooldown period.
    
    It sorts the jobs based on their end times and uses dynamic programming to store the maximum profit 
    that can be achieved up to each job, considering the cooldown period.
    
    Time complexity: O(N log N) due to the sorting step.
    """
    
    # Handle edge case where the list of jobs is empty
    if not jobs:
        return 0
    
    # Sort jobs based on their end times
    jobs.sort(key=lambda x: x[1])
    
    # Initialize a list to store the maximum profit up to each job
    max_profits = [0] * len(jobs)
    max_profits[0] = jobs[0][2]
    
    # Initialize a variable to store the maximum profit so far
    max_profit_so_far = jobs[0][2]
    
    # Iterate through the sorted jobs
    for i in range(1, len(jobs)):
        # Initialize the maximum profit including the current job
        current_max_profit = jobs[i][2]
        
        # Find the last job that does not conflict with the current job, considering the cooldown period
        for j in range(i - 1, -1, -1):
            if jobs[j][1] + cooldown <= jobs[i][0]:
                current_max_profit = max(current_max_profit, max_profits[j] + jobs[i][2])
                break
        
        # Update the maximum profit up to the current job
        max_profits[i] = max(current_max_profit, max_profits[i - 1])
        
        # Update the maximum profit so far
        max_profit_so_far = max(max_profit_so_far, max_profits[i])
    
    # Return the maximum achievable total profit
    return max_profit_so_far
