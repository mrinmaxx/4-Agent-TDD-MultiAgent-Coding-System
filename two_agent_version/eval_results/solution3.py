import heapq

def min_cost_drive(
    n: int,
    roads: list[list[int]],
    fuel_stations: dict[int, int],   # maps city -> refuel cost at that city
    capacity: int,
    start: int,
    target: int,
) -> int:
    """
    This function calculates the minimum total refueling cost required to get from the start city to the target city.
    
    The state is represented as (city, fuel_remaining), which models the current city and the remaining fuel in the tank.
    Dijkstra's algorithm is used to find the shortest path in the state space, considering fuel constraints and refueling options at fuel stations.
    
    The time complexity of this approach is O((N + M) log N), where N is the number of cities and M is the number of roads.

    Args:
        n (int): The number of cities.
        roads (list[list[int]]): A list of roads, where each road is given as [u, v, dist] meaning a one-way road from city u to city v of length dist.
        fuel_stations (dict[int, int]): A dictionary mapping city to refuel cost at that city.
        capacity (int): The maximum fuel tank capacity.
        start (int): The starting city.
        target (int): The target city.

    Returns:
        int: The minimum total refueling cost required to get from the start city to the target city. Returns -1 if the target is unreachable.
    """

    # Create a graph from the roads
    graph = {i: [] for i in range(n)}
    for u, v, dist in roads:
        graph[u].append((v, dist))

    # Initialize the priority queue with the starting state
    pq = [(0, start, capacity)]  # (cost, city, fuel_remaining)
    visited = set()

    while pq:
        # Extract the state with the minimum cost from the priority queue
        cost, city, fuel_remaining = heapq.heappop(pq)

        # If the current city is the target, return the cost
        if city == target:
            return cost

        # If the current state has been visited before, skip it
        if (city, fuel_remaining) in visited:
            continue

        # Mark the current state as visited
        visited.add((city, fuel_remaining))

        # Explore the neighbors of the current city
        for neighbor, dist in graph[city]:
            # If the current fuel is not sufficient to traverse the road, skip this neighbor
            if fuel_remaining < dist:
                continue

            # Calculate the new fuel remaining after traversing the road
            new_fuel_remaining = fuel_remaining - dist

            # Add the neighbor to the priority queue
            heapq.heappush(pq, (cost, neighbor, new_fuel_remaining))

            # If the current city is a fuel station, consider the refueling option
            if city in fuel_stations:
                # Calculate the new cost and fuel remaining after refueling
                new_cost = cost + fuel_stations[city]
                new_fuel_remaining = capacity

                # Add the refueled state to the priority queue
                heapq.heappush(pq, (new_cost, city, new_fuel_remaining))

    # If the target is unreachable, return -1
    return -1

