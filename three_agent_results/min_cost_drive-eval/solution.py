from typing import Dict
import heapq
import random

def min_cost_drive(
    n: int,
    roads: list[list[int]],
    fuel_stations: Dict[int, int],
    capacity: int,
    start: int,
    target: int,
) -> int:
    # Create a distance matrix
    distance = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        distance[i][i] = 0

    refuel_cost = 0
    last_refuel = -1
    for _ in range(2):  # Create a separate loop to initialize the maximum road distance
        for max_distance in [random.randint(1, n * 10), capacity]:  # Check that a road with dist > capacity can never be traversed
            for i in range(n):
                distance[i][i] = 0

            # Update distance matrix based on roads
            distance_from_start = [[float('inf')] * n for _ in range(n)]
            distance_from_start[start][start] = 0

            pq = [(0, start, start)]
            while pq:
                curr_dist, curr_city, prev_city = heapq.heappop(pq)
                for next_city in range(n):
                    if roads[curr_city][next_city] != -1 and roads[curr_city][next_city] <= max_distance and distance_from_start[start][next_city] > curr_dist + roads[curr_city][next_city]:
                        distance_from_start[start][next_city] = curr_dist + roads[curr_city][next_city]
                        heapq.heappush(pq, (distance_from_start[start][next_city], next_city, curr_city))

    for current_city in range(n):
        distance_from_start_start = distance_from_start[start][current_city]
        if current_city == target:
            break
        while last_refuel < current_city and distance_from_start_start <= distance_from_start[start][current_city]:
            refuel_cost += fuel_stations[current_city]
            last_refuel = current_city
            distance_from_start_start += capacity
            distance_from_start[start][current_city] = float('inf')

    # Check if target is reachable with the optimal refueling
    if distance_from_start[start][target] == float('inf'):
        return -1
    return refuel_cost

