import random
import itertools
from solution import min_cost_drive

def _reference(
    n: int,
    roads: list[list[int]],
    fuel_stations: dict,
    capacity: int,
    start: int,
    target: int,
) -> int:
    if n == 0:
        return -1
    if capacity < min(min(road[1] for road in roads) for road in roads):
        return -1

    stations = list(fuel_stations.keys())
    visited = {i for i in range(n)}
    visited_order = []
    dist = {i: float('inf') for i in range(n)}
    dist[start] = 0

    def traverse(from_node, visited=None, cost=0, visited_order=None):
        if from_node == target:
            return cost
        if visited is None:
            visited = set()
        if visited_order is None:
            visited_order = []
        visited_order.append(from_node)
        if from_node in visited:
            return -1
        visited.add(from_node)
        min_dist = float('inf')
        for node in range(n):
            if (from_node, node) not in roads and (node, from_node) not in roads:
                return -1
            if dist[from_node] + roads[(from_node, node)][1] <= capacity:
                min_dist = min(min_dist, dist[from_node] + roads[(from_node, node)][1])
        for station in stations:
            refuel_cost = fuel_stations[station]
            for path in itertools.combinations(visited_order, 2):
                for node1, node2 in itertools.combinations(path, 2):
                    if (node1, node2) not in roads and (node2, node1) not in roads:
                        continue
                    new_cost = traverse(node2, visited, cost + roads[(node1, node2)][1] + refuel_cost, visited_order)
                    if new_cost == -1:
                        continue
                    if new_cost < min_dist:
                        min_dist = new_cost
                        return min_dist
        return -1

    for station in stations:
        refuel_cost = fuel_stations[station]
        min_dur = traverse(start, cost=refuel_cost)
        if min_dur != -1:
            return min_dur

    return -1

def test_fuzz():
    random.seed(0)
    for _ in range(50):
        n = random.randint(1, 10)
        roads = [[]]
        while len(roads[0]) < 10:
            u, v, dist = random.choice(range(n)), random.choice(range(n)), random.randint(1, 50)
            if (u, v) in roads[0]:
                continue
            roads[0].append((u, v, dist))
        fuel_stations = {i: random.randint(0, 10) for i in range(n)}
        capacity = random.randint(10, 50)
        start, target = random.sample(range(n), 2)
        if start == target:
            target = random.choice([i for i in range(n) if i != start])
        for road in roads[0]:
            if road[2] < 0:
                raise ValueError("Negative dist")
        assert min_cost_drive(
            n,
            [(road[0], road[1], road[2]) for road in roads[0]],  # Ensure input is list of [u, v, dist]
            fuel_stations,
            capacity,
            start,
            target,
        ) == _reference(
            n,
            [(road[0], road[1], road[2]) for road in roads[0]],  # Ensure input is list of [u, v, dist]
            fuel_stations,
            capacity,
            start,
            target,
        )

