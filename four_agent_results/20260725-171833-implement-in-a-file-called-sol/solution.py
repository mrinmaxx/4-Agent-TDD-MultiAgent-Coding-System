from typing import Dict, List
import itertools

def min_cost_drive(
    n: int,
    roads: List[List[int]],
    fuel_stations: Dict[int, int],
    capacity: int,
    start: int,
    target: int,
) -> int:
    if n == 0:
        return -1
    for road in roads:
        if road[2] < 0:
            raise ValueError("Negative dist")
    if capacity < min((min(road[1] for road in roads), min(road[2] for road in roads))[1] for road in roads):
        return -1

    stations = list(fuel_stations.keys())
    dist = {i: float('inf') for i in range(n)}
    dist[start] = 0

    def dijkstra(from_node):
        queue = [(dist[from_node], from_node)]
        refueling_cost = 0
        while queue:
            cost, node = min(queue)
            queue.remove((cost, node))
            if node == target:
                return cost
            if node in stations:
                refueling_cost = fuel_stations[node]
            dist[target] = float('inf')  # Add this line
            for road in roads:
                u, v, dist_u_v = road[0], road[1], road[2]
                if (u, v) == (node, from_node) and dist_u_v <= capacity - (cost - dist[u]):
                    if dist[v] > cost + dist_u_v + refueling_cost:
                        dist[v] = cost + dist_u_v + refueling_cost
                        queue.append((dist[v], v))
            for other_node in queue:
                queue.remove(other_node)
                new_cost = other_node[0] + refueling_cost
                if new_cost < dist[other_node[1]]:
                    dist[other_node[1]] = new_cost
                    queue.append((new_cost, other_node[1]))
        return -1

    for station in sorted(stations, key=fuel_stations.get):
        cost = dijkstra(station)
        if cost != -1:
            return cost

    return -1
