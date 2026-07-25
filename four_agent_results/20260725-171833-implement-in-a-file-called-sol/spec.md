Function: def min_cost_drive(
    n: int,
    roads: list[list[int]],
    fuel_stations: dict,
    capacity: int,
    start: int,
    target: int,
) -> int

Summary: The MINIMUM TOTAL REFUELING COST to get from start to target in a
    directed graph with one-way roads and refuelable fuel stations.

Invariants:
- The cost returned is -1 if and only if the target cannot be reached.
- The cost returned is non-negative if the target can be reached.
- If start == target, the cost is 0.
- If a road exists but the fuel capacity is less than that road's dist, it is not traversed.

Edge cases:
- If an empty input is passed (n == 0), the graph is considered trivial and the target is unreachable.
- If a single road with negative distance is passed (i.e., a road with neg. dist), the function should raise an error.
- If the fuel capacity is less than the smallest road's distance, the target is unreachable.
- If roads are disconnected, the target could be unreachable.
- Refueling at a station of zero cost has no effect.
- Refueling at the same station multiple times is allowed if it's a cheaper alternative.

Brute-force verifier: A brute-force solution can try every possible sequence of fuel refuelings
    and edge traversals while checking if reaching the target is possible. It can be
    implemented by considering all possible combinations of refuelings and traversals,
    and recursively trying out each of them until all are exhausted. If a target is
    reached in any sequence, the total refueling cost for that sequence is calculated
    and checked for optimality with the previous known optimal values.
