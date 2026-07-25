def rotate_rings(grid: list[list[int]], k: int) -> list[list[int]]:
    """
    Rotate each ring in the given grid independently by k positions in the CLOCKWISE direction.

    Args:
    grid (list[list[int]]): A 2D list representing the grid.
    k (int): The number of positions to rotate each ring.

    Returns:
    list[list[int]]: The resulting grid after rotating each ring.
    """
    m, n = len(grid), len(grid[0])
    result = [[grid[i][j] for j in range(n)] for i in range(m)]

    for layer in range(min(m, n) // 2):
        ring = []
        for i in range(layer, n - layer):
            ring.append(grid[layer][i])
        for i in range(layer + 1, m - layer):
            ring.append(grid[i][n - layer - 1])
        for i in range(n - layer - 2, layer - 1, -1):
            ring.append(grid[m - layer - 1][i])
        for i in range(m - layer - 2, layer, -1):
            ring.append(grid[i][layer])

        ring_length = len(ring)
        rotated_ring = ring[-k % ring_length:] + ring[:-k % ring_length]

        index = 0
        for i in range(layer, n - layer):
            result[layer][i] = rotated_ring[index]
            index += 1
        for i in range(layer + 1, m - layer):
            result[i][n - layer - 1] = rotated_ring[index]
            index += 1
        for i in range(n - layer - 2, layer - 1, -1):
            result[m - layer - 1][i] = rotated_ring[index]
            index += 1
        for i in range(m - layer - 2, layer, -1):
            result[i][layer] = rotated_ring[index]
            index += 1

    return result
