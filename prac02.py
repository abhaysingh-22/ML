import time
from collections import deque

# 0 = free, 1 = wall
MAZE = [
    [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0],
]
START, GOAL = (0, 0), (5, 5)


def neighbors(r, c):
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(MAZE) and 0 <= nc < len(MAZE[0]) and MAZE[nr][nc] == 0:
            yield nr, nc


def search(use_queue):
    """BFS if use_queue=True (FIFO), else DFS (LIFO)."""
    frontier = deque([START])
    parent = {START: None}
    explored = 0
    while frontier:
        node = frontier.popleft() if use_queue else frontier.pop()
        explored += 1
        if node == GOAL:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            return path[::-1], explored
        for n in neighbors(*node):
            if n not in parent:
                parent[n] = node
                frontier.append(n)
    return None, explored

for name, q in [("bfs", True), ("dfs", False)]:
    t = time.perf_counter()
    path, e = search(q)
    ela = (time.perf_counter() - t) * 1000

    print(name)
    print("path:", path)
    print("path lenth:", len(path) if path else "not found")
    print("node exp: ", e)
    print("time:", round(ela, 4))
    print()