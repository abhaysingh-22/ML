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


print(f"{'Algo':<5}{'Path Length':<13}{'Nodes Explored':<16}{'Time (ms)':<10}")
for name, q in [("BFS", True), ("DFS", False)]:
    t = time.perf_counter()
    path, explored = search(q)
    ms = (time.perf_counter() - t) * 1000
    print(f"{name:<5}{len(path) if path else 'N/A':<13}{explored:<16}{ms:<10.4f}")
    print("Path:", path)