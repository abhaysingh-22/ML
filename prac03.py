import heapq
import time

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


def h(n):  # Manhattan distance (admissible for 4-direction grid)
    return abs(n[0] - GOAL[0]) + abs(n[1] - GOAL[1])


def neighbors(r, c):
    for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(MAZE) and 0 <= nc < len(MAZE[0]) and MAZE[nr][nc] == 0:
            yield nr, nc


def search(astar):
    """A*: f = g + h.  Greedy: f = h."""
    pq = [(h(START), 0, START)]
    parent, cost = {START: None}, {START: 0}
    explored = 0
    while pq:
        _, g, node = heapq.heappop(pq)
        if g > cost[node]:  # stale entry
            continue
        explored += 1
        if node == GOAL:
            path = []
            while node:
                path.append(node)
                node = parent[node]
            return path[::-1], explored
        for n in neighbors(*node):
            ng = g + 1
            if n not in cost or (astar and ng < cost[n]):
                cost[n], parent[n] = ng, node
                heapq.heappush(pq, ((ng if astar else 0) + h(n), ng, n))
    return None, explored


print(f"{'Algo':<8}{'Path Length':<13}{'Nodes Explored':<16}{'Time (ms)':<10}")
for name, a in [("Greedy", False), ("A*", True)]:
    t = time.perf_counter()
    path, explored = search(a)
    ms = (time.perf_counter() - t) * 1000
    print(f"{name:<8}{len(path) if path else 'N/A':<13}{explored:<16}{ms:<10.4f}")
    print("Path:", path)

print("""
Analysis:
             Optimal   Complete (finite graph)   Efficiency
Greedy       No        Yes (with visited set)    Few nodes, fast, but path may be longer
A*           Yes       Yes                       Explores more than Greedy, guarantees shortest
             (if h is admissible)
""")