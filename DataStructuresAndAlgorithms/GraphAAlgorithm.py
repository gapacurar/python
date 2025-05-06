import heapq

def a_star(graph, start, goal, heuristic):
    open_set = [(0 + heuristic(start, goal), 0, start)]
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    
    while open_set:
        _, current_g, node = heapq.heappop(open_set)
        if node == goal:
            path = []
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]
        
        for neighbor, weight in graph[node].items():
            tentative_g = current_g + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = node
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor, goal), tentative_g, neighbor))
    return None

# Example heuristic (Euclidean distance in coordinates)
def heuristic(node, goal):
    coords = {'A': (0,0), 'B': (1,0), 'C': (0,1), 'D': (1,1), 'F': (2,2)}
    x1, y1 = coords[node]
    x2, y2 = coords[goal]
    return ((x2-x1)**2 + (y2-y1)**2)**0.5

# Define the weighted graph
weighted_graph = {
    'A': {'B': 1, 'C': 2},
    'B': {'A': 1, 'D': 2},
    'C': {'A': 2, 'D': 2},
    'D': {'B': 2, 'C': 2, 'F': 3},
    'F': {'D': 3}
}

print(a_star(weighted_graph, 'A', 'F', heuristic))  # Output: ['A', 'B', 'D', 'F']