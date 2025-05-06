import heapq

def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    heap = [(0, start)]
    
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > distances[node]:
            continue
        for neighbor, weight in graph[node].items():
            distance = current_dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(heap, (distance, neighbor))
    return distances

# Weighted graph example
weighted_graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 2},
    'C': {'A': 4, 'D': 3},
    'D': {'B': 2, 'C': 3, 'F': 5},
    'F': {'D': 5}
}

print(dijkstra(weighted_graph, 'A'))  # Output: {'A': 0, 'B': 1, 'C': 4, 'D': 3, 'F': 8}