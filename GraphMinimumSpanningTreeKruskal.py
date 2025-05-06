def kruskal(edges, num_nodes):
    parent = [i for i in range(num_nodes)]
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    
    edges.sort(key=lambda x: x[2])  # Sort by weight
    mst = []
    
    for u, v, weight in edges:
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            mst.append((u, v, weight))
            parent[v_root] = u_root
    return mst

# Example edges: (u, v, weight)
edges = [(0, 1, 2), (0, 2, 3), (1, 2, 1), (1, 3, 4), (2, 3, 5)]
print(kruskal(edges, 4))  # Output: [(1, 2, 1), (0, 1, 2), (1, 3, 4)]