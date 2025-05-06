# **Data Structures & Algorithms Tutorial with Python Examples**  

This tutorial covers key data structures and algorithms from the cheat sheet, with Python code examples for each concept.  

---

## **1. Arrays**  
**Definition:** Contiguous memory storage of elements.  

### **Key Techniques & Examples**  

#### **1.1 Two Pointers**  
**Use Case:** Find pairs in a sorted array that sum to a target.  
```python
def two_sum(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current_sum = nums[left] + nums[right]
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return []

nums = [1, 2, 3, 4]
print(two_sum(nums, 5))  # Output: [0, 3] (1 + 4 = 5)
```

#### **1.2 Sliding Window**  
**Use Case:** Find the maximum sum of a subarray of size `k`.  
```python
def max_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

nums = [1, 3, -1, -3, 5, 3]
print(max_subarray(nums, 3))  # Output: 8 (5 + 3 = 8)
```

#### **1.3 Sorting**  
**Use Case:** Sort an array for efficient searching.  
```python
nums = [4, 2, 1, 3]
nums.sort()
print(nums)  # Output: [1, 2, 3, 4]
```

---

## **2. Linked Lists**  
**Definition:** A sequence of nodes where each node points to the next.  

### **Key Problems & Examples**  

#### **2.1 Reverse a Linked List**  
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def reverse_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev

# Example: 1 → 2 → 3 → 4 → None
head = ListNode(1, ListNode(2, ListNode(3, ListNode(4))))
reversed_head = reverse_list(head)
while reversed_head:
    print(reversed_head.val, end=" → ")  # Output: 4 → 3 → 2 → 1 → 
    reversed_head = reversed_head.next
```

#### **2.2 Detect a Cycle (Floyd’s Algorithm)**  
```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Example with a cycle (3 → 4 → 2 → 3 → ...)
node1 = ListNode(3)
node2 = ListNode(4)
node3 = ListNode(2)
node1.next = node2
node2.next = node3
node3.next = node1  # Creates a cycle
print(has_cycle(node1))  # Output: True
```

#### **2.3 Merge Two Sorted Lists**  
```python
def merge_lists(l1, l2):
    dummy = ListNode()
    current = dummy
    while l1 and l2:
        if l1.val < l2.val:
            current.next = l1
            l1 = l1.next
        else:
            current.next = l2
            l2 = l2.next
        current = current.next
    current.next = l1 or l2
    return dummy.next

# Example: 1 → 3 → 5 & 2 → 4 → 6 → None
l1 = ListNode(1, ListNode(3, ListNode(5)))
l2 = ListNode(2, ListNode(4, ListNode(6)))
merged = merge_lists(l1, l2)
while merged:
    print(merged.val, end=" → ")  # Output: 1 → 2 → 3 → 4 → 5 → 6 → 
    merged = merged.next
```

---

## **3. Stacks & Queues**  
**Definitions:**  
- **Stack (LIFO):** Last In, First Out.  
- **Queue (FIFO):** First In, First Out.  

### **Key Problems & Examples**  

#### **3.1 Evaluate Reverse Polish Notation (RPN)**  
```python
def eval_rpn(tokens):
    stack = []
    for token in tokens:
        if token in "+-*/":
            b = stack.pop()
            a = stack.pop()
            if token == "+": stack.append(a + b)
            elif token == "-": stack.append(a - b)
            elif token == "*": stack.append(a * b)
            else: stack.append(int(a / b))
        else:
            stack.append(int(token))
    return stack[0]

tokens = ["2", "1", "+", "3", "*"]
print(eval_rpn(tokens))  # Output: 9 ( (2+1) * 3 )
```

#### **3.2 Parentheses Validation**  
```python
def is_valid(s):
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack

print(is_valid("()[]{}"))  # Output: True
print(is_valid("(]"))      # Output: False
```

#### **3.3 Implement a Cache (LRU Cache)**  
```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))  # Output: 1
cache.put(3, 3)      # Evicts key 2
print(cache.get(2))  # Output: -1 (not found)
```

---

## **4. Searching**  

### **Key Algorithms & Examples**  

#### **4.1 Binary Search**  
```python
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

nums = [1, 2, 3, 4, 5]
print(binary_search(nums, 3))  # Output: 2
```

#### **4.2 Breadth-First Search (BFS)**  
```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        node = queue.popleft()
        print(node, end=" ")
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
bfs(graph, 2)  # Output: 2 0 3 1
```

#### **4.3 A* Search (With Heuristics)**  
```python
import heapq

def a_star(start, goal, neighbors_fn, heuristic_fn):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_fn(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in neighbors_fn(current):
            tentative_g = g_score[current] + 1  # Assuming uniform cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic_fn(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found

# Example usage requires defining `neighbors_fn` and `heuristic_fn`.
```

---

## **5. Trees**  

### **Key Algorithms & Examples**  

#### **5.1 Tree Traversal (In-Order, Pre-Order, Post-Order)**  
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []

def postorder(root):
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []

root = TreeNode(1, TreeNode(2), TreeNode(3))
print("In-Order:", inorder(root))    # Output: [2, 1, 3]
print("Pre-Order:", preorder(root))  # Output: [1, 2, 3]
print("Post-Order:", postorder(root)) # Output: [2, 3, 1]
```

#### **5.2 Lowest Common Ancestor (LCA)**  
```python
def lca(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root
    return left or right

root = TreeNode(3, TreeNode(5, TreeNode(6), TreeNode(2)), TreeNode(1))
p = root.left.left  # Node 6
q = root.left.right # Node 2
print(lca(root, p, q).val)  # Output: 5 (LCA of 6 and 2)
```

#### **5.3 Diameter of a Tree**  
```python
def diameter_of_binary_tree(root):
    diameter = 0
    def depth(node):
        nonlocal diameter
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        diameter = max(diameter, left + right)
        return max(left, right) + 1
    depth(root)
    return diameter

root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
print(diameter_of_binary_tree(root))  # Output: 3 (Path: 4 → 2 → 5)
```

---

## **6. Sorting**  

### **Key Algorithms & Examples**  

#### **6.1 Quick Sort**  
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

arr = [3, 6, 8, 10, 1, 2, 1]
print(quicksort(arr))  # Output: [1, 1, 2, 3, 6, 8, 10]
```

#### **6.2 Merge Sort**  
```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

arr = [12, 11, 13, 5, 6, 7]
print(merge_sort(arr))  # Output: [5, 6, 7, 11, 12, 13]
```

#### **6.3 Heap Sort**  
```python
import heapq

def heap_sort(arr):
    heapq.heapify(arr)
    return [heapq.heappop(arr) for _ in range(len(arr))]

arr = [12, 11, 13, 5, 6, 7]
print(heap_sort(arr))  # Output: [5, 6, 7, 11, 12, 13]
```

#### **6.4 Insertion Sort**  
**Definition:** Builds the final sorted array one element at a time by repeatedly inserting the next element into its correct position.  

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

arr = [12, 11, 13, 5, 6]
print(insertion_sort(arr))  # Output: [5, 6, 11, 12, 13]
```

**Key Characteristics:**  
- **Time Complexity:**  
  - Best Case: \(O(n)\) (already sorted)  
  - Worst Case: \(O(n^2)\) (reverse sorted)  
- **Space Complexity:** \(O(1)\) (in-place)  
- **Use Case:** Small datasets or nearly sorted arrays.  

---

#### **6.5 Topological Sort**  
**Definition:** A linear ordering of vertices in a Directed Acyclic Graph (DAG) where for every directed edge \(u \rightarrow v\), vertex \(u\) comes before \(v\).  

##### **Python Implementation (Using Kahn’s Algorithm)**  
```python
from collections import deque

def topological_sort(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in in_degree if in_degree[u] == 0])
    top_order = []

    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(top_order) == len(graph):
        return top_order
    else:
        return []  # Graph has a cycle

# Example DAG (Directed Acyclic Graph)
graph = {
    'A': ['C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['F'],
    'F': []
}

print(topological_sort(graph))  
# Output: ['A', 'B', 'C', 'D', 'E', 'F'] (One possible order)
```

**Key Characteristics:**  
- **Time Complexity:** \(O(V + E)\) (Vertices + Edges)  
- **Space Complexity:** \(O(V)\)  
- **Use Cases:**  
  - Task scheduling (e.g., build systems like `make`).  
  - Dependency resolution (e.g., course prerequisites).  

---

## **Updated Sorting Section Summary**  
| Algorithm          | Time Complexity (Worst) | Space Complexity | Use Case                     |  
|--------------------|-------------------------|------------------|------------------------------|  
| **Quick Sort**     | \(O(n^2)\)              | \(O(\log n)\)    | General-purpose, large data  |  
| **Merge Sort**     | \(O(n \log n)\)         | \(O(n)\)         | Stable, external sorting     |  
| **Heap Sort**      | \(O(n \log n)\)         | \(O(1)\)         | In-place, priority queues    |  
| **Insertion Sort** | \(O(n^2)\)              | \(O(1)\)         | Small/nearly-sorted data     |  
| **Topological Sort**| \(O(V + E)\)           | \(O(V)\)         | DAGs, dependency resolution  |  

---

## **Conclusion**  
- **Insertion Sort** is simple but inefficient for large datasets.  
- **Topological Sort** is specialized for dependency-based problems on DAGs.  
- **Practice Tip:** Implement these manually to internalize their mechanics.  

## **Understanding Time Complexity (Worst) and Space Complexity**

## **1. Time Complexity (Worst Case)**
### **Definition:**
- **Time Complexity** measures how the runtime of an algorithm grows as the input size increases.  
- **Worst-case Time Complexity** refers to the maximum time an algorithm could take for any input of size `n`.  

### **Why It Matters:**
- Helps predict how an algorithm will perform with large inputs.  
- Allows comparison between different algorithms (e.g., `O(n log n)` is faster than `O(n²)` for large `n`).  

### **Common Time Complexities (Fastest to Slowest):**
| Notation     | Name               | Example Algorithm          |  
|--------------|--------------------|----------------------------|  
| **O(1)**     | Constant Time      | Accessing an array index.   |  
| **O(log n)** | Logarithmic Time   | Binary search.              |  
| **O(n)**     | Linear Time        | Looping through an array.   |  
| **O(n log n)**| Linearithmic Time | Merge sort, Quick sort.     |  
| **O(n²)**    | Quadratic Time     | Bubble sort, Insertion sort.|  
| **O(2ⁿ)**    | Exponential Time   | Brute-force password cracking.|  

### **Example: Insertion Sort (Worst Case = O(n²))**
- If the input array is **reverse-sorted**, the algorithm must compare and shift every element in each iteration.  
- For `n = 5`, it performs up to `5 + 4 + 3 + 2 + 1 = 15` operations (like a nested loop).  

---

## **2. Space Complexity**
### **Definition:**
- Measures how much **additional memory** an algorithm uses relative to the input size.  
- Includes temporary variables, stacks, and dynamically allocated memory.  

### **Why It Matters:**
- Critical for memory-constrained systems (e.g., embedded devices).  
- Helps optimize algorithms to use less memory.  

### **Common Space Complexities:**
| Notation  | Name               | Example Algorithm          |  
|-----------|--------------------|----------------------------|  
| **O(1)**  | Constant Space     | Iterative algorithms (e.g., Insertion sort). |  
| **O(n)**  | Linear Space       | Storing a copy of the input (e.g., Merge sort). |  
| **O(n²)** | Quadratic Space    | Some DP problems with 2D tables. |  

### **Example: Merge Sort (Space Complexity = O(n))**
- Requires extra space to merge subarrays.  
- For `n = 8`, it needs ~8 additional slots in memory.  

---

## **Key Takeaways**
### 1. **Time Complexity (Worst Case):**  
   - Answers: *"How slow can it get?"*  
   - Example: `O(n²)` means runtime grows quadratically with input size.  

### 2. **Space Complexity:**  
   - Answers: *"How much extra memory does it need?"*  
   - Example: `O(1)` means fixed memory usage (no extra space).  

### **Analogy: Cooking a Meal**
- **Time Complexity** = How long it takes to cook (worst-case: burning food).  
- **Space Complexity** = How many extra bowls/pans you need.  

---

### **How to Use This Knowledge**
#### 1. **Choose algorithms wisely:**  
   - Prefer `O(n log n)` over `O(n²)` for large datasets.  
   - Use `O(1)` space algorithms for low-memory environments.  

#### 2. **Interview Tip:**  
   - If asked *"Can you optimize this?"*, consider both time and space trade-offs.  

#### 3. **Practice:**  
   - Try calculating complexity for simple loops (e.g., `for i in range(n):` → `O(n)`).  

---
## **7. Graph Algorithms Explained for Junior Developers**

Graphs are fundamental data structures used to model relationships between objects. Here's a clear breakdown of essential graph algorithms with Python examples:

### **7.1. Graph Representation**
First, let's understand how to represent graphs in code:

```python
# Adjacency List (Most common)
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['E'],
    'D': ['F'],
    'E': [],
    'F': []
}

# Edge List (Alternative)
edges = [('A','B'), ('A','C'), ('B','D'), ('D','F'), ('C','E')]
```

### **7.2. Traversal Algorithms**

#### **7.2.1 Breadth-First Search (BFS)**
- Explores all neighbors at current depth before moving deeper
- Uses a queue
- **Use cases:** Shortest path in unweighted graphs, social networks

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        print(node, end=' ')
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

bfs(graph, 'A')  # Output: A B C D E F
```

#### **7.2.2 Depth-First Search (DFS)**
- Explores as far as possible along each branch before backtracking
- Uses recursion or a stack
- **Use cases:** Maze solving, cycle detection

```python
def dfs(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    print(node, end=' ')
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

dfs(graph, 'A')  # Output: A B D F C E
```

### **7.3. Path-Finding Algorithms**

#### **7.3.1 Dijkstra's Algorithm**
- Finds shortest paths from a start node to all other nodes
- Works for weighted graphs with non-negative weights
- Uses a priority queue

```python
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
```

#### **7.3.2 A* Algorithm**
- Optimized Dijkstra's with heuristics
- Uses both actual distance and estimated distance to goal
- **Use cases:** Pathfinding in games, GPS navigation

```python
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

print(a_star(weighted_graph, 'A', 'F', heuristic))  # Output: ['A', 'B', 'D', 'F']
```

### **7.4. Special Graph Algorithms**

#### **7.4.1 Topological Sort (Already Covered)**
- For directed acyclic graphs (DAGs)
- Linear ordering where for every edge u → v, u comes before v

#### **7.4.2 Minimum Spanning Tree (Kruskal's Algorithm)**
- Connects all nodes with minimum total edge weight
- Uses Union-Find data structure

```python
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
```

### **7.5. When to Use Which Algorithm**

| Problem Type              | Recommended Algorithm     | Time Complexity  |
|---------------------------|---------------------------|------------------|
| Shortest path (unweighted)| BFS                       | O(V + E)         |
| Shortest path (weighted)  | Dijkstra's                | O((V+E) log V)   |
| Pathfinding with estimate | A*                        | Depends on heuristic |
| Visit all nodes           | DFS/BFS                   | O(V + E)         |
| Minimum spanning tree     | Kruskal's/Prim's          | O(E log V)       |
| Topological sorting       | Kahn's/DFS-based          | O(V + E)         |
| Cycle detection          | DFS                       | O(V + E)         |

**Key Tips:**
1. BFS is better for shortest paths in unweighted graphs
2. Dijkstra's works for weighted graphs with non-negative edges
3. A* is optimized for cases where you have a good heuristic
4. DFS is often simpler to implement recursively

---