# This code calculates the diameter of a binary tree.
# The diameter is defined as the length of the longest path between any two nodes in the tree.
# The path may or may not pass through the root.
# The function `diameter_of_binary_tree` takes the root of the tree as input and returns the diameter.
# The function uses a helper function `depth` to calculate the depth of each node and update the diameter.
# The depth of a node is defined as the number of edges in the longest path from that node to a leaf.
# The function uses a nonlocal variable `diameter` to keep track of the maximum diameter found during the traversal.
# The time complexity of this algorithm is O(n), where n is the number of nodes in the tree.
# The space complexity is O(h), where h is the height of the tree, due to the recursion stack.
# The code is written in Python and uses a simple class definition for the tree nodes.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Function to calculate the diameter of a binary tree
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

def test_diameter_of_binary_tree():
    # Test case 1: Tree with a single node
    root1 = TreeNode(1)
    assert diameter_of_binary_tree(root1) == 0  # Diameter of a single node is 0

    # Test case 2: Tree with multiple levels
    root2 = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
    assert diameter_of_binary_tree(root2) == 3  # Path: 4 → 2 → 5

    # Test case 3: Tree with an unbalanced structure
    root3 = TreeNode(1, TreeNode(2, TreeNode(3, TreeNode(4))))
    assert diameter_of_binary_tree(root3) == 3  # Path: 4 → 3 → 2 → 1

# Main function to run the test cases
def main():
    test_diameter_of_binary_tree()
    print("All test cases passed!")

if __name__ == "__main__":
    main()