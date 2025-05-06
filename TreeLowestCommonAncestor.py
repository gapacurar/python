class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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