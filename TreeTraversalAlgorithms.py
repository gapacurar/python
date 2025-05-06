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