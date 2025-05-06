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