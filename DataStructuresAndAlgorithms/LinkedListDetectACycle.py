class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
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