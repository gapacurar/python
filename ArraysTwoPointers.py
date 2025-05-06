# Use Case: Find pairs in a sorted array that sum to a target.
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