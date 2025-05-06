# Use Case: Find the maximum sum of a subarray of size k.
def max_subarray(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

nums = [1, 3, -1, -3, 5, 3]
print(max_subarray(nums, 3))  # Output: 8 (5 + 3 = 8)