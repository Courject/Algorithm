def max_length(nums):
    n = len(nums)
    if n == 0 or n == 1:
        return n

    opt_n = [1] * n     # opt_n[i]: max length of subset, nums[i] as maximum
    pre = [-1] * n      # record the index of previous number

    for i in range(1, n):
        for j in range(0, i):
            if nums[i] > nums[j]:
                opt_n[i] = max(opt_n[i], opt_n[j] + 1)
                pre[i] = j

    # backtrack to get the max length subset
    subset = []
    max_i = opt_n.index(max(opt_n))
    while max_i != -1:
        subset.append(nums[max_i])
        max_i = pre[max_i]
    subset.reverse()

    return subset

x = max_length([1, 2, 7, 5, 6])
print x
