def encode_way(s):
    n = len(s)
    if n == 0 or s[0] == '0':
        return 0

    pos = [1, 1]
    for i in range(1, n):
        tmp = pos[1]
        if s[i] == '0':
            if s[i-1] == '1' or s[i-1] == '2':
                pos[1] = pos[0]
            else:
                return 0
        elif s[i-1] == '1' or (s[i-1] == '2' and s[i] <= '6'):
            pos[1] += pos[0]
        pos[0] = tmp
    return pos[1]


ways = encode_way('100')
print ways

# n = len(s)
# if n == 0:
#     return 0
#
# if s[0] == '0':
#     return 0
#
# pos = [1, 1]
# for i in range(1, n):
#     tmp = pos[1]
#     num = int(s[i - 1:i + 1])
#     if num % 10 == 0:
#         if num / 10 in [1, 2]:
#             pos[1] = pos[0]
#         else:
#             return 0
#     elif 11 <= num <= 26:
#         pos[1] += pos[0]
#
#     pos[0] = tmp
#
# return pos[1]
