Minimize
    max_interval: y
Subject To
    c1: x2 - x1 - y <= 0
    c2: x3 - x2 - y <= 0
    c3: x4 - x3 - y <= 0
Bounds
    2 <= x1 <= 6
    8 <= x2 <= 12
    14 <= x3 <= 18
    21 <= x4 <= 25
End
