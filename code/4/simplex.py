#!/usr/bin/python3
# Simplex method
# Author: HongXin
# 2016.11.17

import numpy as np


class Simplex:
    def __init__(self):
        pass

    def xlpsol(self, c, A, b):
        """
        Solve linear programming problem with the follow format:
        min     c^Tx
        s.t.    Ax <= b
                x >= 0
        (c^T means transpose of the vector c)
        :return: x - optimal solution, opt - optimal objective value
        """
        self.__init(c, A, b)
        inf = float('inf')
        while True:
            if all(self.v_c >= 0):  # c >= 0
                # just get optimal solution by manipulating index and value
                x = map(lambda t: self.T[
                    self.B.index(t) + 1, 0] if t in self.B else 0,
                        range(1, self.n))
                return x, -self.T[
                    0, 0]  # -T[0, 0] is exactly the optimal value!
            else:
                # choose fist element of v_c smaller than 0
                e = next(i for i, x in enumerate(self.v_c) if x < 0)
                delta = map(lambda i: self.v_b[i] / self.v_A[i, e]
                if self.v_A[i, e] > 0 else inf,
                            range(0, self.m - 1))
                l = delta.index(min(delta))
                if delta[l] == inf:
                    print("unbounded")
                    return None, None
                else:
                    self.__pivot(e, l)

    def __init(self, c, A, b):
        """
        0   c   0
        b   A   I
        """
        # transfer to vector and matrix
        (c, A, b) = map(lambda t: np.array(t), [c, A, b])
        [m, n] = A.shape
        if m != b.size:
            print('The size of b must equal with the row of A!')
            exit(1)
        if n != c.size:
            print('The size of c must equal with the column of A!')
            exit(1)
        part_1 = np.vstack((0, b.reshape(b.size, 1)))
        part_2 = np.vstack((c, A))
        part_3 = np.vstack((np.zeros(m), np.identity(m)))

        self.B = range(n + 1, n + m + 1)
        self.T = np.hstack((np.hstack((part_1, part_2)), part_3))
        self.v_c = self.T[0, 1:]
        self.v_b = self.T[1:, 0]
        self.v_A = self.T[1:, 1:]
        (self.m, self.n) = self.T.shape

    def __pivot(self, e, l):
        self.T[l + 1, :] = np.divide(self.T[l + 1, :], self.v_A[l, e])
        for i in range(0, self.m):
            if i == l + 1:
                continue
            self.T[i, :] -= self.T[l + 1, :] * self.T[i, e + 1]
        self.B.remove(self.B[l])
        self.B.append(e + 1)
        self.B.sort()


if __name__ == '__main__':
    c = [-1, -14, -6]
    A = [[1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 3, 1]]
    b = [4, 2, 3, 6]
    [x, opt] = Simplex().xlpsol(c, A, b)
    print(x, opt)
