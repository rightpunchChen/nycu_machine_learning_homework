class Mat:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __repr__(self):
        return '\n'.join([' '.join(map(str, row)) for row in self.data])

    def add(self, other):
        result = [[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Mat(result)

    def sub(self, other):
        result = [[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)]
        return Mat(result)

    def mul(self, other):
        result = [[sum(self.data[i][j] * other.data[j][k] for j in range(self.cols)) for k in range(other.cols)] for i in range(self.rows)]
        return Mat(result)
    
    def scalar_mul(self, scalar):
        result = [[self.data[i][j] * scalar for j in range(self.cols)] for i in range(self.rows)]
        return Mat(result)
    
    def T(self):
        result = [[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        return Mat(result)
    
    def sign(self):
        result = [[1 if value > 0 else (-1 if value < 0 else 0) for value in row] for row in self.data]
        return Mat(result)
    

    """
    | 1   0   0 |   | u11 u12 u13 |   | u11    u12             u13                  |
    | l21 1   0 | X | 0   u22 u23 | = | l21u11 l21u12 + u22    l21u13 + u23         | = A
    | l31 l32 1 |   | 0   0   u33 |   | l31u11 l31u12 + l32u22 l31u13 + l32u23 +u33 |

    U(i,j) = A(i,j) - Sum[L(i,k)U(k,j)]k=1~(i-1)
    L(j,i) = 1 if j=i
             else {A(j,i) - Sum[L(j,k)U(k,i)]k=1~(i-1)}/U(i,i)
    """
    def LU(self):
        L = [[0.0] * self.rows for _ in range(self.rows)]
        U = [[0.0] * self.rows for _ in range(self.rows)]
        for i in range(self.rows):
            # compute U
            for j in range(i, self.rows):
                u = sum(L[i][k] * U[k][j] for k in range(i))
                U[i][j] = self.data[i][j] - u
            # compute L
            for j in range(i, self.rows):
                if i == j:
                    L[i][i] = 1.0
                else:
                    l = sum(L[j][k] * U[k][i] for k in range(i))
                    L[j][i] = (self.data[j][i] - l) / U[i][i]
        return Mat(L), Mat(U)
    
    """
    A*A_inv = L*U*A_inv = I
    """
    def inv(self):
        L, U = self.LU()
        I = [[0.0] * self.rows for _ in range(self.rows)]
        for i in range(self.rows): I[i][i] = 1.0

        A_inv = [[0.0] * self.rows for _ in range(self.rows)]
        for i in range(self.rows):
            # solve Ly=I (y=UA^-1)
            I_col = [I[j][i] for j in range(self.rows)]
            Y = self.forward_substitution(L, I_col)
            # solve UA^-1=y
            X = self.backward_substitution(U, Y)
            for j in range(self.rows):
                A_inv[j][i] = X[j]
        return Mat(A_inv)


    """
    | l11 0   0   |   |y1|   |b1|
    | l21 l22 0   | X |y2| = |b2|
    | l31 l32 l33 |   |y3|   |b3|
    y1 = b1/l11
    y2 = (b2 - l21y1)/l22
    yi = {bi - sum(l(i,j)yj)}/l(i,i)
    """
    def forward_substitution(self, L, B):
        L_d = L.data
        n = len(L_d)
        Y = [0.0] * n
        for i in range(n):
            sum_LY = sum(L_d[i][j] * Y[j] for j in range(i))
            Y[i] = (B[i] - sum_LY) / L_d[i][i]
        return Y

    def backward_substitution(self, U, Y):
        U_d = U.data
        n = len(U_d)
        X = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_UX = sum(U_d[i][j] * X[j] for j in range(i + 1, n))
            X[i] = (Y[i] - sum_UX) / U_d[i][i]
        return X

if __name__ == '__main__':
    m1 = Mat([[1.0, 2.0], [3.0, 4.0]])
    # m2 = Mat([[5.0, 6.0], [7.0, 8.0]])
    # m3 = m1.add(m2)
    # m4 = m1.sub(m2)
    # print(m1)
    # print(m1.scalar_mul(2))
    # print(m2)
    # print(m3)
    # print(m4)
    # m5 = m1.T()
    # print(m5)

    # m6 = Mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # m7 = Mat([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # m8 = m6.mul(m7)
    # print(m6)
    # print(m7)
    # print(m8)
    
    m9 = Mat([[1, 2, 3, 4, 5],
              [0, 1, 4, 2, 3],
              [2, 1, 0, 3, 4],
              [4, 2, 1, 0, 3],
              [5, 3, 4, 1, 2]])
    l,u = m9.LU()
    print(l)
    print(u)
    print(l.mul(u))
    # inv = m9.inv()
    # print(inv.mul(m9))
    # m10 = Mat([[-1, 2, 3, 4, 5],
    #           [0, 1, 4, 2, 3],
    #           [2, -1, 0, -3, 4],
    #           [4, -2, -1, 0, 3],
    #           [5, 3, 4, 1, 2]])
    # print(m10.sign())