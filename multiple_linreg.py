from math import inf


class LinearModel:

    @staticmethod
    def transpose_matrix(m: list[list]):
        return list(map(list, zip(*m)))

    @staticmethod
    def get_matrix_minor(m, i, j):
        return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]

    @staticmethod
    def get_matrix_determinant(m):
        # base case for 2x2 matrix
        if len(m) == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]

        determinant = 0
        for c in range(len(m)):
            determinant += ((-1) ** c) * m[0][c] * LinearModel.get_matrix_determinant(
                LinearModel.get_matrix_minor(m, 0, c))
        return determinant

    @staticmethod
    def get_matrix_inverse(m):
        determinant = LinearModel.get_matrix_determinant(m)
        # special case for 2x2 matrix:
        if len(m) == 2:
            return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                    [-1 * m[1][0] / determinant, m[0][0] / determinant]]

        # find matrix of cofactors
        cofactors = []
        for r in range(len(m)):
            cofactorRow = []
            for c in range(len(m)):
                minor = LinearModel.get_matrix_minor(m, r, c)
                cofactorRow.append(((-1) ** (r + c)) * LinearModel.get_matrix_determinant(minor))
            cofactors.append(cofactorRow)
        cofactors = LinearModel.transpose_matrix(cofactors)
        for r in range(len(cofactors)):
            for c in range(len(cofactors)):
                cofactors[r][c] = cofactors[r][c] / determinant
        return cofactors

    @staticmethod
    def multiply_matrices(m1, m2):
        m3 = [[0] * len(m2[0]) for _ in range(len(m1))]

        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for h in range(len(m1[0])):
                    m3[i][j] += m1[i][h] * m2[h][j]

        return m3

    @staticmethod
    def mean(iterable):
        return sum(_[0] for _ in iterable) / sum(1 for _ in iterable)

    @staticmethod
    def add_constant(matrix):
        return [[1] + _ for _ in matrix]

    @staticmethod
    def vector_to_matrix(v):
        if isinstance(v[0], list):
            return v
        return [[_] for _ in v]

    def fit(self, x, y, with_constant=True):
        self.__constant = with_constant
        self.__x = self.vector_to_matrix(x)
        self.__y = self.vector_to_matrix(y)
        self.__n = sum(1 for _ in self.__x)
        self.__k = sum(1 for _ in self.__x[0])
        if with_constant:
            self.__x = self.add_constant(self.__x)

        Xt = self.transpose_matrix(self.__x)
        XtXi = self.get_matrix_inverse(self.multiply_matrices(self.transpose_matrix(self.__x), self.__x))
        self.__B = [_ for _ in self.multiply_matrices(self.multiply_matrices(XtXi, Xt), self.__y)]

        y_pred = self.multiply_matrices(self.__x, self.__B)
        y_pred_m = self.mean(y_pred)
        y_m = self.mean(self.__y)
        ESS = sum((_[0] - y_pred_m) ** 2 for _ in y_pred)
        TSS = sum((_[0] - y_m) ** 2 for _ in self.__y)
        self.__R_squared = ESS / TSS
        self.__adj_R_squared = 1 - (1 - self.__R_squared) * ((self.__n - 1) / (self.__n - self.__k - 1))
        try:
            self.__f_statistic = (self.__R_squared / self.__k) / ((1 - self.__R_squared) / (self.__n - self.__k - 1))
        except ZeroDivisionError:
            self.__f_statistic = inf

    def describe(self):
        y = bytes.fromhex('C5 B7').decode('UTF-8')
        B = map(lambda x: round(x[0], 2), self.__B)
        if self.__constant:
            X = f'{next(B)} + '
            X = X + ' + '.join(f'{_}x{i}' for i, _ in enumerate((_ for _ in B), start=1))
        else:
            X = ' + '.join(f'{_}x{i}' for i, _ in enumerate((_ for _ in B), start=1))
        print(f'The empirical regression equation is {y} = {X} \n'
              f'Determination coefficient of the model is {round(self.__R_squared, 2)}\n'
              f'Adjusted determination coefficient of the model is {round(self.__adj_R_squared, 2)}\n'
              f'F-statistic for the model is {round(self.__f_statistic, 2)}'
              )


x = [
    [1, 2, 3, 4, 5]
    , [6, 4, 8, 9, 1]
    , [8, 4, 1, 8, 3]
    , [12, 7, 3, 2, 9]
    , [8, 2, 8, 4, 5]
    , [11, 6, 2, 17, 4]
    , [10, 2, 6, 11, 0]
]
y = [6, 7, 8, 9, 15, 11, 3]

model = LinearModel()
model.fit(x, y)

model.describe()
