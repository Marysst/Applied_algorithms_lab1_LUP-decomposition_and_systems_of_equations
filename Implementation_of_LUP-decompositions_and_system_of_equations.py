import numpy as np

class Vector:
    def __init__(self, data):
        # Ініціалізуємо вектор з даних, які подаються як вхідний параметр
        self.data = np.array(data)
        self.size = self.data.shape[0]

    def add(self, other):
        # Додаємо інший вектор
        return Vector(self.data + other.data)

    def subtract(self, other):
        # Віднімаємо інший вектор
        return Vector(self.data - other.data)

    def dot(self, other):
        # Скалярний добуток з іншим вектором
        return np.dot(self.data, other.data)

    def __str__(self):
        # Повертає рядкове представлення вектора
        return str(self.data)


class Matrix:
    def __init__(self, data):
        # Ініціалізуємо матрицю з даних, що подаються як вхідний параметр
        self.data = np.array(data)
        self.n = self.data.shape[0]

    def add(self, other):
        # Додаємо іншу матрицю
        return Matrix(self.data + other.data)

    def subtract(self, other):
        # Віднімаємо іншу матрицю
        return Matrix(self.data - other.data)

    def multiply(self, other):
        if isinstance(other, Matrix):
            # Множимо на іншу матрицю
            return Matrix(self.data @ other.data)
        elif isinstance(other, Vector):
            # Множимо на вектор
            return Vector(self.data @ other.data)
        else:
            raise TypeError("Unsupported multiplication type. Must be Matrix or Vector.")

    def __str__(self):
        # Повертає рядкове представлення матриці
        return str(self.data)


class LUPDecomposition:
    def __init__(self, matrix):
        # Ініціалізуємо об'єкт матриці для LUP-розкладу
        self.A = np.array(matrix, dtype=float)
        self.n = self.A.shape[0]
        self.P = np.arange(self.n)  # Ініціалізація вектора перестановок

    def decompose(self):
        for k in range(self.n):
            # Знаходження індексу з максимальним значенням у стовпці
            k_prime = np.argmax(abs(self.A[k:self.n, k])) + k
            if self.A[k_prime, k] == 0:
                raise ValueError("Матриця вироджена!")

            # Поміняти місцями рядки k та k'
            if k != k_prime:
                self.A[[k, k_prime]] = self.A[[k_prime, k]]
                self.P[[k, k_prime]] = self.P[[k_prime, k]]

            # Оновлення значень в матриці
            for i in range(k + 1, self.n):
                self.A[i, k] /= self.A[k, k]
                for j in range(k + 1, self.n):
                    self.A[i, j] -= self.A[i, k] * self.A[k, j]

        return self.A, self.get_permutation_matrix()

    def get_L(self):
        # Повертає нижню трикутну матрицю L
        L = np.tril(self.A, -1) + np.eye(self.n)
        return Matrix(L)

    def get_U(self):
        # Повертає верхню трикутну матрицю U
        U = np.triu(self.A)
        return Matrix(U)

    def get_permutation_matrix(self):
        # Побудова матриці перестановок з вектора P
        P_matrix = np.eye(self.n)[self.P]
        return Matrix(P_matrix)

    def solve(self, b):
        """Розв'язує систему A * x = b за допомогою LUP-розкладу"""
        # Перетворення b відповідно до матриці перестановок P
        Pb = self.get_permutation_matrix().multiply(Vector(b))

        # Прямий хід: розв'язуємо L * y = Pb
        y = np.zeros(self.n)
        L = self.get_L().data
        for i in range(self.n):
            y[i] = Pb.data[i] - np.dot(L[i, :i], y[:i])

        # Зворотний хід: розв'язуємо U * x = y
        x = np.zeros(self.n)
        U = self.get_U().data
        for i in range(self.n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

        return Vector(x)
