from typing import List, Tuple, Callable
import math


Vector = List[float]
Matrix = List[List[float]]

class LinearAlgebra:
    
    @staticmethod
    def add(u: Vector, v: Vector) -> Vector:
        assert len(u) == len(v), "Vectors should be the same size"
        return [xu + xv for xu, xv in zip(u, v)]

    @staticmethod
    def subtract(u: Vector, v: Vector) -> Vector:
        assert len(u) == len(v), "Vectors should be the same size"
        return [xu - xv for xu, xv in zip(u, v)]

    @staticmethod
    def vector_sum(v: List[Vector]) -> Vector:
        assert v, "Vectors list is empty"
        vectors_size = len(v[0])
        assert all(len(vi) == vectors_size for vi in v), "Different sizes"
        return [sum(vector[i] for vector in v) for i in range(vectors_size)]

    @staticmethod
    def scalar_multiplication(a: float, v: Vector) -> Vector:
        return [a * vi for vi in v]

    @staticmethod
    def vector_mean(vectors: List[Vector]) -> Vector:
        vectors_size = len(vectors)
        return LinearAlgebra.scalar_multiplication(1/vectors_size, LinearAlgebra.vector_sum(vectors))

    @staticmethod
    def dot(u: Vector, v: Vector) -> float:
        assert len(u) == len(v)
        return sum([u_i * v_i for u_i, v_i in zip(u, v)])

    @staticmethod
    def sum_of_squares(v: Vector) -> float:
        return LinearAlgebra.dot(v, v)

    @staticmethod
    def magnitude(v: Vector) -> float:
        return math.sqrt(LinearAlgebra.sum_of_squares(v))

    @staticmethod
    def squared_distance(v: Vector, w: Vector) -> float:
        return LinearAlgebra.sum_of_squares(LinearAlgebra.subtract(v, w))

    @staticmethod
    def distance(v: Vector, w: Vector) -> float:
        return LinearAlgebra.magnitude(LinearAlgebra.subtract(v, w))

    @staticmethod
    def shape(A: Matrix) -> Tuple[int, int]:
        num_rows = len(A)
        num_cols = len(A[0]) if A else 0
        return num_rows, num_cols

    @staticmethod
    def get_row(A: Matrix, i: int) -> Vector:
        return A[i]

    @staticmethod
    def get_column(A: Matrix, j: int) -> Vector:
        return [A_i[j] for A_i in A]

    @staticmethod
    def make_matrix(num_rows: int, num_cols: int, entry_fun: Callable[[int, int], float]) -> Matrix:
        return [[entry_fun(i, j) for j in range(num_cols)] for i in range(num_rows)]

    @staticmethod
    def identity_matrix(n: int) -> Matrix:
        return LinearAlgebra.make_matrix(n, n, lambda i, j: 1 if i == j else 0)
