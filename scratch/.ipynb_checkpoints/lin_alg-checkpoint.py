#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra: Math field which Vectorial Spaces are calculated.

# ## Vectors: Objects subject to operations in order to form other values

# In[1]:

class LinearAlgebra:
    from typing import List
    #Alias to describe a Vector as simply List[int]
    Vector = List[float]
    height_weight_age = [70, #Inch
                         170,#Pounds
                         40] #Years
    grades = [95,
              80,
              75,
              62]
    
    
    # ### Vector addition:
    
    # In[2]:
    
    
    #Addition is performed component wise
    def add(u: Vector, v: Vector) -> Vector:
        assert len(u) == len(v), "Vectors should be the same size"
        return [xu + xv for xu, xv in zip(u, v)]
        #List comprehension to unzip both vectors components into temporary variables
    
    
    # In[3]:
    
    
    assert add([3,3], [4,4]) == [7,7], "Sum was not as expected"
    
    
    # ### Vector subtraction:
    
    # In[4]:
    
    
    def subtract(u: Vector, v: Vector) -> Vector:
        assert len(u) == len(v), "Vectors should be the same size"
        return [xu - xv for xu, xv in zip(u, v)]
    
    
    # In[5]:
    
    
    assert subtract([3, 5, 7], [3, 5, 7]) == [0, 0, 0], "Sub result was not as expected"
    
    
    # ### Vector summation:
    
    # In[6]:
    
    
    def vector_sum(v: List[Vector]) -> Vector:
        assert v, "Vetores vazios"
        vectors_size = len(v[0])
        assert all(len(vi) == vectors_size for vi in v), "Different sizes"
        return [sum(vector[i] for vector in v) #para todo elemento i de cada vector em v
                for i in range(vectors_size)] #intervalo de i até a quantidade de elementos presente em cada vetor
    
    
    # In[7]:
    
    
    assert vector_sum([[1, 3], [2,4], [5, 6]]) == [8, 13], "sum was not as expected"
    
    
    # ### Scalar multiplication:
    
    # In[8]:
    
    
    #Multiplying every component of a vector by a scalar.
    def scalar_multiplication(a: float, v: Vector) -> Vector:
        return [a * vi for vi in v]
    
    
    # In[9]:
    
    
    assert scalar_multiplication(3.0, [1, 4, 6]) == [3.0, 12.0, 18.0], "multiplication result was not as expected"
    
    
    # ### Vector mean:
    
    # In[10]:
    
    
    def vector_mean(vectors: List[Vector]) -> Vector:
        vectors_size = len(vectors)
        return scalar_multiplication(1/vectors_size, vector_sum(vectors))
    
    
    # In[11]:
    
    
    assert vector_mean([[1,3],[2,4],[3,5]]) == [2.0, 4.0], "mean is not as expected"
    
    
    # ## Matrix mean:
    # ![Matrix Mean Calculation](matrix_mean.jpg "Matrix Mean")
    
    # ### Dot product: Most fundamental vectorial operation!
    
    # In[12]:
    
    
    #Sum of products of vector components
    def dot(u: Vector, v: Vector) -> float:
        # u_1*v_1 + u_2*v_2 ... u_n*v_n
        assert len(u) == len(v)
        return sum([u_i * v_i for u_i, v_i in zip(u, v)])
    
    
    # In[13]:
    
    
    assert dot([1,2,3], [4,5,6]) == 32, "dot product was not as expected"
    
    
    # ## Magnitude of a vector:
    # ![Vector Magnitude](vector_magnitude.jpg "Vector Magnitude")
    
    # ### Sum of Squares:
    
    # In[14]:
    
    
    def sum_of_squares(v: Vector) -> float:
        return dot(v, v)
    
    
    # In[15]:
    
    
    assert sum_of_squares([1,2,3]) == 14, "sum of squares is not as expected"
    
    
    # In[16]:
    
    
    import math
    
    
    # In[17]:
    
    
    def magnitude(v: Vector) ->float: #magnitude is the length of a vector 
        return math.sqrt(sum_of_squares(v))
    
    
    # In[18]:
    
    
    assert magnitude([3, 4]) == 5, "magnitude was not as expected"
    
    
    # ## Distance:
    
    # In[19]:
    
    
    def squared_distance(v: Vector, w: Vector) -> float:
        return sum_of_squares(subtract(v,w))
    
    
    # In[20]:
    
    
    def distance(v: Vector, w: Vector) -> float:
        return math.sqrt(squared_distance(v,w))
    
    
    # In[21]:
    
    
    def distance(v: Vector, w: Vector) -> float:
        return magnitude(subtract(v, w))
    
    
    # ## Matrices: Bidimensional number collection
    
    # In[24]:
    
    
    Matrix = List[List[float]]
    
    A = [[1, 2 ,3], #Matrix A has: 2 rows and 3 columns
         [4, 5, 6]]
    
    B = [[1, 2], #Matrix B has: 3 rows and 2 columns
         [3, 4],
         [5, 6]]
    print(f"Matrix A\nRows: {len(A)} \nColumns: {len(A[0])}")
    print(f"Matrix B\nRows: {len(B)} \nColumns: {len(B[0])}")
    
    
    # ### Shape of a Matrix:
    
    # In[25]:
    
    
    from typing import Tuple
    def shape(A:Matrix) -> Tuple[int,int]:
        num_rows = len(A)
        num_cols = len(A[0]) if A else 0
        return num_rows, num_cols
    
    
    # In[26]:
    
    
    assert shape(A) == (len(A), len(A[0]))
    
    
    # In[27]:
    
    
    def get_row(A:Matrix, i: int) -> Vector:
        return A[i]
    
    
    # In[28]:
    
    
    def get_column(A: Matrix, j: int) -> Vector:
        return[A_i[j] #Selecting only jth column in all rows
               for A_i in A] # Selecting all rows
    
    
    # In[29]:
    
    
    from typing import Callable
    
    def make_matrix(num_rows: int, num_cols: int, entry_fun: Callable[[int,int], float]) -> Matrix:
        return [[entry_fun(i,j)
                 for j in range(num_cols)] # Create a value for each value
                for i in range(num_rows)] # Perform operation for each row (vector)
                
    
    
    # In[30]:
    
    
    def identity_matrix(n:int):
        return make_matrix(n,n, lambda i, j: 1 if i == j else 0)
    
    
    # In[31]:
    
    
    assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                                  [0, 1, 0, 0, 0],
                                  [0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 1]]
    
    
    # In[ ]:
    
    
    
    
