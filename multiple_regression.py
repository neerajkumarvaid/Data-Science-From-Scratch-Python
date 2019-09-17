from vector_operations import dot, Vector;

def predict(x: Vector, beta: Vector) -> float:
    """Assumes that the first element of x is 1"""
    return dot(x, beta)

predict([1,49,4,0],[3,5,2,4])

def error(x: Vector, y: float, beta: Vector) -> float:
    return predict(x, beta) -y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
    return error(x, y, beta) ** 2

x = [1,2,3]
y = 30
beta = [4,4,4]
print(error(x,y,beta))
print(squared_error(x,y,beta))

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
    return [2 * error(x,y,beta) * x_i for x_i in x]

sqerror_gradient(x,y,beta)

from typing import List
import random
from vector_operations import vector_mean
from gradient_descent import gradient_step;
import tqdm 
