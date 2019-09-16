def predict(alpha: float, beta: float, x_i: float) -> float:
    return beta * x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
    """
    The error from predicting beta * x_i + alpha
    when the actual value is y_i
    """
    return predict(alpha, beta, x_i) - y_i

from vector_operations import Vector;

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))

from typing import Tuple
from statistics import correlation, standard_deviation, mean;

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float,float]:
    """Given two vectors x and y,
    find the least-squares value of alpha and beta"""
    beta = correlation(x,y)*standard_deviation(y)/standard_deviation(x)
    alpha = mean(y) - beta*mean(x)
    #print(alpha, beta)
    return alpha, beta

x = [i for i in range(-100, 110, 10)]
y = [3 * i - 5 for i in x]

least_squares_fit(x,y)


