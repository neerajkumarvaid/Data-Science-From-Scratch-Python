#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:08:59 2019

@author: Neeraj

Description: This code illustrates how implement gradient descent algorithm in Python.
Minibatch and stochastic versions are also demonstrated.

Reference: Chapter 8 : Gradient Descent
"""

from vector_operations import Vector, dot

def sum_of_squares(v: Vector) -> float:
    """Computes the sum of squared elements in v"""
    return dot(v,v)

from typing import Callable
def difference_quotient(f: Callable[[float],float],
                       x: float,
                       h: float) -> float:
    return (f(x+h) - f(x))/h

def square(x: float) -> float:
    return x*x

def derivative(x: float) -> float:
    return 2*x

xs = range(-10,11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square,x, h = 0.001) for x in xs]

import matplotlib.pyplot as plt
plt.title("Actual derivatives vs. estimates")
plt.plot(xs, actuals, 'rx',label = 'Actuals')
plt.plot(xs, estimates, 'b+',  label = 'Estimates')
plt.legend(loc = 9)
plt.show()

def partial_difference_quotient(f: Callable[[float],float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Return the partial difference quotient in i-th direction"""
    # Compute the next point in i-th direction
    w = [v_j + (h if i == j else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v))/h

def estimate_gradient(f: Callable[[float], float],
                      v: Vector,
                      h: float = 0.0001) -> Vector:
    return [partial_difference_quotient(f, v, i, h) 
            for i in range(len(v))]
    
from vector_operations import scalar_mulitply, distance, add
import random
def gradient_step(v: Vector, 
                  gradient: Vector, 
                  step_size: float) -> Vector:
    """Moves 'step size' in the 'gradient' direction of v"""
    assert len(v) == len(gradient)
    return add(v, scalar_mulitply(step_size, gradient))
# create a gradient function (d(x**2)/dx = 2*x )
def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2*v_i for v_i in v]
    
# pick a random starting point
v = [random.uniform(-10,10) for i in range(3)]

for epoch in range(1000):
    # compute gradient
    grad = sum_of_squares_gradient(v)
    # update vector in the gradient direction
    v = gradient_step(v, grad, step_size = -0.01) # take a step in the negative gradient direction
    print(epoch, v)
 
print("Distance = ",distance(v,[0,0,0]))

assert distance(v,[0,0,0]) < 0.001 # v should be close to zero



"""Using the gradient descent algorith to find the slope and intercept of a linear equation"""
# Create a linear equation with know parameters (slope = 20, intercept = 5)
input = [(x, 20*x + 5) for x in range(-50,50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope*x + intercept # prediction of a linear model
    error = (predicted - y)
    squared_error = error**2 # minimize squared error
    grad = [2*error*x, 2*error] # using its gradient
    return grad

from vector_operations import vector_mean
# start with a random slope and intercept
theta = [random.uniform(-1,1), random.uniform(-1,1)]

learning_rate = 0.001

for epoch in range(5000):
    # compute mean of the gradients
    grad = vector_mean([linear_gradient(x,y, theta) for x,y in input])
    # Take a step in that direction
    theta = gradient_step(theta, grad,-learning_rate)
    print(epoch, theta)
    
slope, intercept = theta
assert 19.9 < slope < 20.1 # slope should be close to 20
assert 4.9 < intercept  < 5.1 # intercept should be close to 5

"""Let's solve the above problem in minibatches"""
from typing import TypeVar, List, Iterator

T = TypeVar('T') # this allows us to type generic functions
def minibatches(dataset: List[T],
               batch_size = int,
               shuffle: bool = True) -> Iterator[List]:
    """Generate 'batch_size'-sized batches from the data"""
    # create starting indices of the batches
    batch_starts = [start for start in range(0, len(dataset), batch_size)]
    # shuffle the batches
    if shuffle: random.shuffle(batch_starts)
        
    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]

theta = [random.uniform(-1,1), random.uniform(-1,1)]

for epoch in range(1000):
    for batch in minibatches(input, batch_size = 20):
        grad = vector_mean([linear_gradient(x,y, theta) for x,y in batch])
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

"""Let's solve the above problem using stochastic gradient descent"""
theta = [random.uniform(-1,1), random.uniform(-1,1)]

for epoch in range(1000):
    for x,y in input:
        grad = linear_gradient(x,y, theta)
        theta = gradient_step(theta, grad, -learning_rate)
    print(epoch, theta)

    
    
    
    