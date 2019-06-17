# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 00:55:07 2019
@author: Neeraj
Description: This code illustrates how to create and manipulate vectors in Python. List is used as a data structure to repesent
vectors in this code.
Reference: Chapter 4 : Linear Algebra 
"""

from typing import List

Vector = List[float]

def add(v: Vector, w: Vector) -> Vector:
    """A function to add two vectors element-wise
    Input: Vectors v and w of the same length
    Output: A vector
    """
    # Check if both vectors are of the same length
    assert len(v) ==  len(w), "The vectors must of the same lenght"

    return [v_i + w_i for v_i,w_i in zip(v,w)]
   
v = [1,2,3]
w = [4,5,6]

print(add(v,w))

def subtract(v: Vector, w: Vector) -> Vector:
    """A function to subtract two vectors element-wise
    Input: Vectors v and w of the same length
    Output: A vector
    """
    # Check if both vectors are of the same length
    assert len(v) ==  len(w), "The vectors must of the same lenght"

    return [v_i - w_i for v_i,w_i in zip(v,w)]
   
v = [5,7,9]
w = [4,5,6]

print(subtract(v,w))

def vector_sum(vectors):
    """Computes the mean vector of a list of vectos.
    Input: A list of vectors
    Output: A vector"""
    
    # check if no vectors are provided
    assert vectors, "no vectors provided!"
    
    # Check if all vectors are of same size
    n = len(vectors[0]) # size of a vector
    
    assert all(len(v) == n for v in vectors),"All vectors should be of the same size"
    
    # i-th element of result is the sume of i-th element of all vectors 
    return [sum(vector[i] for vector in vectors) for i in range(n)]
    
vectors = [[1,2],[3,4],[5,6],[7,8]]

print(vector_sum(vectors))    
    
def scalar_mulitply(c: float, v : Vector) -> Vector:
    """MUltiply a scalar to a vector
    Input: a scalar and a vector
    Output: A vector"""
    
    return [c*v_i for v_i in v]

c = 2
v = [1,2,3]
print(scalar_mulitply(c,v))


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes component wise mean of a list of vectors
    Input: List of vectors
    Output: Mean Vector"""
    #Check if the vector list is empty
    assert vectors, "no vectors provided!"
    
    # Check if the length of all vectors is equal
    n = len(vectors[0]) # length of the first vector
    assert all(len(v) == n for v in vectors), "Length of all vectors should be same"
    
    return scalar_mulitply(1/len(vectors),vector_sum(vectors))

vectors = [[1,2],[3,4],[5,6]]

print(vector_mean(vectors))


def dot(v: Vector, w:Vector) -> float:
    """Computes the dot product between two vectors
    Input: two vectors of equal length
    Output: a scalar"""
    
    # Check if both vectors have equal length
    assert len(v) == len(w), "Vectors should be of the same length"
    return sum([v_i*w_i for v_i, w_i in zip(v,w)])
    
v =[1,2,3]
w = [4,5,6]

print(dot(v,w))

def sum_of_squares(v:Vector) -> float:
    """Computes the sum of squares of a vector's elements"""
    return dot(v,v)

v = [1,2,3]
print(sum_of_squares(v))

import math

def magnitude(v: Vector) -> float:
    """Computes the length of a vector"""
    return math.sqrt(sum_of_squares(v))

v = [3,4]
print(magnitude(v))


def squared_distance(v:Vector, w:Vector) -> Vector:
    """Computes the distance between two vectors"""
    # Check if both vectors have equal dimensions
    assert len(v) == len(w),"Vectors should be of the same length"
    return sum_of_squares(subtract(v,w))


def distance(v:Vector, w:Vector) -> Vector:
    """Computes the distance between two vectors"""
    # Check if both vectors have equal dimensions
    assert len(v) == len(w),"Vectors should be of the same length"
    return math.sqrt(sum_of_squares(subtract(v,w)))














    
    
    
    
