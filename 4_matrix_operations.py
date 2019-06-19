#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:55:07 2019
@author: Neeraj
Description: This code illustrates how to create and manipulate matrices in Python. List of lists is used as a data structure to 
represent matrices in this code.
Reference: Chapter 4 : Linear Algebra 
"""

from typing import List, Tuple, Callable

Matrix = List[List[float]]

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns the size of a matrix"""
    num_rows = len(A) # number of rows in a matrix
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    
    return num_rows, num_cols

def make_matrix(num_rows: int, 
                num_cols: int, 
                entry_fn: Callable[[int,int], float]) -> Matrix:
    """Creates a matrix given the number of rows and columns,
    and a generator function."""
    return [[entry_fn(i,j) for j in range(num_cols)] for i in range(num_rows)]
    
def identity_matrix(n: int) -> Matrix:
    """Creates an nxn identity matrix"""
    return make_matrix(n,n,lambda i,j:1 if i == j else 0)

print(identity_matrix(5))
