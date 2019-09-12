"""
Created on Sun Sep  12 04:36:40 2019
@author: Neeraj
Description: Computes model assessment metrics like accuracy, precision, recall and F1-score.
Reference: Chapter 11 : Machine Learning
"""

import random
from typing import List, TypeVar, Tuple
X = TypeVar('X') # generic type to represent a data point

def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1-prob]"""
    data = data[:] # make a shallow copy of the data
    random.shuffle(data)
    cut = int(len(data)*prob)
    return data[:cut], data[cut:]

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

print(len(train))
print(len(test))

Y = TypeVar('Y') # generic type to represent output variable

def train_test_split(xs: List[X], ys: List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
        # Generate indices and split them
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1-test_pct)
        
    return ([xs[i] for i in train_idxs], # x_train
                [xs[i] for i in test_idxs], # x_test
                [ys[i] for i in train_idxs], # y_train
                [ys[i] for i in test_idxs])  # y_test
               
xs = [x for x in range(1000)]
ys = [2*x for x in xs]

x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

print("Number of points in training input =",len(x_train))
print("Number of points in training output =", len(y_train))
print("Number of points in testing input =",len(x_test))
print("Number of points in testing output =", len(y_test))

