# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

num_friends = [100.0,49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

from collections import Counter

friend_counts = Counter(num_friends)

import matplotlib.pyplot as plt

xs = range(101) #x-axis goes from 0 to 100
ys = [friend_counts[x] for x in xs]

plt.bar(xs,ys)
plt.axis([0, 101, 0, 25])
plt.xlabel("# of friends")
plt.ylabel("Friends Count")

# Central tendencies
from typing import List
def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)

mean(num_friends)

def _median_odd(xs: List[float]) -> float:
    return sorted(xs)[len(xs)//2]

def _median_even(xs: List[float]) -> float:
    return (sorted(xs)[len(xs)//2] + sorted(xs)[(len(xs)//2) - 1])/2

def median(xs: List[float]) -> float:
    return _median_odd(xs) if len(xs)%2 > 0 else _median_even(xs)

median([1,10,2,9,5])
median([1,9,2,10])

print(median(num_friends))

def quantile(xs: List[float], p: float) -> float:
    p_index =  int(p*len(xs))
    return sorted(xs)[p_index]

print(quantile(num_friends, 0.10))
print(quantile(num_friends, 0.25))
print(quantile(num_friends, 0.75))
print(quantile(num_friends, 0.90))


def mode(xs: List[float]) -> List[float]:
    """Mode is the most common value in a list and since there can be
    more than one most common value, this function returns a list of
    all such common values."""
    counts = Counter(xs)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items() if count == max_count]


print(set(mode(num_friends)))


def data_range(xs: List[float]) -> float:
    """Range of data is the difference between the min and the max value"""
    return max(xs) -  min(xs)


print(data_range(num_friends))

def variance(xs: List[float]) -> float:
    mean = sum(xs)/len(xs) # first compute the mean of the data
    # Compute the sum of squared difference of each data point from the mean
    sum_squared_difference = sum([(x_i-mean)*(x_i-mean) for x_i in xs])
    return sum_squared_difference/(len(xs)-1)

print(variance(num_friends))
    
from math import sqrt
def standard_deviation(xs: List[float]) -> float:
    return sqrt(variance(xs))

print(standard_deviation(num_friends))

def interquartile_range(xs: List[float]) -> float:
    return quantile(xs, 0.75) - quantile(xs, 0.25)

print(interquartile_range(num_friends))

from scratch.linear_algebra import dot

def covariance(xs: List[float], ys: List[float]) -> float:
    """ A function to compute covariance between two vectors of same length"""
    
    assert len(xs) == len(ys),"Vectors must be of the same length"
    
    mean_xs = sum(xs)/len(xs)
    de_mean_xs = [x_i - mean_xs for x_i in xs]
    
    mean_ys = sum(ys)/len(ys)
    de_mean_ys = [y_i - mean_ys for y_i in ys]
    
    return dot(de_mean_xs,de_mean_ys)/(len(xs) - 1)

daily_minutes = [1,68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]

print(covariance(num_friends, daily_minutes))

def correlation(xs: List[float], ys: List[float]) -> float:
    """Measures how much xs and ys vary in tandem about their means"""
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)
    
    if stdev_x > 0 and stdev_y > 0:
        return covariance(xs,ys)/ stdev_x /stdev_y
    else:
        return 0
    
print(correlation(num_friends, daily_minutes))







