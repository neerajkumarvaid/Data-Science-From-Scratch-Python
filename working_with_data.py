"""
Created on Sun Sep  7 09:54:40 2019
@author: Neeraj
Description: This code performs basic hypothesis testing in Python. 
Reference: Chapter 7 : Hypothesis and Inference
"""
from typing import List, Dict
from collections import Counter
import math
import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
    """Floor the point to the next lower multiple of bucket_size"""
    return bucket_size*math.floor(point/bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
    """Buckets the points and counts how many are in each bucket"""
    return Counter(bucketize(point, bucket_size) for point in points)
    
def plot_histogram(points: List[float], bucket_size: float, title: str):
    histogram = make_histogram(points, bucket_size)
    plt.bar(histogram.keys(), histogram.values(), width = bucket_size)
    plt.title(title)
    
# Let's use above functions to plot some histograms
import random
from probability import inverse_normal_cdf;

random.seed()
# sample 100 points uniformly between -100 and 100
uniform  = [200*random.random() - 100 for _ in range(100)]
# sample 100 points from normal distribution of mean 0 and std 57
normal = [57*inverse_normal_cdf(random.random()) for _ in range(10000)]

plot_histogram(uniform, 10, "Uniform Histogram")
plot_histogram(normal, 10, "Normal Distribution")
