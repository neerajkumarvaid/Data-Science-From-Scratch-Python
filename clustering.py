# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 07:27:33 2020
@author: Neeraj
Description: This file contains an impelementation fo k-means clustering from scratch in Python.
Reference: Chapter 20 Clustering
"""

from vector_operations import Vector

def num_differences(v1: Vector, v2: Vector) -> int:
    assert len(v1) == len(v2)
    return len([x1 for x1, x2 in zip(v1,v2) if x1 != x2])

print(f'num_differences([1,2,3],[2,1,3]) = {num_differences([1,2,3],[2,1,3])}')
print(f'num_differences([1,2],[1,2]) = {num_differences([1,2],[1,2])}')

from typing import List
from vector_operations import vector_mean
import random

def cluster_means(k: int,
                 imputs: List[Vector],
                 assignments: List[int]) -> List[Vector]:
    # cluster i contains the inputs whose assignment is i
    clusters = [[] for i in range(k)]
    
    for input, assignment in zip(inputs, assignments):
        clusters[assignment].append(input)
        
    # if cluster is empty then just use a random point
    return [vector_mean(cluster) if cluster else random.choice(inputs)
            for cluster in clusters]
