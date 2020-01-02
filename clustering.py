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

import itertools
import tqdm
from vector_operations import squared_distance

class kMeans:
    def __init__(self, k: int) -> None:
        self.k = k # number of clusters
        self.means = None
        
    def classify(self, input: Vector)->int:
        """Return the index of the cluster closest to the input"""
        return min(range(self.k),
                   key = lambda i: squared_distance(input,self.means[i]))
    
    def train(self, inputs: List[Vector]) -> None:
        # Start with random assignments
        assignments = [random.randrange(self.k) for _ in inputs]
        #print(inputs[:10])
        #print(assignments[:10])
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute means and find new assignments
                self.means = cluster_means(self.k, inputs, assignments)
                #print(self.means)
                new_assignments = [self.classify(input) for input in inputs]
                
                # Check how many assignments changed and if we are done
                
                num_changed = num_differences(assignments, new_assignments)
                
                if num_changed == 0:
                    return
                # Otherwise keep the new assignments and compute new means
                assignments = new_assignments
                self.means = cluster_means(self.k, inputs, assignments)
                t.set_description(f"changed: {num_changed}/{len(inputs)}")
