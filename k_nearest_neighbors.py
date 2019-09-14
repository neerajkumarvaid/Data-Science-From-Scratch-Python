"""
Created on Fri Sep 13 23:37:57 2019

@author: Neeraj

Description: Implemention of k-nearest neighbors algorithm from scratch in Python.

Reference: Chapter 12: k-Nearest Neighbors
"""

from typing import List
from collections import Counter

def raw_majority_votes(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

assert raw_majority_votes(['a','b','c','d','b']) == 'b'

def majority_vote(labels: List[str]) -> str:
    """Assumes that the lables are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values()
                      if count == winner_count])
    if num_winners == 1:
        return winner
    else:
        return majority_vote(labels[:-1]) # try again without the farthest
    
assert majority_vote(['a','b','c','b','a']) == 'b'

from typing import NamedTuple
from vector_operations import Vector, distance;

class LabeledPoint(NamedTuple):
    point: Vector
    label: str
        
def knn_classify(k: int,
                labeled_points: List[Vector],
                new_point: Vector) -> str:
    # Order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points, 
                         key = lambda lp: distance(lp.point, new_point))
    
    # Find the labels for the k closest
    k_nearest_labels = [lp.label for lp in by_distance[:k]]
    

    
    # and let them vote
    return majority_vote(k_nearest_labels)

# Example on the iris dataset
import requests
data = requests.get(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

with open('iris.csv','w') as f:
    f.write(data.text)
data.text

from typing import Dict
import csv
from collections import defaultdict

def parse_iris_row(row: List[str]) -> LabeledPoint:
    """sepal length, sepal width, petal length, petal width, class"""
    measurements = [float(value) for value in row[:-1]]
    # class is e.g. "Iris-virginica", we just want "virginica"
    label = row[-1].split("-")[-1]
    return LabeledPoint(measurements, label)

with open('iris.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    iris_data = []
    for row in csv_reader:
        iris_data.append(row)
    print(f'Processed {line_count} lines.')

iris_data = iris_data[:-2]

iris_data = [parse_iris_row(row) for row in iris_data]
#iris_data

# group the points by species for plotting
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
    points_by_species[iris.label].append(iris.point)
    
import matplotlib.pyplot as plt
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i,j) for i in range(4) for j in range(4) if i < j]
marks = ['+','.','x'] # 3 markers for our 3 classes

fig, ax = plt.subplots(2,3, figsize=(10, 5))

for row in range(2):
    for col in range(3):
        i, j = pairs[3*row + col]
        ax[row][col].set_title(f"{metrics[i]} vs. {metrics[j]}", fontsize = 8)
        ax[row][col].set_xticks([])
        ax[row][col].set_xticks([])
        
        for mark, (species, points) in zip(marks, points_by_species.items()):
            xs = [point[i] for point in points]
            ys = [point[j] for point in points]
            ax[row][col].scatter(xs,ys, marker = mark, label = species)
            
ax[-1][-1].legend(loc = 'lower right', prop = {'size':6})
plt.show()

import random
from machine_learning import split_data;
import math
random.seed(12)
iris_train, iris_test = split_data(iris_data, 0.70)

assert len(iris_train) == math.floor(0.7*len(iris_data))
assert len(iris_test) == math.ceil(0.3*len(iris_data))

from typing import Tuple

# track how many times we see (predicted, actual)

confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0
k = 5
for iris in iris_test:
    predicted = knn_classify(k, iris_train, iris.point)
    actual = iris.label
    
    if predicted == actual:
        num_correct += 1
    
    confusion_matrix[(predicted, actual)] += 1
    
pct_correct = num_correct/ len(iris_test)

print(pct_correct, confusion_matrix)

# curse of dimensionality
def random_point(dim: int) -> Vector:
    return [random.random() for _ in range(dim)]

def random_distance(dim: int, num_pairs: int) -> List[float]:
    return [distance(random_point(dim), random_point(dim)) 
            for _ in range(num_pairs)]
import tqdm
dimensions = range(1,101)
avg_distances = []
min_distances = []
random.seed(0)
for dim in tqdm.tqdm(dimensions, desc = "Curse of Dimensionality"):
    distances = random_distance(dim, 10000) # 10,000 random pairs
    avg_distances.append(sum(distances) / 10000) # track averages
    min_distances.append(min(distances)) # track minimums
    
plt.plot(dimensions, avg_distances)

min_avg_ratio = [min_dist / avg_dist 
                for min_dist, avg_dist in zip(min_distances, avg_distances)]
plt.plot(dimensions, avg_distances)
plt.plot(dimensions, min_avg_ratio)
