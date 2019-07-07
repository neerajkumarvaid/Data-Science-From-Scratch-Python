"""
Created on Mon Jun 26 00:55:07 2019
@author: Neeraj
Description: This code illustrates how to do basic porbability operations in Python. 
Reference: Chapter 6 : Probability
"""
import enum, random

class Kid(enum.Enum):
  BOY = 0
  GIRL = 1
  
def random_kid() -> Kid:
  return random.choice([Kid.BOY, Kid.GIRL])

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0) # Ensures reproducibility of the experiments

for _ in range(10000):
  younger = random_kid()
  older = random_kid()
  
  if older == Kid.GIRL:
    older_girl += 1
  if older == Kid.GIRL and younger == Kid.GIRL:
    both_girls += 1
  if older == Kid.GIRL or younger == Kid.GIRL:
    either_girl += 1
    
print(f"P(both girls | older girl = {both_girls/older_girl})")
print(f"P(both girls | either girl = {both_girls/either_girl})")


# Create pdf of a uniform random variable
def uniform_pdf(x: float) -> float:
  """A uniform random variable gives equal probability to values between 0 and 1."""
  return 1 if  0 <= x < 1 else 0

# Create cdf of a uniform random variable
def uniform_cdf(x: float) -> float:
  if x <= 0: return 0 
  elif x < 1: return x
  else: return 1

# Plot uniform pdf
import numpy as np
x = np.linspace(-0.5,1.5,21)

y = []

for i in x:
    y.append(uniform_pdf(i))

import matplotlib.pyplot as plt
plt.plot(x,y)

# Plot uniform cdf
y = []

for i in x:
    y.append(uniform_cdf(i))

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.axis([-0.5, 1.5, -0.5, 1.2])

# The normal distribution
from math import sqrt, pi, exp

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    num = exp(-(x-mu)**2 / 2 / sigma**2)
    den = sqrt(2*pi)*sigma
    return num/den

# Plot normal pdf
x = [x/10.0 for x in range(-50,50)]
y = [normal_pdf(i) for i in x]
plt.plot(x,y, '-', label ='mu = 0, sigma = 1')
plt.plot(x,[normal_pdf(i,0,2) for i in x], '--', label ='mu = 0, sigma = 2')
plt.plot(x,[normal_pdf(i,1,1) for i in x], '-.', label ='mu = 1, sigma = 1')

from math import erf
def normal_cdf(x:float, mu: float = 0, sigma: float = 1) -> float:
    return (1 + erf((x-mu)/sqrt(2) / sigma))/2

x = [x/10.0 for x in range(-50,50)]
plt.plot(x,[normal_cdf(i,0,1) for i in x], '-', label ='mu = 0, sigma = 1')
plt.plot(x,[normal_cdf(i,0,2) for i in x], '--', label ='mu = 0, sigma = 2')
plt.plot(x,[normal_cdf(i,1,1) for i in x], '-.', label ='mu = 1, sigma = 1')


# Inverse of normal distribution
def inverse_normal_cdf(p: float, 
                       mu: float = 0, 
                       sigma: float = 1,
                       tol: float = 0.01) -> float:
    """Computes a value from Z(0,1) at specified probability (p) level"""
    
    # Convert to standard normal distribution
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p)
    
    low_z = -10.0 # standard normal is 0 at cdf(-10)
    high_z = 10.0 # standard normal is 0 at cdf(10)
        
    # Use binary search to find the desired value
    while high_z - low_z > tol:
        midz = (high_z + low_z)/2
        pmid = normal_cdf(midz)
            
        if p > pmid:
            low_z = midz
        else:
            high_z = midz
        #print(midz)
    return midz


print(inverse_normal_cdf(0.3))      
        
import random

def bernoulli_trial(p: float) -> int:
    """Returns 1 with probability p and 0 with probability 1-p"""
    return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
    """Returns the sum of n bernoulli(p) trials"""
    return sum(bernoulli_trial(p) for _ in range(n))

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int) -> None:
    """Picks points from a Binomial(n, p) and plots their histogram"""
    data = [binomial(n, p) for _ in range(num_points)]

    # use a bar chart to show the actual binomial samples
    histogram = Counter(data)
    plt.bar([x - 0.4 for x in histogram.keys()],
            [v / num_points for v in histogram.values()],
            0.8,
            color='0.75')

    mu = p * n
    sigma = sqrt(n * p * (1 - p))

    # use a line chart to show the normal approximation
    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
          for i in xs]
    plt.plot(xs,ys)
    plt.title("Binomial Distribution vs. Normal Approximation")
