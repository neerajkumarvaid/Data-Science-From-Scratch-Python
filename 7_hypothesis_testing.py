#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 06:48:40 2019

@author: Neeraj
Description: This code performs basic hypothesis testing in Python. 
Reference: Chapter 6 : Hypothesis and Inference
"""
import os
os.chdir('/Users/apple/Documents/Courses/DSS')

from typing import Tuple
import math

def normal_approximation_binomial(n: int, p: float) -> Tuple[float,float]:
    """Estimates mu and sigma for a specified p and n"""
    mu = n*p
    sigma = math.sqrt(n*p*(1-p))
    return mu, sigma


# Import normal_cdf function from your code of chapter 6
from scratch.probability import normal_cdf 

#The normal cdf is the probability that a variable is below the threshold

normal_probability_below = normal_cdf

#It's above threshold if it's not below the threshold
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    return 1 - normal_cdf(lo, mu, sigma)

#It's in between if it is less than hi but greater than lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# It's outside if not in between
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    return 1 - normal_probability_between(lo, hi, mu, sigma)


from scratch.probability import inverse_normal_cdf 

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z<=z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """Returns the z for which P(Z>=z) = probability"""
    return inverse_normal_cdf(1-probability, mu, sigma)


def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """ Returns symmetric bounds (around the mean) that
    contains the specified probability"""
    tail_probability = (1-probability)/2
    
    # Upper bound should have tail probability above it
    upper_bound  = normal_upper_bound(tail_probability, mu, sigma)
    
     # Lower bound should have tail probability below it
    lower_bound  = normal_lower_bound(tail_probability, mu, sigma)
    
    return upper_bound, lower_bound

# Examples to run the code
mu_0, sigma_0 = normal_approximation_binomial(1000,0.5)
    

def two_sided_p_value(x: float,
                      mu: float = 0,
                      sigma: float = 1) -> float:
    if x >= mu:
        return 2*normal_probability_above(x, mu, sigma)
    else:
        return 2*normal_probability_below(x, mu, sigma)
    
two_sided_p_value(529.5, mu_0, sigma_0)   # 0.062

import random

extreme_value_count = 0
for _ in range(1000):
    num_heads = sum(1 if random.random() < 0.5 else 0    # Count # of heads
                    for _ in range(1000))                # in 1000 flips,
    if num_heads >= 530 or num_heads <= 470:             # and count how often
        extreme_value_count += 1                         # the # is 'extreme'

# p-value was 0.062 => ~62 extreme values out of 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

two_sided_p_value(531.5, mu_0, sigma_0)   # 0.0463


tspv = two_sided_p_value(531.5, mu_0, sigma_0)
assert 0.0463 < tspv < 0.0464

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upper_p_value(524.5, mu_0, sigma_0) # 0.061

upper_p_value(526.5, mu_0, sigma_0) # 0.047

p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)   # 0.0158

normal_two_sided_bounds(0.95, mu, sigma)        # [0.4940, 0.5560]

p_hat = 540 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000) # 0.0158
normal_two_sided_bounds(0.95, mu, sigma) # [0.5091, 0.5709]

    
