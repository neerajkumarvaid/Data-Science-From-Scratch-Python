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
