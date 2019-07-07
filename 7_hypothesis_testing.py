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
