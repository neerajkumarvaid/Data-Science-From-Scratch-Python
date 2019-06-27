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
    orlder_girl += 1
  if older == Kid.GIRL and younger == Kid.Girl:
    both_girls += 1
  if older == Kid.Girl or younger == Kid.Girl:
    either_girl += 1
    
print(f"P(both girls | older girl = {both_girls/older_girl})")
print(f"P(both girls | either girl = {both_girls/either_girl})")

# Create pdf of a uniform random variable
def uniform_pdf(x: float) -> float:
  """ A uniform random variable gives equal probability to values between 0 and 1.""""
  return 1 if  0 <= x < 1 else 0

# Create cdf of a uniform random variable
def uniform_cdf(x: float) -> float:
  if x <= 0: return 0 
  elif x < 1: return x
  else: return 1
  return 0 if x <= 0

import matplotlib.pylot as plt
plt.line(uniform_cdf)

