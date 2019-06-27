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

  
