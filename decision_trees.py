import math
x = [i/100.0 for i in range(101) ]
entropy_x = [0] + [-p*math.log(p,2) for p in x if p > 0]
import matplotlib.pyplot as plt
plt.plot(x, entropy_x)
plt.xlabel('probability')
plt.ylabel('-p*log(p)')
plt.show()

from typing import List

def entropy(class_probabilities: List[float]) -> float:
    """Given a list of class probabilities, computes entropy."""
    return sum(-p*math.log(p,2) 
                for p in class_probabilities 
                if p > 0) # ignore zero probabilties
  
  
