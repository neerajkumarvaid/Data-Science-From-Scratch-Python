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
  
  
print(f"entropy([1.0]) = {entropy([1.0])}")
print(f"entropy([0.5, 0.5]) = {entropy([0.5, 0.5])}")
print(f"entropy([0.25, 0.75]) = {entropy([0.25, 0.75])}")

from typing import Any
from collections import Counter

def class_probabilities(labels: List[Any]) -> List[float]:
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]

def data_entropy(labels: List[Any]) -> float:
    return entropy(class_probabilities(labels))

print(f"data_entropy(['a']) = {data_entropy(['a'])}")
print(f"data_entropy([True, False]) = {data_entropy([True, False])}")
print(f"data_entropy([3,4,4,4]) = {data_entropy([3,4,4,4])}")

def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
              for subset in subsets)
from typing import NamedTuple, Optional

class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None # allow unlabeled data
        
                  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
         ]

from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T') # generic type for inputs

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
    """Partition the inputs into lists based on the specified attribute"""
    partitions: Dict[Any, List[T]] = defaultdict(list)
        
    for input in inputs:
        key = getattr(input, attribute) # value of the specified attribute
        partitions[key].append(input)
        
    return partitions

partition = partition_by(inputs, 'level')
print(partition.keys())
print(partition.values())

def partition_entropy_by(inputs: List[Any], 
                         attribute: str,
                         label_attribute: str) -> float:
    """Compute the entropy according to the given partition."""
    # partitions consists of our inputs
    partitions = partition_by(inputs, attribute)
    
    # but partition entropy just needs the class labels
    labels = [[getattr(input, label_attribute) for input in partition]
             for partition in partitions.values()]
    
    return partition_entropy(labels)
