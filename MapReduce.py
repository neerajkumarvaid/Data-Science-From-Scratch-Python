"""
Created on Sun Jan 26 16:51:40 2020

@author: Neeraj

Description: An impelementation of mapper and reducer functions with a few
examples in Python.
Reference: Chapter 25 MapReduce
"""

from typing import Iterable, Iterator, Tuple, List

def tokenize(document: str) -> List[str]:
    """Just split on whitespace"""
    return document.split()

def wc_mapper(document: str) -> Iterator[Tuple[str, int]]:
    """For each word in a document, emit (word, 1)"""
    for word in tokenize(document):
        yield (word, 1)

def wc_reducer(word: str,
              counts: Iterable[int]) -> Iterator[Tuple[str, int]]:
    """Sum up the counts for a word"""
    yield (word, sum(counts))

from collections import defaultdict

def word_count(documents: List[str]) -> List[Tuple[str, int]]:
    """Count the words in the input documents using MapReduce"""
    collector = defaultdict(list) # to store grouped values
    
    for document in documents:
        for word, count in wc_mapper(document):
            collector[word].append(count)
            
    return [output
           for word, counts in collector.items()
           for output in wc_reducer(word, counts)]

# Writing a general map_reduce function

from typing import Callable, Iterable, Any, Tuple

# A key/value pair is just a 2-tuple
KV = Tuple[Any, Any]

# A Mapper is a function that returns an Iterable of key/value pairs
Mapper = Callable[..., Iterable[KV]]

# A Reducer is a function that takes a key and an iterable of values
# and returns a key/value pair

Reducer = Callable[[Any, Iterable], KV]

def map_reduce(inputs: Iterable,
              mapper: Mapper,
              reducer: Reducer) -> List[KV]:
    """Run MapReduce on the inputs using mapper and reducer"""
    collector = defaultdict(list)
    
    for input in inputs:
        for key, value in mapper(input):
            collector[key].append(value)
    
    return [output
           for key, values in collector.items()
           for output in reducer(key, values)]

def values_reducer(values_fn: Callable) -> Reducer:
    """Return a reducer that just applies values_fn to its values"""
    def reduce(key, values: Iterable) -> KV:
        return (key, values_fn(values))         
    return reduce

sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values: len(set(values)))
