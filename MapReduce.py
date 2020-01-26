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
