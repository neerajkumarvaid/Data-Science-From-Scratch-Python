from typing import List
from collections import Counter

def raw_majority_votes(labels: List[str]) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

assert raw_majority_votes(['a','b','c','d','b']) == 'b'
