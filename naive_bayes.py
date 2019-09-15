
from typing import Set
import re

def tokenize(text: str) -> Set[str]:
    text = text.lower() # lowercase all the letter in text
    all_words = re.findall("[a-z0-9']+", text) # find all words containing alphabets, number and apostrophies
    # remove duplicates / extract unique words
    return set(all_words)

print(tokenize('Data Science is a science.'))
